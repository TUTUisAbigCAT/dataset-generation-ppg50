r"""Whole-night PPG quality of generated vs real windows: all participants together and by sleep stage.

  python scripts/analyze_ppg_quality.py OUT                         # one generation folder
  python scripts/analyze_ppg_quality.py OUT_E050 OUT_E146           # compare checkpoints (same real windows)

Reads OUT/subjects/S*.npz written by scripts/generate_ppg_nights.py (per-window quality of every generated
window and of the real window of the same epoch) and reports, for all windows and for each stage group
(W, N1, N2, Light = N1+N2 as conditioned in training, Deep, REM):

  distributions   median and 10th/90th percentile of the six measures, generated vs real, W1 and KS distance;
  bands           templatematch >= 0.9 (good), 0.8-0.9 (fair), < 0.8 (poor), no score (fewer than 2 beats);
  same epoch      share of epochs where the generated window is poor and the real one is not ("added"),
                  and the reverse ("removed"), Cohen's kappa and Spearman correlation of templatematch;
  participants    median over participants of their share of poor windows (each night counts once);
  C2ST            classifier two-sample test: cross-validated AUC (participant folds) of a logistic regression
                  telling generated from real windows by the six measures - 0.5 = indistinguishable, 1 = separable.

Writes quality_analysis.json, quality_by_stage.csv and quality_by_stage.png into the first OUT (or --output)
and prints the main tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

MEASURES = ("templatematch", "disimilarity", "skewness", "kurtosis", "relative_power", "beats")
GROUPS = (("ALL", (0, 1, 2, 3, 4)), ("W", (0,)), ("N1", (1,)), ("N2", (2,)), ("Light", (1, 2)),
          ("Deep", (3,)), ("REM", (4,)))
GOOD, POOR = 0.9, 0.8
MIN_WINDOWS_PER_SUBJECT = 10
INK, MUTED, GRID = "#52514e", "#9a9893", "#e4e3df"
REAL_COLOR = INK
GEN_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def load(out):
    """Pooled per-window arrays of one generation folder."""
    files = sorted((Path(out) / "subjects").glob("S*.npz"))
    if not files:
        raise SystemExit(f"No subjects/S*.npz in {out}")
    parts = {k: [] for k in ("subject", "stage", "epoch_index")}
    for side in ("gen", "real"):
        for m in MEASURES:
            parts[f"{side}_{m}"] = []
    for path in files:
        with np.load(path) as z:
            n = z["stages"].size
            parts["subject"].append(np.full(n, path.stem))
            parts["stage"].append(z["stages"].astype(np.int64))
            parts["epoch_index"].append(z["epoch_index"].astype(np.int64))
            for side in ("gen", "real"):
                for m in MEASURES:
                    key = f"quality_{side}_{m}"
                    parts[f"{side}_{m}"].append(z[key].astype(np.float64) if key in z.files else np.full(n, np.nan))
    return {k: np.concatenate(v) for k, v in parts.items()}


def describe(values):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return dict(median=None, p10=None, p90=None, missing=float(1.0))
    p10, median, p90 = np.percentile(finite, [10, 50, 90])
    return dict(median=float(median), p10=float(p10), p90=float(p90), missing=float(1 - finite.size / values.size))


def distance(gen, real):
    from scipy.stats import ks_2samp, wasserstein_distance

    g, r = gen[np.isfinite(gen)], real[np.isfinite(real)]
    if g.size < 2 or r.size < 2:
        return dict(w1=None, ks=None)
    return dict(w1=float(wasserstein_distance(g, r)), ks=float(ks_2samp(g, r).statistic))


def bands(tm):
    n = tm.size
    return dict(good=float(np.mean(tm >= GOOD)), fair=float(np.mean((tm >= POOR) & (tm < GOOD))),
                poor=float(np.mean(tm < POOR)), no_score=float(np.mean(~np.isfinite(tm)))) if n else {}


def same_epoch(tm_gen, tm_real):
    """Agreement of poor (< 0.8 or no score) windows between generated and real windows of the same epochs."""
    from scipy.stats import spearmanr

    gen_poor = ~(tm_gen >= POOR)
    real_poor = ~(tm_real >= POOR)
    n = tm_gen.size
    if n == 0:
        return {}
    both_ok, added = np.mean(~gen_poor & ~real_poor), np.mean(gen_poor & ~real_poor)
    removed, both_poor = np.mean(~gen_poor & real_poor), np.mean(gen_poor & real_poor)
    observed = both_ok + both_poor
    expected = gen_poor.mean() * real_poor.mean() + (1 - gen_poor.mean()) * (1 - real_poor.mean())
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    finite = np.isfinite(tm_gen) & np.isfinite(tm_real)
    rho = float(spearmanr(tm_gen[finite], tm_real[finite]).statistic) if finite.sum() > 2 else None
    return dict(both_ok=float(both_ok), added=float(added), removed=float(removed), both_poor=float(both_poor),
                added_given_real_ok=float(added / (both_ok + added)) if both_ok + added > 0 else None,
                removed_given_real_poor=float(removed / (removed + both_poor)) if removed + both_poor > 0 else None,
                kappa=None if kappa is None else float(kappa), spearman=rho)


def participants(subject, tm_gen, tm_real):
    shares = []
    for sid in np.unique(subject):
        rows = subject == sid
        if rows.sum() >= MIN_WINDOWS_PER_SUBJECT:
            shares.append((float(np.mean(~(tm_gen[rows] >= POOR))), float(np.mean(~(tm_real[rows] >= POOR)))))
    if not shares:
        return dict(subjects=0)
    s = np.array(shares)
    return dict(subjects=int(len(s)), poor_median_gen=float(np.median(s[:, 0])), poor_median_real=float(np.median(s[:, 1])),
                poor_difference_median=float(np.median(s[:, 0] - s[:, 1])),
                subjects_more_poor_generated=float(np.mean(s[:, 0] > s[:, 1])))


def _features(values):
    """Six quality measures -> standardized linear and squared terms (skewed measures log-scaled)."""
    x = np.column_stack(values).astype(np.float64)
    x[:, MEASURES.index("disimilarity")] = np.log10(np.maximum(x[:, MEASURES.index("disimilarity")], 0) + 1e-6)
    k = MEASURES.index("kurtosis")
    x[:, k] = np.sign(x[:, k]) * np.log1p(np.abs(x[:, k]))
    return x


def _logistic(x, y, l2=1e-3, iterations=30):
    w = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
        gradient = x.T @ (p - y) + l2 * w
        hessian = (x * (p * (1 - p))[:, None]).T @ x + l2 * np.eye(x.shape[1])
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w


def _auc(score, y):
    from scipy.stats import rankdata

    ranks = rankdata(score)
    positives = y == 1
    n1, n0 = positives.sum(), (~positives).sum()
    return float((ranks[positives].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def c2st(data, rows, folds=5, seed=0):
    """Cross-validated AUC of generated-vs-real classification by quality measures, folds by participant."""
    gen = _features([data[f"gen_{m}"][rows] for m in MEASURES])
    real = _features([data[f"real_{m}"][rows] for m in MEASURES])
    subject = data["subject"][rows]
    x = np.vstack([gen, real])
    y = np.r_[np.ones(len(gen)), np.zeros(len(real))]
    groups = np.r_[subject, subject]
    keep = np.isfinite(x).all(axis=1)
    x, y, groups = x[keep], y[keep], groups[keep]
    sids = np.unique(groups)
    if len(sids) < folds or len(y) < 200:
        return dict(auc=None, windows=int(len(y)))
    order = np.random.default_rng(seed).permutation(len(sids))
    fold_of = {sid: int(order[i] % folds) for i, sid in enumerate(sids)}
    fold = np.array([fold_of[g] for g in groups])
    score = np.empty(len(y))
    for f in range(folds):
        train, test = fold != f, fold == f
        mean, std = x[train].mean(axis=0), x[train].std(axis=0)
        std = np.where(std > 0, std, 1.0)
        z_train, z_test = (x[train] - mean) / std, (x[test] - mean) / std
        design = lambda z: np.column_stack([np.ones(len(z)), z, z ** 2])
        w = _logistic(design(z_train), y[train])
        score[test] = design(z_test) @ w
    return dict(auc=_auc(score, y), windows=int(len(y)), generated_share=float(y.mean()))


def analyze(data):
    result = {}
    for name, codes in GROUPS:
        rows = np.isin(data["stage"], codes)
        if not rows.any():
            continue
        tm_gen, tm_real = data["gen_templatematch"][rows], data["real_templatematch"][rows]
        group = dict(windows=int(rows.sum()), subjects=int(np.unique(data["subject"][rows]).size), measures={})
        for m in MEASURES:
            g, r = data[f"gen_{m}"][rows], data[f"real_{m}"][rows]
            group["measures"][m] = dict(gen=describe(g), real=describe(r), **distance(g, r))
        group["bands"] = dict(gen=bands(tm_gen), real=bands(tm_real))
        group["same_epoch"] = same_epoch(tm_gen, tm_real)
        group["participants"] = participants(data["subject"][rows], tm_gen, tm_real)
        group["c2st"] = c2st(data, rows)
        result[name] = group
    return result


def flat_rows(label, result):
    rows = []
    for name, g in result.items():
        row = dict(run=label, group=name, windows=g["windows"], subjects=g["subjects"])
        for m, v in g["measures"].items():
            for side in ("real", "gen"):
                for k in ("median", "p10", "p90"):
                    row[f"{m}_{k}_{side}"] = v[side][k]
            row[f"{m}_no_value_gen"] = v["gen"]["missing"]
            row[f"{m}_no_value_real"] = v["real"]["missing"]
            row[f"{m}_w1"], row[f"{m}_ks"] = v["w1"], v["ks"]
        for side in ("real", "gen"):
            for k, value in g["bands"][side].items():
                row[f"templatematch_{k}_{side}"] = value
        row.update({f"same_epoch_{k}": v for k, v in g["same_epoch"].items()})
        row.update({f"participants_{k}": v for k, v in g["participants"].items()})
        row["c2st_auc"] = g["c2st"].get("auc")
        rows.append(row)
    return rows


def _fmt(value, digits=3):
    return "  -  " if value is None else f"{value:.{digits}f}"


def print_tables(label, result):
    print(f"\n=== {label}: generated vs real (real / generated) ===")
    print(f"{'group':6s} {'windows':>8s}  {'tm median':>13s}  {'tm<0.8':>13s}  {'tm>=0.9':>13s}  {'beats':>9s}  "
          f"{'skewness':>15s}  {'rel.power':>13s}  {'C2ST AUC':>8s}")
    for name, g in result.items():
        m, b = g["measures"], g["bands"]
        print(f"{name:6s} {g['windows']:8d}  "
              f"{_fmt(m['templatematch']['real']['median'])}/{_fmt(m['templatematch']['gen']['median'])}  "
              f"{_fmt(b['real']['poor'])}/{_fmt(b['gen']['poor'])}  {_fmt(b['real']['good'])}/{_fmt(b['gen']['good'])}  "
              f"{_fmt(m['beats']['real']['median'], 0):>4s}/{_fmt(m['beats']['gen']['median'], 0):<4s}  "
              f"{_fmt(m['skewness']['real']['median'])}/{_fmt(m['skewness']['gen']['median'])}  "
              f"{_fmt(m['relative_power']['real']['median'])}/{_fmt(m['relative_power']['gen']['median'])}  "
              f"{_fmt(g['c2st'].get('auc')):>8s}")
    print(f"\n{'group':6s} {'poor added':>11s} {'poor removed':>13s} {'kappa':>7s} {'spearman':>9s}   "
          f"{'per-participant poor share (real/gen)':s}")
    for name, g in result.items():
        s, p = g["same_epoch"], g["participants"]
        print(f"{name:6s} {_fmt(s.get('added_given_real_ok')):>11s} {_fmt(s.get('removed_given_real_poor')):>13s} "
              f"{_fmt(s.get('kappa')):>7s} {_fmt(s.get('spearman')):>9s}   "
              f"{_fmt(p.get('poor_median_real'))}/{_fmt(p.get('poor_median_gen'))}"
              f"  (generated poorer in {round((p.get('subjects_more_poor_generated') or 0) * p.get('subjects', 0))}"
              f" of {p.get('subjects', 0)} participants)")
    print("poor added = P(generated < 0.8 | real >= 0.8); poor removed = P(generated >= 0.8 | real < 0.8)")


def _bars(ax, names, series, title, *, ylim=None, reference=None, digits=2):
    width = 0.8 / len(series)
    for i, (label, color, values) in enumerate(series):
        x = np.arange(len(names)) + (i - (len(series) - 1) / 2) * width
        heights = [np.nan if v is None else v for v in values]
        ax.bar(x, heights, width * 0.92, color=color, label=label, zorder=2)
    if reference is not None:
        ax.axhline(reference, color=MUTED, lw=1, ls="--", zorder=1)
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_title(title, fontsize=10, color=INK)
    if ylim:
        ax.set_ylim(*ylim)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)


def plot(path, labels, results, datasets):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n, _ in GROUPS if n != "Light" and n in results[0]]
    stage_names = [n for n in names if n != "ALL"]
    fig = plt.figure(figsize=(18, 11))
    grid = fig.add_gridspec(3, 10, hspace=0.45, wspace=1.2)
    first = results[0]
    gen = [(f"generated {lab}", GEN_COLORS[i % len(GEN_COLORS)], r) for i, (lab, r) in enumerate(zip(labels, results))]

    def series(getter, with_real=True):
        out = [("real", REAL_COLOR, [getter(first[n], "real") for n in names])] if with_real else []
        return out + [(lab, color, [getter(r[n], "gen") if n in r else None for n in names]) for lab, color, r in gen]

    panels = [
        (lambda g, s: g["bands"][s]["poor"], "Share of windows with templatematch < 0.8", (0, 1), None, True),
        (lambda g, s: g["measures"]["beats"][s]["median"], "Median beats per 30 s", None, None, True),
        (lambda g, s: g["measures"]["skewness"][s]["median"], "Median skewness", None, 0.0, True),
        (lambda g, s: g["measures"]["relative_power"][s]["median"], "Median power ratio 1-2.25 / 0-8 Hz", (0, 1), None, True),
        (lambda g, s: g["c2st"].get("auc"), "C2ST AUC (0.5 = indistinguishable)", (0.4, 1), 0.5, False),
    ]
    for i, (getter, title, ylim, reference, with_real) in enumerate(panels):
        ax = fig.add_subplot(grid[0, 2 * i:2 * i + 2])
        _bars(ax, names, series(getter, with_real), title, ylim=ylim, reference=reference)
        _style(ax)
        if i == 0:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    codes = dict(GROUPS)
    for i, name in enumerate(stage_names):
        ax = fig.add_subplot(grid[1, 2 * i:2 * i + 2])
        rows = np.isin(datasets[0]["stage"], codes[name])
        curves = [("real", REAL_COLOR, datasets[0]["real_templatematch"][rows])]
        curves += [(lab, color, d["gen_templatematch"][np.isin(d["stage"], codes[name])])
                   for (lab, color, _), d in zip(gen, datasets)]
        for lab, color, values in curves:
            v = np.sort(values[np.isfinite(values)])
            if v.size:
                ax.plot(v, np.arange(1, v.size + 1) / v.size, color=color, lw=2, label=lab)
        ax.axvline(POOR, color=MUTED, lw=1, ls="--")
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0, 1)
        ax.set_title(f"{name}: templatematch, cumulative" if i == 0 else name, fontsize=10, color=INK)
        ax.set_xlabel("templatematch", fontsize=8, color=INK)
        if i == 0:
            ax.set_ylabel("share of windows at or below", fontsize=8, color=INK)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
        _style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for j, (key, title) in enumerate((("added_given_real_ok", "Same epoch: generated < 0.8 where real >= 0.8"),
                                      ("removed_given_real_poor", "Same epoch: generated >= 0.8 where real < 0.8"))):
        ax = fig.add_subplot(grid[2, 5 * j:5 * j + 5])
        _bars(ax, names, [(lab, color, [r[n]["same_epoch"].get(key) if n in r else None for n in names])
                          for lab, color, r in gen], title, ylim=(0, 1))
        _style(ax)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    windows = first["ALL"]["windows"]
    fig.suptitle(f"Generated vs real PPG quality, all nights and by sleep stage "
                 f"({first['ALL']['subjects']} participants, {windows} windows)", fontsize=13, color=INK)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outputs", nargs="+", type=Path, help="generation folders (OUT of generate_ppg_nights.py)")
    parser.add_argument("--labels", nargs="+", help="names of the folders in tables and figure (default: folder names)")
    parser.add_argument("--output", type=Path, help="where to write the results (default: the first folder)")
    return parser.parse_args(argv)


def main(argv=None):
    cli = parse_args(argv)
    labels = cli.labels or [p.name for p in cli.outputs]
    if len(labels) != len(cli.outputs):
        raise SystemExit("--labels needs one name per folder")
    datasets = [load(p) for p in cli.outputs]
    reference = datasets[0]
    for label, d in zip(labels[1:], datasets[1:]):
        same = (d["subject"].size == reference["subject"].size and np.array_equal(d["subject"], reference["subject"])
                and np.array_equal(d["epoch_index"], reference["epoch_index"]))
        if not same:
            print(f"warning: {label} does not cover the same participants/epochs as {labels[0]}; "
                  "each folder is compared with its own real windows", file=sys.stderr)
    results = [analyze(d) for d in datasets]
    out = cli.output or cli.outputs[0]
    out.mkdir(parents=True, exist_ok=True)
    report = dict(runs={lab: dict(folder=str(p), groups=r) for lab, p, r in zip(labels, cli.outputs, results)},
                  thresholds=dict(good=GOOD, poor=POOR), measures=list(MEASURES))
    (out / "quality_analysis.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    import csv

    rows = [row for lab, r in zip(labels, results) for row in flat_rows(lab, r)]
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "quality_by_stage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for lab, r in zip(labels, results):
        print_tables(lab, r)
    figure = plot(out / "quality_by_stage.png", labels, results, datasets)
    print(f"\nwritten: {out / 'quality_analysis.json'}, {out / 'quality_by_stage.csv'}, {figure}")
    return dict(results=results, figure=str(figure))


if __name__ == "__main__":
    main()
