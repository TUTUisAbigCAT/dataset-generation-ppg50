r"""Plot the PPG quality of generated vs real nights written by scripts/generate_ppg_nights.py.

  python scripts/plot_ppg_quality.py OUT        ->  OUT/quality_comparison.png

Row 1: pooled per-window distributions of every quality measure (generated vs real).
Row 2: per-participant median templatematch, share of windows below templatematch 0.8, and the
       median templatematch by sleep stage.
Rows 3-5: real and generated windows of the same epochs at the 10th, 50th and 90th percentile of
       the generated templatematch.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

MEASURES = ("templatematch", "disimilarity", "skewness", "kurtosis", "relative_power", "beats")
STAGES = (("W", (0,)), ("Light", (1, 2)), ("REM", (4,)), ("Deep", (3,)))
INK, BLUE, MUTED = "#52514e", "#2a78d6", "#9a9893"


def load(out):
    files = sorted((out / "subjects").glob("S*.npz"))
    if not files:
        raise SystemExit(f"No subjects/*.npz in {out}")
    data = []
    for path in files:
        with np.load(path) as z:
            if "quality_gen_templatematch" not in z.files:
                continue
            data.append(dict(sid=path.stem, stages=z["stages"], generated=z["generated_ppg"],
                             gen={m: z[f"quality_gen_{m}"] for m in MEASURES},
                             real={m: z[f"quality_real_{m}"] for m in MEASURES}))
    return data


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--raw-ppg-dir", type=Path, default=None,
                        help="extraction folder for the real example windows (default: from generation.json)")
    cli = parser.parse_args(argv)
    out = cli.output_dir.expanduser().resolve()
    data = load(out)
    pooled = {side: {m: np.concatenate([d[side][m] for d in data]) for m in MEASURES} for side in ("gen", "real")}

    fig = plt.figure(figsize=(18, 17), layout="constrained")
    grid = fig.add_gridspec(5, 6, height_ratios=[1, 1.1, 0.8, 0.8, 0.8])
    for k, m in enumerate(MEASURES):
        ax = fig.add_subplot(grid[0, k])
        values = [pooled[s][m][np.isfinite(pooled[s][m])] for s in ("real", "gen")]
        both = np.concatenate(values)
        if m == "disimilarity":
            both = both[both > 0]
        if both.size < 2 or not all(v.size for v in values):
            ax.set_title(f"{m} (no values)", fontsize=11)
            continue
        if m == "disimilarity":
            bins = np.logspace(np.log10(max(np.quantile(both, 0.005), 1e-6)), np.log10(np.quantile(both, 0.995)), 40)
            ax.set_xscale("log")
        else:
            bins = np.linspace(np.quantile(both, 0.005), np.quantile(both, 0.995), 40)
        for v, colour, label in zip(values, (INK, BLUE), ("real", "generated")):
            ax.hist(v, bins=bins, histtype="step", linewidth=2, color=colour, density=True,
                    label=f"{label} (median {np.median(v):.3g})")
        ax.set_title(m, fontsize=11)
        ax.legend(frameon=False, fontsize=8)
        ax.set_yticks([])

    sids = [d["sid"] for d in data]
    med = {s: np.array([np.nanmedian(d[s]["templatematch"]) for d in data]) for s in ("gen", "real")}
    low = {s: np.array([np.mean(~(d[s]["templatematch"] >= 0.8)) for d in data]) for s in ("gen", "real")}
    for column, (values, title, limits) in enumerate((
            (med, "Median templatematch per participant", (0.5, 1.0)),
            (low, "Share of windows with templatematch < 0.8", (0, 1)))):
        ax = fig.add_subplot(grid[1, column * 2:column * 2 + 2])
        ax.plot(limits, limits, color=MUTED, linewidth=1)
        ax.scatter(values["real"], values["gen"], s=25, color=BLUE)
        worst = np.argsort(-(np.abs(values["gen"] - values["real"])))[:6]
        for i in worst:
            ax.annotate(sids[i], (values["real"][i], values["gen"][i]), textcoords="offset points", xytext=(5, 2),
                        fontsize=8, color=INK)
        ax.set(title=title, xlabel="real", ylabel="generated", xlim=limits, ylim=limits)
    ax = fig.add_subplot(grid[1, 4:6])
    positions = np.arange(len(STAGES))
    for offset, side, colour in ((-0.18, "real", INK), (0.18, "gen", BLUE)):
        values = []
        for _, codes in STAGES:
            v = np.concatenate([d[side]["templatematch"][np.isin(d["stages"], codes)] for d in data])
            values.append(np.nanmedian(v) if np.isfinite(v).any() else np.nan)
        ax.bar(positions + offset, values, width=0.34, color=colour, label="real" if side == "real" else "generated")
    ax.set_xticks(positions, [s for s, _ in STAGES])
    ax.set(title="Median templatematch by stage", ylim=(0.5, 1.0))
    ax.legend(frameon=False)

    raw_dir = cli.raw_ppg_dir
    if raw_dir is None:
        import json
        raw_dir = Path(json.loads((out / "generation.json").read_text())["data_dir"])
    flat = [(d["sid"], k, d["gen"]["templatematch"][k]) for d in data for k in range(len(d["stages"]))
            if np.isfinite(d["gen"]["templatematch"][k])]
    flat.sort(key=lambda x: x[2])
    t = np.arange(1500) / 50.0
    by_sid = {d["sid"]: d for d in data}
    for row, q in enumerate((0.1, 0.5, 0.9)):
        sid, k, value = flat[int(q * (len(flat) - 1))]
        real_path = raw_dir / f"{sid}.npz"
        real = None
        if real_path.is_file():
            with np.load(real_path) as z:
                real = z["ppg"][k].astype(np.float64)
                real = (real - np.median(real)) / ((np.percentile(real, 99) - np.percentile(real, 1)) / 2 or 1.0)
        for column, (signal, colour, label, score) in enumerate((
                (real, INK, "real", by_sid[sid]["real"]["templatematch"][k]),
                (by_sid[sid]["generated"][k], BLUE, "generated", value))):
            ax = fig.add_subplot(grid[2 + row, column * 3:column * 3 + 3])
            if signal is not None:
                ax.plot(t[t < 15], np.clip(signal, -3, 3)[t < 15], color=colour, linewidth=1.1)
            ax.set_yticks([])
            ax.set_title(f"{sid} epoch {k} ({label}, templatematch {score:.3f}; generated at the {int(q * 100)}th "
                         f"percentile)", fontsize=10, loc="left")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    fig.suptitle(f"PPG quality, generated vs real ({len(data)} participants, {len(pooled['gen']['beats'])} windows)",
                 fontsize=14)
    target = out / "quality_comparison.png"
    fig.savefig(target, dpi=90)
    print(target)
    return target


if __name__ == "__main__":
    main()
