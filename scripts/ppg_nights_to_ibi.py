r"""Generated and real PPG of a finished night generation -> IBI, and the comparison (CPU only, no model).

  python scripts/ppg_nights_to_ibi.py --nights NIGHTS/ppg50_1393726_e146 [--workers N]

Reads the generated windows in NIGHTS/subjects/S*.npz and the real windows of the same epochs from the
extraction folder named in NIGHTS/generation.json (both are only read). Every 30-s window, generated and
real alike, goes through ppg_extraction.ppg_to_ibi on its own (the windows are generated independently),
with --peak-method (default elgendi_local_fix: the current route + neurokit2's artifact correction, the best
of the routes compared against the ECG). Output in --output-dir (default NIGHTS/ibi_<peak method>):

  subjects/<SID>.npz     ibi_generated / ibi_real [n, 750] at 25 Hz, their status codes, stages, epoch_index
  subjects/<SID>.json    the participant's comparison
  night_npz/, night_npz_real/   generated and real nights in the WatchSleepNet NPZ format (unless --no-nights)
  ibi_summary.csv        one row per participant;  ibi_summary.json  pooled and per-stage results
  ibi_comparison.png     distributions, per-participant means and variability, event contrast, valid share

Compared (valid samples only, IBI > 0): the IBI distribution (W1, KS, mean, SD) overall and per stage; the
share of valid samples and of epochs with epoch_ok; per participant the mean IBI and the within-window SD of
IBI (beat-to-beat variability); the window-by-window mean IBI over the night (does the generated night follow
the real one); and the respiratory-event contrast (IBI during event samples minus elsewhere, same window).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import ppg_extraction as px

STAGES = (("ALL", (0, 1, 2, 3, 4)), ("W", (0,)), ("N1", (1,)), ("N2", (2,)), ("Light", (1, 2)), ("Deep", (3,)),
          ("REM", (4,)))
POOL_STEP = 25                      # pooled distributions keep one valid sample per second
STAGE_MAPPING = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
CSV_FIELDS = ("subject", "status", "epochs", "w1_ms", "ks", "mean_gen_ms", "mean_real_ms", "sd_gen_ms", "sd_real_ms",
              "valid_fraction_gen", "valid_fraction_real", "epoch_ok_gen", "epoch_ok_real",
              "within_window_sd_gen_ms", "within_window_sd_real_ms", "epoch_mean_correlation",
              "event_contrast_gen_ms", "event_contrast_real_ms", "event_windows",
              *[f"{s}_{k}" for s, _ in STAGES[1:] for k in ("w1_ms", "mean_gen_ms", "mean_real_ms")],
              "seconds", "error")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nights", type=Path, required=True, help="a generate_ppg_nights.py output folder")
    parser.add_argument("--data-dir", type=Path, help="extraction folder (default: the one in generation.json)")
    parser.add_argument("--output-dir", type=Path, help="default: NIGHTS/ibi_<peak method>")
    parser.add_argument("--peak-method", default="elgendi_local_fix", choices=px.ALL_PEAK_METHODS)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-nights", action="store_true", help="do not write the WatchSleepNet night NPZ files")
    return parser.parse_args(argv)


def windowed_ibi(windows, method):
    """ppg_to_ibi on every 30-s window on its own: (ibi [n, 750], status [n, 750], epoch_ok [n])."""
    windows = np.asarray(windows, dtype=np.float64)
    result = px.ppg_to_ibi(windows, px.FS_PPG, np.arange(len(windows)) * 1000.0, peak_method=method)
    return result["ibi"].astype(np.float32), result["status"].astype(np.uint8), result["epoch_ok"].astype(bool)


def distribution(gen, real):
    from scipy.stats import ks_2samp, wasserstein_distance

    g, r = gen[gen > 0], real[real > 0]
    if g.size < 2 or r.size < 2:
        return dict(w1_ms=None, ks=None, mean_gen_ms=None, mean_real_ms=None, sd_gen_ms=None, sd_real_ms=None,
                    samples_gen=int(g.size), samples_real=int(r.size))
    return dict(w1_ms=float(wasserstein_distance(g, r) * 1000), ks=float(ks_2samp(g, r).statistic),
                mean_gen_ms=float(g.mean() * 1000), mean_real_ms=float(r.mean() * 1000),
                sd_gen_ms=float(g.std() * 1000), sd_real_ms=float(r.std() * 1000),
                samples_gen=int(g.size), samples_real=int(r.size))


def window_stats(ibi, min_samples=125):
    """Per window: mean and SD of valid IBI (nan where fewer than min_samples valid samples)."""
    valid = ibi > 0
    count = valid.sum(axis=1)
    total = np.where(valid, ibi, 0).sum(axis=1)
    mean = np.where(count >= min_samples, total / np.maximum(count, 1), np.nan)
    sq = np.where(valid, (ibi - np.nan_to_num(mean)[:, None]) ** 2, 0).sum(axis=1)
    sd = np.where(count >= min_samples, np.sqrt(sq / np.maximum(count, 1)), np.nan)
    return mean, sd


def event_contrast(ibi, events25, min_samples=125):
    values = []
    for row, flag in zip(ibi, events25.astype(bool)):
        inside, outside = row[flag & (row > 0)], row[~flag & (row > 0)]
        if inside.size >= min_samples and outside.size >= min_samples:
            values.append((inside.mean() - outside.mean()) * 1000.0)
    return (float(np.mean(values)) if values else None), len(values)


def night_arrays(ibi, stages, epoch_index, apnea25, ahi, *, synthetic, epoch_start_s, method):
    return dict(data=np.asarray(ibi, np.float32).reshape(-1)[:, None],
                stages=np.repeat(stages.astype(np.int64), px.EPOCH_IBI), fs=np.array(px.FS_IBI),
                ahi=np.array(np.nan if ahi is None else float(ahi)), synthetic=np.array(bool(synthetic)),
                stage_mapping=np.array(json.dumps(STAGE_MAPPING, sort_keys=True)), epoch_index=epoch_index,
                apnea=apnea25.reshape(-1).astype(np.uint8), epoch_start_seconds=np.asarray(epoch_start_s),
                ibi_processing=np.array(f"each 30-s window through ppg_extraction.ppg_to_ibi on its own "
                                        f"(peak_method={method})"))


def convert_subject(sid, nights, data_dir, output, method, write_nights):
    started = time.time()
    with np.load(nights / "subjects" / f"{sid}.npz") as z:
        gen_ppg = z["generated_ppg"].astype(np.float64)
        gen_index = z["epoch_index"].astype(np.int64)
        stages = z["stages"].astype(np.int64)
        apnea25 = z["apnea_25hz"].astype(np.uint8)
    with np.load(data_dir / f"{sid}.npz") as z:
        real_index = z["epoch_index"].astype(np.int64)
        position = {int(e): k for k, e in enumerate(real_index)}
        rows = np.array([position[int(e)] for e in gen_index])              # KeyError: epochs do not match
        real_ppg = z["ppg"][rows].astype(np.float64)
        start_s = z["epoch_start_s"][rows]
        report = json.loads(str(z["report"])) if "report" in z.files else {}
    ibi_g, st_g, ok_g = windowed_ibi(gen_ppg, method)
    ibi_r, st_r, ok_r = windowed_ibi(real_ppg, method)
    mean_g, sd_g = window_stats(ibi_g)
    mean_r, sd_r = window_stats(ibi_r)
    both = np.isfinite(mean_g) & np.isfinite(mean_r)
    contrast_g, n_events = event_contrast(ibi_g, apnea25)
    contrast_r, _ = event_contrast(ibi_r, apnea25)
    metrics = dict(subject=sid, status="done", epochs=int(len(stages)), peak_method=method,
                   valid_fraction_gen=float((ibi_g > 0).mean()), valid_fraction_real=float((ibi_r > 0).mean()),
                   epoch_ok_gen=float(ok_g.mean()), epoch_ok_real=float(ok_r.mean()),
                   within_window_sd_gen_ms=float(np.nanmedian(sd_g) * 1000) if np.isfinite(sd_g).any() else None,
                   within_window_sd_real_ms=float(np.nanmedian(sd_r) * 1000) if np.isfinite(sd_r).any() else None,
                   epoch_mean_correlation=float(np.corrcoef(mean_g[both], mean_r[both])[0, 1]) if both.sum() > 10 else None,
                   event_contrast_gen_ms=contrast_g, event_contrast_real_ms=contrast_r, event_windows=n_events,
                   by_stage={})
    metrics.update(distribution(ibi_g, ibi_r))
    pooled = {}
    for name, codes in STAGES:
        rows_s = np.isin(stages, codes)
        if name != "ALL":
            metrics["by_stage"][name] = dict(epochs=int(rows_s.sum()), **distribution(ibi_g[rows_s], ibi_r[rows_s]))
        if name in ("ALL", "Light"):
            continue
        g, r = ibi_g[rows_s][:, ::POOL_STEP], ibi_r[rows_s][:, ::POOL_STEP]
        pooled[name] = (g[g > 0], r[r > 0])
    np.savez_compressed(output / "subjects" / f"{sid}.npz", ibi_generated=ibi_g, ibi_generated_status=st_g,
                        ibi_real=ibi_r, ibi_real_status=st_r, epoch_ok_generated=ok_g, epoch_ok_real=ok_r,
                        stages=stages, epoch_index=gen_index, apnea_25hz=apnea25,
                        peak_method=np.array(method))
    if write_nights:
        ahi = report.get("participant_ahi")
        for folder, ibi, synthetic in (("night_npz", ibi_g, True), ("night_npz_real", ibi_r, False)):
            np.savez_compressed(output / folder / f"{sid}.npz",
                                **night_arrays(ibi, stages, gen_index, apnea25, ahi, synthetic=synthetic,
                                               epoch_start_s=start_s, method=method))
    metrics["seconds"] = round(time.time() - started, 1)
    (output / "subjects" / f"{sid}.json").write_text(json.dumps(metrics, indent=1, default=float), encoding="utf-8")
    person = dict(mean_gen=metrics["mean_gen_ms"], mean_real=metrics["mean_real_ms"],
                  sd_gen=metrics["within_window_sd_gen_ms"], sd_real=metrics["within_window_sd_real_ms"],
                  contrast_gen=contrast_g, contrast_real=contrast_r)
    return sid, metrics, pooled, person


def _worker(task):
    sid = task[0]
    try:
        return convert_subject(*task)
    except Exception as error:
        return sid, dict(subject=sid, status="failed", error=str(error), traceback=traceback.format_exc()), {}, {}


def flat_row(m):
    row = {k: m.get(k) for k in CSV_FIELDS if not k.startswith(tuple(f"{s}_" for s, _ in STAGES[1:]))}
    for name, values in (m.get("by_stage") or {}).items():
        for k in ("w1_ms", "mean_gen_ms", "mean_real_ms"):
            row[f"{name}_{k}"] = values.get(k)
    return row


def summarize(results, pooled):
    from scipy.stats import ks_2samp, wasserstein_distance

    done = [m for m in results.values() if m.get("status") == "done"]
    med = lambda key: float(np.median([m[key] for m in done if m.get(key) is not None])) if done else None
    corr = lambda a, b: (float(np.corrcoef([m[a] for m in done if m.get(a) is not None and m.get(b) is not None],
                                            [m[b] for m in done if m.get(a) is not None and m.get(b) is not None])[0, 1])
                         if len(done) > 2 else None)
    summary = dict(participants=len(done), failed=sorted(s for s, m in results.items() if m.get("status") == "failed"),
                   per_participant=dict(w1_ms_median=med("w1_ms"), epoch_mean_correlation_median=med("epoch_mean_correlation"),
                                        valid_fraction_gen_median=med("valid_fraction_gen"),
                                        valid_fraction_real_median=med("valid_fraction_real"),
                                        epoch_ok_gen_median=med("epoch_ok_gen"), epoch_ok_real_median=med("epoch_ok_real"),
                                        within_window_sd_gen_ms_median=med("within_window_sd_gen_ms"),
                                        within_window_sd_real_ms_median=med("within_window_sd_real_ms"),
                                        mean_ibi_correlation=corr("mean_gen_ms", "mean_real_ms"),
                                        within_window_sd_correlation=corr("within_window_sd_gen_ms", "within_window_sd_real_ms"),
                                        event_contrast_gen_ms_median=med("event_contrast_gen_ms"),
                                        event_contrast_real_ms_median=med("event_contrast_real_ms"),
                                        event_contrast_correlation=corr("event_contrast_gen_ms", "event_contrast_real_ms")),
                   pooled={})
    names = [n for n, _ in STAGES if n != "Light"]
    for name in names:
        if name == "ALL":
            g = np.concatenate([np.concatenate([pooled[s][k][0] for k in pooled[s]]) for s in pooled if pooled[s]])
            r = np.concatenate([np.concatenate([pooled[s][k][1] for k in pooled[s]]) for s in pooled if pooled[s]])
        else:
            g = np.concatenate([pooled[s][name][0] for s in pooled if name in pooled[s]] or [np.zeros(0)])
            r = np.concatenate([pooled[s][name][1] for s in pooled if name in pooled[s]] or [np.zeros(0)])
        if g.size > 1 and r.size > 1:
            summary["pooled"][name] = dict(w1_ms=float(wasserstein_distance(g, r) * 1000), ks=float(ks_2samp(g, r).statistic),
                                           median_gen_ms=float(np.median(g) * 1000), median_real_ms=float(np.median(r) * 1000),
                                           mean_gen_ms=float(g.mean() * 1000), mean_real_ms=float(r.mean() * 1000),
                                           sd_gen_ms=float(g.std() * 1000), sd_real_ms=float(r.std() * 1000))
    return summary


def print_summary(summary):
    f = lambda v, d=1: "  -  " if v is None else f"{v:.{d}f}"
    print(f"\n=== IBI, generated vs real: {summary['participants']} participants (pooled, valid samples) ===")
    print(f"{'stage':6s} {'W1 ms':>7s} {'KS':>6s} {'median real/gen ms':>20s} {'SD real/gen ms':>16s}")
    for name, v in summary["pooled"].items():
        print(f"{name:6s} {f(v['w1_ms']):>7s} {f(v['ks'], 3):>6s} {f(v['median_real_ms'], 0):>9s} / {f(v['median_gen_ms'], 0):<8s}"
              f" {f(v['sd_real_ms'], 0):>7s} / {f(v['sd_gen_ms'], 0):<6s}")
    p = summary["per_participant"]
    print(f"\nper participant (medians): W1 {f(p['w1_ms_median'])} ms | valid share real/gen "
          f"{f(p['valid_fraction_real_median'], 3)}/{f(p['valid_fraction_gen_median'], 3)} | epoch_ok real/gen "
          f"{f(p['epoch_ok_real_median'], 3)}/{f(p['epoch_ok_gen_median'], 3)}")
    print(f"mean IBI across participants: r = {f(p['mean_ibi_correlation'], 2)} | within-window SD real/gen "
          f"{f(p['within_window_sd_real_ms_median'])}/{f(p['within_window_sd_gen_ms_median'])} ms (r = "
          f"{f(p['within_window_sd_correlation'], 2)})")
    print(f"window-by-window mean IBI over the night, real vs generated: median r = {f(p['epoch_mean_correlation_median'], 2)}")
    print(f"event contrast (IBI during events - elsewhere): real {f(p['event_contrast_real_ms_median'])} ms, "
          f"generated {f(p['event_contrast_gen_ms_median'])} ms (r across participants = {f(p['event_contrast_correlation'], 2)})")


def plot(path, summary, pooled, people):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID, REAL, GEN = "#52514e", "#9a9893", "#e4e3df", "#52514e", "#2a78d6"

    def style(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=INK, labelsize=9)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    ax = axes[0, 0]
    g = np.concatenate([np.concatenate([pooled[s][k][0] for k in pooled[s]]) for s in pooled if pooled[s]])
    r = np.concatenate([np.concatenate([pooled[s][k][1] for k in pooled[s]]) for s in pooled if pooled[s]])
    bins = np.linspace(0.4, 1.6, 121)
    ax.hist(r, bins, density=True, histtype="step", lw=2, color=REAL, label="real")
    ax.hist(g, bins, density=True, histtype="step", lw=2, color=GEN, label="generated")
    ax.set_title(f"IBI, all valid samples (W1 {summary['pooled']['ALL']['w1_ms']:.1f} ms)", color=INK)
    ax.set_xlabel("IBI (s)", color=INK)
    ax.legend(frameon=False, labelcolor=INK)
    style(ax)
    ax = axes[0, 1]
    names = [n for n in ("W", "N1", "N2", "Deep", "REM") if n in summary["pooled"]]
    data_r = [np.concatenate([pooled[s][n][1] for s in pooled if n in pooled[s]]) for n in names]
    data_g = [np.concatenate([pooled[s][n][0] for s in pooled if n in pooled[s]]) for n in names]
    x = np.arange(len(names))
    for offset, data, color in ((-0.18, data_r, REAL), (0.18, data_g, GEN)):
        q = np.array([np.percentile(d, [10, 25, 50, 75, 90]) for d in data])
        ax.vlines(x + offset, q[:, 0], q[:, 4], color=color, lw=1.5)
        ax.bar(x + offset, q[:, 3] - q[:, 1], 0.3, bottom=q[:, 1], color=color, alpha=0.35, edgecolor=color)
        ax.scatter(x + offset, q[:, 2], color=color, s=40, zorder=3)
    ax.set_xticks(x, names)
    ax.set_title("IBI by stage: median, 25-75 % box, 10-90 % line (grey real, blue generated)", color=INK, fontsize=10)
    ax.set_ylabel("IBI (s)", color=INK)
    style(ax)
    for ax, (a, b, title, unit) in zip((axes[0, 2], axes[1, 0], axes[1, 1]), (
            ("mean_real", "mean_gen", "Mean IBI per participant", "ms"),
            ("sd_real", "sd_gen", "Within-window IBI SD per participant", "ms"),
            ("contrast_real", "contrast_gen", "Event contrast per participant (IBI events - elsewhere)", "ms"))):
        pairs = np.array([(p[a], p[b]) for p in people.values() if p.get(a) is not None and p.get(b) is not None])
        if len(pairs):
            lo, hi = pairs.min(), pairs.max()
            pad = (hi - lo) * 0.05 + 1e-6
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, ls="--", lw=1)
            ax.scatter(pairs[:, 0], pairs[:, 1], s=28, color=GEN, alpha=0.85, edgecolor="white", lw=0.5)
            rr = np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1] if len(pairs) > 2 else np.nan
            ax.set_title(f"{title} (r = {rr:.2f})", color=INK, fontsize=10)
        ax.set_xlabel(f"real ({unit})", color=INK)
        ax.set_ylabel(f"generated ({unit})", color=INK)
        style(ax)
    ax = axes[1, 2]
    p = summary["per_participant"]
    labels = ["valid samples", "epoch_ok"]
    real_v = [p["valid_fraction_real_median"], p["epoch_ok_real_median"]]
    gen_v = [p["valid_fraction_gen_median"], p["epoch_ok_gen_median"]]
    xx = np.arange(2)
    ax.bar(xx - 0.18, real_v, 0.34, color=REAL, label="real")
    ax.bar(xx + 0.18, gen_v, 0.34, color=GEN, label="generated")
    for xi, v in zip(xx - 0.18, real_v):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", color=INK, fontsize=9)
    for xi, v in zip(xx + 0.18, gen_v):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_xticks(xx, labels)
    ax.set_ylim(0, 1.08)
    ax.set_title("Share usable, median over participants", color=INK, fontsize=10)
    style(ax)
    fig.suptitle(f"PPG -> IBI, generated vs real ({summary['participants']} participants)", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    cli = parse_args(argv)
    nights = cli.nights.expanduser().resolve()
    manifest = json.loads((nights / "generation.json").read_text(encoding="utf-8"))
    data_dir = (cli.data_dir or Path(manifest["data_dir"])).expanduser().resolve()
    output = (cli.output_dir or nights / f"ibi_{cli.peak_method}").expanduser().resolve()
    for folder in ("subjects",) + (() if cli.no_nights else ("night_npz", "night_npz_real")):
        (output / folder).mkdir(parents=True, exist_ok=True)
    sids = sorted(p.stem for p in (nights / "subjects").glob("S*.npz"))
    if cli.subjects:
        sids = [s for s in sids if s in set(cli.subjects)]
    print(json.dumps(dict(nights=str(nights), data_dir=str(data_dir), output=str(output), participants=len(sids),
                          peak_method=cli.peak_method, workers=cli.workers,
                          checkpoint_epoch=manifest.get("checkpoint", {}).get("epoch"))), flush=True)
    tasks = [(sid, nights, data_dir, output, cli.peak_method, not cli.no_nights) for sid in sids]
    results, pooled, people = {}, {}, {}

    def record(sid, metrics, pool, person):
        results[sid], pooled[sid], people[sid] = metrics, pool, person
        print(json.dumps({k: metrics.get(k) for k in ("subject", "status", "w1_ms", "mean_real_ms", "mean_gen_ms",
                                                      "epoch_mean_correlation", "seconds", "error")}, default=float),
              flush=True)

    if cli.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(cli.workers, len(tasks))) as pool:
            for future in as_completed([pool.submit(_worker, t) for t in tasks]):
                record(*future.result())
    else:
        for task in tasks:
            record(*_worker(task))
    with (output / "ibi_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for sid in sorted(results):
            writer.writerow(flat_row(results[sid]))
    pooled = {s: p for s, p in pooled.items() if p}
    summary = summarize(results, pooled)
    summary.update(peak_method=cli.peak_method, nights=str(nights), checkpoint_epoch=manifest.get("checkpoint", {}).get("epoch"))
    (output / "ibi_summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print_summary(summary)
    if pooled:
        plot(output / "ibi_comparison.png", summary, pooled, {s: people[s] for s in pooled})
    print(f"\nwritten: {output / 'ibi_summary.csv'}, {output / 'ibi_summary.json'}, {output / 'ibi_comparison.png'}",
          flush=True)
    return summary


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
