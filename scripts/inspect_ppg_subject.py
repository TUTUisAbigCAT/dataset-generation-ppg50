"""Why do PPG and ECG beats disagree for one participant? Text summary + one figure.

Prints, for an extracted participant (scripts/extract_ppg.py output):
  * the raw ECG / BVP channels (finite fraction, percentiles);
  * beats and IBI distribution of every detector (PPG 50->100 Hz, PPG 100 Hz, WatchSleepNet, ECG);
  * heart-rate cross-correlation lag of PPG against ECG for every hour (alignment or clock drift);
  * the stored alignment and accuracy report.
Writes <out>/inspect_<SID>.png: IBI histograms, 5-min heart rate over the night, hourly lag, the
delay of every PPG beat after the preceding R peak (flat at 0.2-0.4 s if E4 and PSG share one
clock), and three 20-s windows (cleaned PPG with its peaks, cleaned ECG with its R peaks).

  python scripts/inspect_ppg_subject.py --subject S002 \
      --raw-dir /home/yxtao/links/scratch/dreamt/data_100Hz \
      --npz-dir /home/yxtao/links/scratch/WatchSleepProcessedDataset/DREAMT22_PPG50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import ppg_extraction as px

INK, BLUE, ORANGE, GREEN = "#52514e", "#2a78d6", "#eb6834", "#1baf7a"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--npz-dir", type=Path, required=True, help="output folder of extract_ppg.py")
    parser.add_argument("--out", type=Path, default=None, help="folder for the PNG (default: --npz-dir)")
    parser.add_argument("--ecg-column", default="ECG")
    return parser.parse_args(argv)


def ibi_summary(name, beats):
    beats = np.sort(np.asarray(beats, dtype=np.float64))
    ibi = np.diff(beats)
    if not ibi.size:
        return f"{name:16s} beats {beats.size:6d}"
    ok = (ibi >= 0.3) & (ibi < 2.0)
    q = np.quantile(ibi[ok], [0.1, 0.5, 0.9]) if ok.any() else [np.nan] * 3
    return (f"{name:16s} beats {beats.size:6d}  IBI p10/p50/p90 {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f} s  "
            f"<0.3 s {np.mean(ibi < 0.3):.3f}  >=2.0 s {np.mean(ibi >= 2.0):.3f}")


def hourly_lags(ecg_t, ppg_t, *, block_s=3600.0, max_lag_s=120.0):
    rows = []
    if not ecg_t.size or not ppg_t.size:
        return rows
    for lo in np.arange(max(ecg_t[0], ppg_t[0]), min(ecg_t[-1], ppg_t[-1]) - block_s / 2, block_s):
        e = ecg_t[(ecg_t >= lo) & (ecg_t < lo + block_s)]
        p = ppg_t[(ppg_t >= lo - max_lag_s) & (ppg_t < lo + block_s + max_lag_s)]
        lag, corr = px.hr_cross_lag(e, p, max_lag_s=max_lag_s)
        rows.append((lo, e.size, p.size, lag, corr))
    return rows


def heart_rate(beats, edges):
    beats = np.sort(np.asarray(beats, dtype=np.float64))
    ibi, mid = np.diff(beats), (beats[1:] + beats[:-1]) / 2
    ok = (ibi >= 0.3) & (ibi < 2.0)
    out = np.full(edges.size - 1, np.nan)
    index = np.digitize(mid[ok], edges) - 1
    for k in range(out.size):
        values = ibi[ok][index == k]
        if values.size >= 20:
            out[k] = 60.0 / np.median(values)
    return out


def main(argv=None):
    import neurokit2 as nk
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cli = parse_args(argv)
    sid = cli.subject.strip()
    csv = sorted(list(cli.raw_dir.glob(f"{sid}_*.csv")) + list(cli.raw_dir.glob(f"{sid}.csv")))
    if not csv:
        raise SystemExit(f"No raw CSV for {sid} in {cli.raw_dir}")
    npz_path = cli.npz_dir / f"{sid}.npz"
    with np.load(npz_path) as saved:
        z = {key: saved[key] for key in saved.files}
    report = json.loads(str(z["report"]))
    frame = px.read_subject_csv(csv[0], cli.ecg_column)
    print(f"== {sid}  {csv[0].name}  rows {len(frame)} ({len(frame) / 360000:.2f} h)  columns {list(frame.columns)}")
    for column in ("BVP", cli.ecg_column):
        if column not in frame.columns:
            print(f"{column}: missing")
            continue
        values = frame[column].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        q = np.nanquantile(values, [0.01, 0.5, 0.99]) if finite.any() else [np.nan] * 3
        print(f"{column:5s} finite {finite.mean():.3f}  p1/p50/p99 {q[0]:.4g}/{q[1]:.4g}/{q[2]:.4g}  "
              f"distinct values {np.unique(values[finite][:200000]).size}")
    print(f"epochs {z['stages'].size}  stage changes on grid {report['stage_changes_on_grid_fraction']:.3f}  "
          f"first epoch at {z['epoch_start_s'][0]:.1f} s  epoch_ok {z['epoch_ok'].mean():.3f}")
    ecg_t, ppg_t = z["ecg_peaks_s"], z["ppg_peaks_s"]
    for name, key in (("PPG 50->100 Hz", "ppg_peaks_s"), ("PPG 100 Hz", "ppg100_peaks_s"),
                      ("WatchSleepNet", "wsn_peaks_s"), ("ECG", "ecg_peaks_s")):
        print(ibi_summary(name, z[key]))
    print("hourly heart-rate cross-correlation (PPG minus ECG lag, s):")
    for lo, ne, npg, lag, corr in hourly_lags(ecg_t, ppg_t):
        lag_text = "  n/a" if lag is None else f"{lag:+7.2f}"
        corr_text = "  n/a" if corr is None else f"{corr:.2f}"
        print(f"  hour from {lo / 3600:5.2f} h  ECG beats {ne:5d}  PPG beats {npg:5d}  lag {lag_text}  corr {corr_text}")
    print("alignment:", json.dumps(report.get("alignment"), default=float))
    for name, metrics in (report.get("beat_accuracy") or {}).items():
        print(f"beats  {name:18s}", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()}))
    for name, metrics in (report.get("series_accuracy") or {}).items():
        print(f"series {name:18s}", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()}))

    # ---- figure
    fig = plt.figure(figsize=(14, 16), layout="constrained")
    grid = fig.add_gridspec(6, 2, height_ratios=[1.1, 1.1, 1.1, 1, 1, 1])
    ax = fig.add_subplot(grid[0, 0])
    bins = np.arange(0.2, 2.1, 0.02)
    for name, key, colour in (("ECG", "ecg_peaks_s", INK), ("PPG 50->100 Hz", "ppg_peaks_s", BLUE),
                              ("WatchSleepNet", "wsn_peaks_s", ORANGE)):
        ibi = np.diff(np.sort(z[key]))
        ax.hist(ibi[(ibi > 0.2) & (ibi < 2.1)], bins=bins, histtype="step", linewidth=1.5, color=colour,
                label=name, density=True)
    ax.set(title="IBI distribution over the night", xlabel="IBI (s)", ylabel="density")
    ax.legend(frameon=False)
    ax = fig.add_subplot(grid[0, 1])
    edges = np.arange(0, len(frame) / 100 + 300, 300.0)
    centres = (edges[:-1] + edges[1:]) / 2 / 3600
    ax.plot(centres, heart_rate(ecg_t, edges), color=INK, linewidth=2, label="ECG")
    ax.plot(centres, heart_rate(ppg_t, edges), color=BLUE, linewidth=2, label="PPG 50->100 Hz")
    ax.set(title="Median heart rate per 5 min", xlabel="hours from recording start", ylabel="beats / min")
    ax.legend(frameon=False)
    ax = fig.add_subplot(grid[1, :])
    lags = hourly_lags(ecg_t, ppg_t)
    if lags:
        ax.plot([r[0] / 3600 for r in lags], [np.nan if r[3] is None else r[3] for r in lags], "o-",
                color=BLUE, linewidth=2, markersize=8)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set(title="Hourly heart-rate cross-correlation lag (PPG minus ECG)", xlabel="hours", ylabel="lag (s)")

    # every PPG beat minus the R peak before it: a flat band at 0.2-0.4 s means one shared clock
    ppg100 = np.sort(z["ppg100_peaks_s"])
    before = np.searchsorted(ecg_t, ppg100, side="right") - 1
    ok = before >= 0
    delay, at = ppg100[ok] - ecg_t[before[ok]], ppg100[ok]
    near = delay < 2.0
    sleep_epochs = np.flatnonzero(z["stages"] > 0)
    centre = z["epoch_start_s"][sleep_epochs[sleep_epochs.size // 2]] if sleep_epochs.size else at.mean()
    for column, (lo, hi, label) in enumerate(((at.min(), at.max(), "whole night"),
                                              (centre - 300, centre + 300, "10 min in mid-sleep"))):
        ax = fig.add_subplot(grid[2, column])
        m = near & (at >= lo) & (at <= hi)
        ax.axhspan(0.2, 0.4, color=INK, alpha=0.08, linewidth=0)
        scale = 3600.0 if column == 0 else 60.0
        ax.plot(at[m] / scale, delay[m], ".", markersize=1 if column == 0 else 3, color=BLUE)
        ax.set(title=f"PPG beat minus preceding R peak, {label}", ylim=(0, 2.0),
               xlabel="hours" if column == 0 else "minutes", ylabel="delay (s); grey = one clock")

    sleep = np.flatnonzero(z["stages"] > 0)
    picks = sleep[np.linspace(0, sleep.size - 1, 5).astype(int)[1:4]] if sleep.size >= 5 else np.arange(min(3, z["stages"].size))
    bvp, _ = px._fill_nan(frame["BVP"].to_numpy(dtype=np.float64))
    ecg_values = frame[cli.ecg_column].to_numpy(dtype=np.float64) if cli.ecg_column in frame.columns else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ppg_clean = np.asarray(nk.ppg_clean(bvp, sampling_rate=100, method="elgendi"))
        if ecg_values is not None:
            ecg_clean = np.asarray(nk.ecg_clean(px._fill_nan(ecg_values)[0], sampling_rate=100))
            if report.get("alignment", {}).get("ecg_inverted"):
                ecg_clean = -ecg_clean
    for row, epoch in enumerate(picks):
        start = z["epoch_start_s"][epoch] + 5.0
        t = np.arange(int(start * 100), int((start + 20) * 100)) / 100
        rows = (t * 100).astype(int)
        ax = fig.add_subplot(grid[3 + row, :])
        y = ppg_clean[rows]
        ax.plot(t, (y - y.mean()) / (y.std() or 1) + 3, color=BLUE, linewidth=1.2)
        for beats, colour, level in ((ppg_t, BLUE, 5.6), (z["wsn_peaks_s"], ORANGE, 6.4)):
            inside = beats[(beats >= t[0]) & (beats <= t[-1])]
            ax.plot(inside, np.full(inside.size, level), "v", color=colour, markersize=8)
        if ecg_values is not None:
            e = ecg_clean[rows]
            ax.plot(t, (e - e.mean()) / (e.std() or 1) - 3, color=INK, linewidth=1.0)
            inside = ecg_t[(ecg_t >= t[0]) & (ecg_t <= t[-1])]
            ax.plot(inside, np.full(inside.size, 1.0), "^", color=INK, markersize=8)
        stage = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}[int(z["stages"][epoch])]
        ax.set(title=f"{stage} epoch at {start / 3600:.2f} h: cleaned PPG (top, blue = our peaks, orange = "
                     f"WatchSleepNet) and cleaned ECG (bottom, R peaks)", xlabel="s", yticks=[])
    out = (cli.out or cli.npz_dir) / f"inspect_{sid}.png"
    fig.suptitle(f"{sid}: PPG vs ECG beats", fontsize=14)
    fig.savefig(out, dpi=110)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
