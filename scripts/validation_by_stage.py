r"""Validation windows of training runs, generated vs real, by sleep stage (reads runs; writes a new folder only).

  python scripts/validation_by_stage.py RUN_DIR [RUN_DIR ...] [--epochs 10 50 100 150 best last] [--output-dir DIR]

Every epoch, training generates the run's validation windows (validation/epoch_XXX.npz: generated, real, owners)
but scores them only by participant. This rebuilds the run's validation windows from its config.json (same
cohort, same deterministic choice; checked against the saved real windows), so each window has its stage, and
derives the IBI of the generated and the real windows the way training does (ppg_extraction.ppg_to_ibi). Per run,
epoch and stage (W, N1, N2, Deep, REM, all):

  windows; W1 of the 25 Hz IBI values (ms); mean heart rate; median SDNN and RMSSD of the beat intervals in a
  window (ms; windows with >= 5 beats), generated and real

and per participant at each epoch: mean heart rate and median RMSSD, generated and real. Writes
stage_metrics.csv, subject_metrics.csv and summary.txt (also printed: each run at its last chosen epoch, and all
runs pooled at the last epoch they all have). Needs the run's data_dir; CPU only, a few minutes per run.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ppg_training as pt

STAGES = {0: "W", 1: "N1", 2: "N2", 3: "Deep", 4: "REM"}
DEFAULT_OUTPUT = Path("/home/yxtao/links/scratch/dataset-generation-ppg50/validation_by_stage")
MIN_BEATS = 5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="run folders (with config.json and validation/)")
    parser.add_argument("--epochs", nargs="+", default=["10", "50", "100", "150", "best", "last"],
                        help="epoch numbers, 'best' (status.json) and/or 'last' (latest validation file)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=f"new folder (default: a new timestamped folder under {DEFAULT_OUTPUT})")
    return parser.parse_args(argv)


def beat_intervals(ibi_25hz):
    """Beat intervals (s) of one window's 25 Hz IBI series: one value per run of a constant positive value."""
    v = np.asarray(ibi_25hz, dtype=np.float64)
    if v.size == 0:
        return v
    starts = np.r_[True, v[1:] != v[:-1]]
    values = v[starts]
    return values[values > 0]


def window_hrv(ibi):
    """Per window [N, L] -> (heart rate bpm, SDNN ms, RMSSD ms), NaN where fewer than MIN_BEATS beats."""
    out = np.full((len(ibi), 3), np.nan)
    for i, row in enumerate(ibi):
        beats = beat_intervals(row) * 1000.0
        if beats.size >= MIN_BEATS:
            out[i] = 60000.0 / beats.mean(), beats.std(), np.sqrt(np.mean(np.diff(beats) ** 2))
    return out


def w1_ms(a, b):
    from scipy.stats import wasserstein_distance

    a, b = a[a > 0], b[b > 0]
    return float(wasserstein_distance(a, b) * 1000.0) if a.size and b.size else float("nan")


def rebuild_validation(run):
    """(config, stages per validation window, owners) of a run, as training chose them."""
    config = pt.load_config(run / "config.json")
    trainer = pt.Trainer(config, run, device="cpu", log=lambda *_: None)
    windows = trainer.validation_windows
    stages = np.array([int(trainer.cohort.records[sid]["stages"][first]) for sid, first in windows])
    real = np.stack([trainer.cohort.window_array(sid, first)[:, 0] for sid, first in windows])
    return config, stages, [sid for sid, _ in windows], real


def chosen_epochs(run, wanted):
    files = {int(p.stem.split("_")[1]): p for p in (run / "validation").glob("epoch_*.npz")}
    if not files:
        raise SystemExit(f"{run}: no validation/epoch_*.npz")
    status = json.loads((run / "status.json").read_text()) if (run / "status.json").is_file() else {}
    epochs = []
    for item in wanted:
        if item == "last":
            epochs.append(max(files))
        elif item == "best":
            best = (status.get("best") or {}).get("epoch")
            if best in files:
                epochs.append(best)
        elif int(item) in files:
            epochs.append(int(item))
    return sorted(set(epochs)), files


def stage_rows(run_name, epoch, stages, owners, ibi_gen, ibi_real, hrv_gen, hrv_real):
    rows = []
    groups = [(name, stages == code) for code, name in STAGES.items()] + [("all", np.ones(len(stages), bool))]
    for name, rows_mask in groups:
        n = int(rows_mask.sum())
        row = dict(run=run_name, epoch=epoch, stage=name, windows=n)
        if n:
            g, r = hrv_gen[rows_mask], hrv_real[rows_mask]
            row.update(w1_ms=w1_ms(ibi_gen[rows_mask].ravel(), ibi_real[rows_mask].ravel()),
                       hr_gen=np.nanmean(g[:, 0]), hr_real=np.nanmean(r[:, 0]),
                       sdnn_gen=np.nanmedian(g[:, 1]), sdnn_real=np.nanmedian(r[:, 1]),
                       rmssd_gen=np.nanmedian(g[:, 2]), rmssd_real=np.nanmedian(r[:, 2]))
        rows.append(row)
    return rows


def subject_rows(run_name, epoch, owners, hrv_gen, hrv_real):
    owners = np.asarray(owners)
    rows = []
    for sid in sorted(set(owners)):
        m = owners == sid
        rows.append(dict(run=run_name, epoch=epoch, subject=sid, windows=int(m.sum()),
                         hr_gen=np.nanmean(hrv_gen[m, 0]), hr_real=np.nanmean(hrv_real[m, 0]),
                         rmssd_gen=np.nanmedian(hrv_gen[m, 2]), rmssd_real=np.nanmedian(hrv_real[m, 2])))
    return rows


def table(rows, title):
    head = f"{title}\n  {'stage':5s} {'win':>4s} {'W1 ms':>6s}  {'HR gen/real bpm':>15s}  {'SDNN gen/real ms':>16s}  {'RMSSD gen/real ms':>17s}"
    lines = [head]
    fmt = lambda v: "   nan" if v is None or not np.isfinite(v) else f"{v:6.1f}"
    for r in rows:
        if not r["windows"]:
            lines.append(f"  {r['stage']:5s} {0:4d}")
            continue
        lines.append(f"  {r['stage']:5s} {r['windows']:4d} {fmt(r['w1_ms'])}  {fmt(r['hr_gen'])} / {fmt(r['hr_real'])}"
                     f"   {fmt(r['sdnn_gen'])} / {fmt(r['sdnn_real'])}    {fmt(r['rmssd_gen'])} / {fmt(r['rmssd_real'])}")
    return "\n".join(lines)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir is None:
        output = DEFAULT_OUTPUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    else:
        output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"never overwritten: {output}")
    all_stage, all_subject, pooled, summary = [], [], {}, []
    for run in [r.expanduser().resolve() for r in args.runs]:
        config, stages, owners, real_rebuilt = rebuild_validation(run)
        epochs, files = chosen_epochs(run, args.epochs)
        cache_real = None
        for epoch in epochs:
            z = np.load(files[epoch])
            if list(z["owners"]) != owners or not np.allclose(z["real"], real_rebuilt, atol=1e-5):
                raise SystemExit(f"{run.name} epoch {epoch}: the rebuilt validation windows differ from the saved "
                                 "ones (data or code changed since training)")
            ibi_gen, _ = pt.derived_ibi(z["generated"], config)
            if cache_real is None:
                ibi_real, _ = pt.derived_ibi(z["real"], config)
                cache_real = (ibi_real, window_hrv(ibi_real))
            ibi_real, hrv_real = cache_real
            hrv_gen = window_hrv(ibi_gen)
            all_stage += stage_rows(run.name, epoch, stages, owners, ibi_gen, ibi_real, hrv_gen, hrv_real)
            all_subject += subject_rows(run.name, epoch, owners, hrv_gen, hrv_real)
            pooled.setdefault(epoch, []).append((stages, ibi_gen, ibi_real, hrv_gen, hrv_real))
        last = [r for r in all_stage if r["run"] == run.name and r["epoch"] == epochs[-1]]
        summary.append(table(last, f"{run.name}, epoch {epochs[-1]}"))
        print(summary[-1], flush=True)
    common = [e for e, parts in pooled.items() if len(parts) == len(args.runs)]
    if len(args.runs) > 1 and common:
        e = max(common)
        parts = pooled[e]
        cat = [np.concatenate([p[i] for p in parts]) for i in range(5)]
        rows = stage_rows("pooled", e, cat[0], None, cat[1], cat[2], cat[3], cat[4])
        all_stage += rows
        summary.append(table(rows, f"all {len(args.runs)} runs pooled, epoch {e}"))
        print(summary[-1])
    output.mkdir(parents=True)
    write_csv(output / "stage_metrics.csv", all_stage)
    write_csv(output / "subject_metrics.csv", all_subject)
    (output / "summary.txt").write_text("\n\n".join(summary) + "\n", encoding="utf-8")
    print(f"output: {output}")
    return dict(output=str(output), stage_rows=all_stage, subject_rows=all_subject)


if __name__ == "__main__":
    main()
