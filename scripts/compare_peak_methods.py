r"""Compare PPG beat-detection routes against the PSG ECG, from existing extraction files (read only).

  python scripts/compare_peak_methods.py --data-dir DREAMT22_PPG50 --output-dir OUT [--workers N]

For every participant the stored 50 Hz PPG goes through each route of ppg_extraction.ppg_blocks:

  elgendi_local        current route: 50 -> 100 Hz, nk.ppg_clean, local-RMS-normalized Elgendi, parabolic peaks
  elgendi              the same without the local normalization
  elgendi_local_fix    current route + correct_artifacts (neurokit2's Kubios correction)
  native500            nk.ppg_clean at 50 Hz, cleaned wave upsampled to 500 Hz, Elgendi there
  native500_fix        the teacher's route: native500 + correct_artifacts=True
  native500_local_fix  the teacher's route with the local normalization

and is scored as in the extraction: beats tracked against the ECG R peaks (the E4 clock drifts), beat
sensitivity / PPV and beat-to-beat IBI error against the matched RR intervals, and the 25 Hz IBI series
(same cleaning rules for every route) against the lag-aligned ECG IBI. Rhythm: the share of RR jumps > 30 %
(irregular above 0.2), as in the extraction report.

Writes OUT/subjects/<SID>.json, OUT/compare.csv (one row per participant and route) and OUT/summary.json,
and prints medians over participants with regular and with irregular rhythm.
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

DEFAULT_METHODS = ("elgendi_local", "elgendi", "elgendi_local_fix", "native500", "native500_fix",
                   "native500_local_fix")
FIELDS = ("subject", "method", "ecg_available", "ecg_irregular", "ecg_rr_jump_fraction", "beats", "sensitivity",
          "ppv", "beat_ibi_mae_ms", "beat_ibi_median_abs_ms", "beat_ibi_p95_abs_ms", "beat_ibi_bias_ms",
          "series_mae_ms", "series_coverage", "series_big_error_fraction", "epoch_ok_fraction",
          "measured_fraction", "interpolated_fraction", "artifact_fraction", "seconds")
SUMMARY_KEYS = ("sensitivity", "ppv", "beat_ibi_mae_ms", "beat_ibi_median_abs_ms", "series_mae_ms",
                "series_coverage", "series_big_error_fraction", "epoch_ok_fraction", "seconds")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="extraction folder (S*.npz from extract_ppg.py)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=px.ALL_PEAK_METHODS,
                        help="routes to compare; the first is the reference for alignment and 'better than'")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def compare_subject(path, methods):
    """(sid, result dict) for one extraction file."""
    sid = Path(path).stem
    started = time.time()
    with np.load(path) as z:
        schema = str(z["schema"]) if "schema" in z.files else ""
        if not schema.startswith("dreamt_ppg50_extraction_"):
            return sid, dict(subject=sid, status="skipped", reason=f"not an extraction file ({schema or 'no schema'})")
        fs = int(z["fs_ppg"]) if "fs_ppg" in z.files else px.FS_PPG
        ppg = z["ppg"].astype(np.float64)
        start_s = z["epoch_start_s"].astype(np.float64)
        epoch_index = z["epoch_index"].astype(np.int64)
        offset = int(z["grid_offset_rows"])
        ecg_t = z["ecg_peaks_s"].astype(np.float64) if "ecg_peaks_s" in z.files else np.zeros(0)
    options = px.cleaning_options()
    blocks, seconds = {}, {}
    for method in methods:
        t0 = time.time()
        blocks[method] = px.ppg_blocks(ppg, fs, start_s, peak_method=method)
        seconds[method] = round(time.time() - t0, 1)
    join = lambda b: np.concatenate([block["beat_times_s"] for block in b]) if b else np.zeros(0)
    beat_times = {m: join(blocks[m]) for m in methods}

    ecg_epoch = np.floor((ecg_t * px.FS_RAW - offset) / px.EPOCH_ROWS).astype(np.int64)
    ecg_inside = np.isin(ecg_epoch, epoch_index)
    have_ecg = bool(ecg_t.size > 100 and beat_times[methods[0]].size > 100)
    rhythm = px.ecg_rhythm(ecg_t[ecg_inside]) if have_ecg else None
    rows = {}
    reference = track_options = base_track = None
    if have_ecg:
        base_t = beat_times[methods[0]]
        coarse, _ = px.hr_cross_lag(ecg_t, base_t)
        _, centre = px.align_beats(ecg_t, base_t, coarse=coarse)
        prior = 0.3 if centre is None else centre
        track_options = dict(priors=(prior,), centre=prior, max_dev=4.0)
        base_track = px.track_beats(ecg_t, base_t, **track_options)
        times25 = px.epoch_sample_times(start_s, px.FS_IBI, px.EPOCH_IBI)
        shift = px.lag_series(times25, base_t, base_track)
        reference = np.where(np.isfinite(shift), px.ibi_at(times25 - np.nan_to_num(shift), ecg_t), 0.0)
    for method in methods:
        t = beat_times[method]
        variant = px.blocks_to_ibi(blocks[method], start_s, options)
        status = variant["status"]
        row = dict(beats=int(t.size), seconds=seconds[method], epoch_ok_fraction=float(variant["epoch_ok"].mean()),
                   measured_fraction=float((status == px.MEASURED).mean()),
                   interpolated_fraction=float((status == px.INTERPOLATED).mean()),
                   artifact_fraction=float((status == px.ARTIFACT).mean()))
        if have_ecg:
            track = base_track if method == methods[0] else px.track_beats(ecg_t, t, **track_options)
            accuracy = px.tracked_accuracy(ecg_t, t, track, ecg_inside=ecg_inside)
            row.update(sensitivity=accuracy["sensitivity"], ppv=accuracy["ppv"], beat_ibi_mae_ms=accuracy["mae_ms"],
                       beat_ibi_median_abs_ms=accuracy["median_abs_ms"], beat_ibi_p95_abs_ms=accuracy["p95_abs_ms"],
                       beat_ibi_bias_ms=accuracy["bias_ms"])
            row.update({f"series_{k}": v for k, v in px.series_accuracy(variant["ibi"], reference).items()})
        rows[method] = row
    return sid, dict(subject=sid, status="done", methods=rows, ecg_available=have_ecg, ecg_rr_jump_fraction=rhythm,
                     ecg_irregular=bool(rhythm is not None and rhythm > 0.2), seconds=round(time.time() - started, 1))


def _worker(task):
    path, methods = task
    try:
        return compare_subject(path, methods)
    except Exception as error:
        return Path(path).stem, dict(subject=Path(path).stem, status="failed", error=str(error),
                                     traceback=traceback.format_exc())


def flat_rows(results):
    rows = []
    for sid in sorted(results):
        r = results[sid]
        if r.get("status") != "done":
            continue
        for method, values in r["methods"].items():
            row = dict(subject=sid, method=method, ecg_available=r["ecg_available"], ecg_irregular=r["ecg_irregular"],
                       ecg_rr_jump_fraction=r["ecg_rr_jump_fraction"])
            row.update(values)
            rows.append(row)
    return rows


def summarize(results, methods):
    done = [r for r in results.values() if r.get("status") == "done" and r["ecg_available"]]
    groups = dict(regular=[r for r in done if not r["ecg_irregular"]], irregular=[r for r in done if r["ecg_irregular"]])
    summary = dict(methods=list(methods), reference=methods[0], participants_with_ecg=len(done), groups={})
    for name, members in groups.items():
        out = dict(participants=len(members), methods={})
        for method in methods:
            values = {}
            for key in SUMMARY_KEYS:
                data = [r["methods"][method].get(key) for r in members]
                data = [float(v) for v in data if v is not None]
                values[key] = float(np.median(data)) if data else None
            for key in ("beat_ibi_mae_ms", "series_mae_ms"):
                pairs = [(r["methods"][method].get(key), r["methods"][methods[0]].get(key)) for r in members]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                values[f"{key}_better_than_reference"] = f"{sum(a < b for a, b in pairs)}/{len(pairs)}"
            out["methods"][method] = values
        summary["groups"][name] = out
    return summary


def print_summary(summary):
    def fmt(v, d=1):
        return "  -  " if v is None else f"{v:.{d}f}"

    for name, group in summary["groups"].items():
        print(f"\n=== {name} rhythm: {group['participants']} participants (medians over participants) ===")
        print(f"{'method':20s} {'sens':>6s} {'PPV':>6s} {'beat MAE':>9s} {'beat med':>9s} {'better':>7s}"
              f" {'series MAE':>11s} {'cover':>6s} {'big err':>8s} {'better':>7s} {'ok':>6s} {'sec':>6s}")
        for method, v in group["methods"].items():
            print(f"{method:20s} {fmt(v['sensitivity'], 3):>6s} {fmt(v['ppv'], 3):>6s} {fmt(v['beat_ibi_mae_ms']):>9s}"
                  f" {fmt(v['beat_ibi_median_abs_ms']):>9s} {v['beat_ibi_mae_ms_better_than_reference']:>7s}"
                  f" {fmt(v['series_mae_ms']):>11s} {fmt(v['series_coverage'], 3):>6s}"
                  f" {fmt(v['series_big_error_fraction'], 3):>8s} {v['series_mae_ms_better_than_reference']:>7s}"
                  f" {fmt(v['epoch_ok_fraction'], 3):>6s} {fmt(v['seconds']):>6s}")
    print(f"\nbetter = participants with a lower error than {summary['reference']}; beat MAE/med = |PP - RR| (ms);"
          " series = 25 Hz IBI vs lag-aligned ECG IBI; ok = epoch_ok fraction; sec = seconds per night")


def main(argv=None):
    cli = parse_args(argv)
    methods = list(dict.fromkeys(cli.methods))
    data = cli.data_dir.expanduser().resolve()
    output = cli.output_dir.expanduser().resolve()
    if output == data:
        raise SystemExit("--output-dir must differ from the extraction folder")
    files = sorted(p for p in data.glob("S*.npz") if not p.name.startswith("."))
    if cli.subjects:
        wanted = set(cli.subjects)
        files = [p for p in files if p.stem in wanted]
        missing = wanted - {p.stem for p in files}
        if missing:
            raise SystemExit(f"No extraction file for {sorted(missing)}")
    (output / "subjects").mkdir(parents=True, exist_ok=True)
    results, tasks = {}, []
    for path in files:
        stored = output / "subjects" / f"{path.stem}.json"
        if stored.is_file():
            previous = json.loads(stored.read_text(encoding="utf-8"))
            if previous.get("status") in ("done", "skipped") and \
                    (previous.get("status") == "skipped" or list(previous.get("methods", {})) == methods):
                results[path.stem] = previous
                continue
        tasks.append((str(path), methods))
    print(json.dumps(dict(participants=len(files), to_run=len(tasks), methods=methods, workers=cli.workers)), flush=True)

    def record(sid, result):
        results[sid] = result
        (output / "subjects" / f"{sid}.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
        line = {k: result.get(k) for k in ("subject", "status", "ecg_irregular", "seconds", "error")}
        if result.get("status") == "done" and result["ecg_available"]:
            line.update({m: result["methods"][m].get("beat_ibi_mae_ms") for m in methods})
        print(json.dumps(line, default=float), flush=True)

    if cli.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(cli.workers, len(tasks))) as pool:
            for future in as_completed([pool.submit(_worker, t) for t in tasks]):
                record(*future.result())
    else:
        for task in tasks:
            record(*_worker(task))
    rows = flat_rows(results)
    with (output / "compare.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(results, methods)
    summary["failed"] = sorted(s for s, r in results.items() if r.get("status") == "failed")
    summary["skipped"] = sorted(s for s, r in results.items() if r.get("status") == "skipped")
    (output / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print_summary(summary)
    print(f"\nwritten: {output / 'compare.csv'}, {output / 'summary.json'}", flush=True)
    return summary


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
