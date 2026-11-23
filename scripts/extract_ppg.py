r"""Extract 50 Hz PPG, ECG/PPG beats, quality and 25 Hz IBI from DREAMT 2.2.0 (raw CSVs are only read).

See ppg_extraction.py for what is computed. Outputs (in --output-dir):
  <SID>.npz     per-epoch PPG (50 Hz), stages, events, beats, 25 Hz IBI variants, epoch_ok, quality
  report.json   every participant's full report (alignment, accuracy vs ECG, quality)
  report.csv    one row per participant with the key numbers
Participants already written are kept, so the job can be resubmitted.

Example:
  python scripts/extract_ppg.py --raw-dir /home/yxtao/links/scratch/dreamt/data_100Hz \
      --metadata /home/yxtao/links/scratch/dreamt/physionet.org/files/dreamt/2.2.0/participant_info.csv \
      --output-dir /home/yxtao/links/scratch/WatchSleepProcessedDataset/DREAMT22_PPG50 --workers 96
"""
from __future__ import annotations

import argparse
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

SERIES = ("ppg50_clean", "ppg50_relative", "ppg50_raw", "ppg100_clean", "watchsleepnet")
BEATS = ("ppg50", "ppg50_unrefined", "ppg100", "ppg100_unrefined", "watchsleepnet")
ALIGNMENT = ("ecg_inverted", "ecg_rr_jump_fraction", "ecg_irregular", "coarse_hr_lag_s", "tracked_fraction",
             "lag_median_s", "lag_p5_s", "lag_p95_s")
AGREEMENT = ("reference_matched", "other_matched", "dt_median_abs_ms", "dt_p95_abs_ms", "ibi_median_abs_ms",
             "ibi_p95_abs_ms")
CSV_FIELDS = (("subject", "status", "epochs", "sleep_period_epochs", "epoch_runs", "stage_changes_on_grid_fraction")
              + ALIGNMENT + ("ppg50_sensitivity", "ppg50_ppv")
              + tuple(f"route_50_vs_100_{key}" for key in AGREEMENT)
              + tuple(f"{name}_beat_ibi_mae_ms" for name in BEATS)
              + tuple(f"{name}_{key}" for name in SERIES for key in ("mae_ms", "coverage", "big_error_fraction"))
              + ("reference_coverage", "epoch_ok_fraction", "epoch_ok_fraction_sleep_period", "epoch_ok_fraction_relative",
                 "measured_sample_fraction", "interpolated_sample_fraction", "artifact_sample_fraction",
                 "no_beat_sample_fraction", "median_sqi", "median_acc_std_g", "event_sample_fraction",
                 "participant_ahi", "seconds", "error"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True, help="DREAMT 2.2.0 data_100Hz folder")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None, help="participant_info.csv (for AHI in the report)")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ecg-column", default="ECG")
    parser.add_argument("--no-watchsleepnet", action="store_true", help="Skip the per-epoch WatchSleepNet IBI (faster)")
    c = px.DEFAULT_CLEANING
    parser.add_argument("--ibi-min", type=float, default=c["ibi_min"], help="shortest valid interval (s)")
    parser.add_argument("--ibi-max", type=float, default=c["ibi_max"], help="valid intervals are below this (s)")
    parser.add_argument("--max-fill-intervals", type=int, default=c["max_fill_intervals"],
                        help="a run of at most this many bad intervals is interpolated (0 = never fill)")
    parser.add_argument("--max-fill-seconds", type=float, default=c["max_fill_s"],
                        help="... and only if it lasts at most this long (s); longer runs are 0")
    parser.add_argument("--min-valid-fraction", type=float, default=c["min_valid_fraction"],
                        help="epoch_ok needs at least this fraction of measured or filled samples")
    parser.add_argument("--relative-threshold", type=float, default=px.DEFAULT_RELATIVE_THRESHOLD,
                        help="extra rule of the ibi_ppg_relative variant: bad if |ibi - local median| > this x median")
    parser.add_argument("--relative-window", type=int, default=c["relative_window"],
                        help="intervals in the local median")
    cli = parser.parse_args(argv)
    if cli.workers < 1:
        parser.error("--workers must be positive")
    return cli


def cleaning_from(cli):
    return px.cleaning_options(ibi_min=cli.ibi_min, ibi_max=cli.ibi_max, max_fill_intervals=cli.max_fill_intervals,
                               max_fill_s=cli.max_fill_seconds, min_valid_fraction=cli.min_valid_fraction,
                               relative_window=cli.relative_window)


def csv_index(raw_dir):
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"--raw-dir does not exist: {raw_dir}")
    paths = sorted(list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.csv.gz")))
    if not paths:
        raise FileNotFoundError(f"No CSV files directly inside --raw-dir {raw_dir}")
    index = {}
    for path in paths:
        sid = path.name.split("_")[0].split(".")[0]
        if sid in index:
            raise ValueError(f"Two raw CSVs for {sid}: {index[sid].name}, {path.name}")
        index[sid] = path
    return index


def participant_ahi(path):
    if path is None or not Path(path).is_file():
        return {}
    import pandas as pd
    frame = pd.read_csv(path, dtype={"SID": str})
    return {str(sid).strip(): float(ahi) for sid, ahi in zip(frame["SID"], frame["AHI"])}


def flat_row(report):
    alignment = report.get("alignment") or {}
    beats, series = report.get("beat_accuracy") or {}, report.get("series_accuracy") or {}
    row = {key: report.get(key) for key in CSV_FIELDS}
    row.update({key: alignment.get(key) for key in ALIGNMENT})
    agreement = report.get("route_agreement_50_vs_100") or {}
    row.update({f"route_50_vs_100_{key}": agreement.get(key) for key in AGREEMENT})
    row["reference_coverage"] = series.get("reference_coverage")
    row.update(ppg50_sensitivity=(beats.get("ppg50") or {}).get("sensitivity"),
               ppg50_ppv=(beats.get("ppg50") or {}).get("ppv"))
    for name in BEATS:
        row[f"{name}_beat_ibi_mae_ms"] = (beats.get(name) or {}).get("mae_ms")
    for name in SERIES:
        for key in ("mae_ms", "coverage", "big_error_fraction"):
            row[f"{name}_{key}"] = (series.get(name) or {}).get(key)
    return row


def process_subject(task):
    sid, csv_path, target, options = task
    started = time.time()
    try:
        arrays, report = px.extract_subject(csv_path, ecg_column=options["ecg_column"],
                                            watchsleepnet=options["watchsleepnet"], cleaning=options["cleaning"],
                                            relative_threshold=options["relative_threshold"],
                                            ahi=options["ahi"].get(sid))
        temporary = Path(target).with_name("." + Path(target).stem + ".partial.npz")
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, target)
        report.update(status="written", seconds=round(time.time() - started, 1))
        return sid, report
    except Exception as error:
        return sid, dict(subject=sid, status="failed", error=str(error), traceback=traceback.format_exc(),
                         seconds=round(time.time() - started, 1))


def run(cli):
    output = cli.output_dir.expanduser().resolve()
    raw_dir = cli.raw_dir.expanduser().resolve()
    if output == raw_dir:
        raise ValueError("--output-dir must differ from the raw folder")
    csvs = csv_index(raw_dir)
    output.mkdir(parents=True, exist_ok=True)
    subjects = sorted(csvs)
    if cli.subjects:
        wanted = {s.strip() for s in cli.subjects}
        unknown = wanted - set(subjects)
        if unknown:
            raise ValueError(f"No raw CSV for {sorted(unknown)}")
        subjects = [s for s in subjects if s in wanted]
    previous = output / "report.json"
    reports = json.loads(previous.read_text(encoding="utf-8")).get("subjects", {}) if previous.is_file() else {}
    options = dict(ecg_column=cli.ecg_column, watchsleepnet=not cli.no_watchsleepnet, ahi=participant_ahi(cli.metadata),
                   cleaning=cleaning_from(cli), relative_threshold=cli.relative_threshold)
    tasks = []
    for sid in subjects:
        target = output / f"{sid}.npz"
        if target.exists():
            kept = reports.get(sid) if reports.get(sid, {}).get("status") in ("written", "kept_existing") else None
            if kept is None:                       # written by a job that stopped before saving report.json
                try:
                    with np.load(target) as saved:
                        kept = json.loads(str(saved["report"]))
                except Exception:
                    kept = None
            if kept is not None:
                reports[sid] = dict(kept, status="kept_existing")
                continue
        tasks.append((sid, csvs[sid], target, options))

    def save_reports():
        failed = sorted(sid for sid, r in reports.items() if r.get("status") == "failed")
        summary = dict(schema=px.SCHEMA + "_report", raw_dir=str(raw_dir), output_dir=str(output),
                       metadata=None if cli.metadata is None else str(cli.metadata),
                       options={k: v for k, v in options.items() if k != "ahi"}, subjects_total=len(reports),
                       subjects_ok=sum(r.get("status") in ("written", "kept_existing") for r in reports.values()),
                       subjects_failed=failed, subjects=reports)
        temporary = output / ".report.json.partial"
        temporary.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
        os.replace(temporary, previous)
        with (output / "report.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for sid in sorted(reports):
                writer.writerow(flat_row(reports[sid]))
        return summary

    def record(result):
        sid, report = result
        reports[sid] = report
        save_reports()                             # after every participant, so a stopped job loses nothing
        row = flat_row(report)
        print(json.dumps({key: row.get(key) for key in ("subject", "status", "ecg_irregular", "tracked_fraction",
                                                         "lag_median_s", "route_50_vs_100_reference_matched",
                                                         "route_50_vs_100_ibi_median_abs_ms", "ppg50_clean_mae_ms",
                                                         "ppg100_clean_mae_ms", "watchsleepnet_mae_ms",
                                                         "epoch_ok_fraction", "seconds", "error")}), flush=True)

    if cli.workers > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=min(cli.workers, len(tasks))) as pool:
            for future in as_completed([pool.submit(process_subject, task) for task in tasks]):
                record(future.result())
    else:
        for task in tasks:
            record(process_subject(task))
    summary = save_reports()
    failed = summary["subjects_failed"]
    print(json.dumps(dict(event="done", subjects_ok=summary["subjects_ok"], subjects_total=len(reports),
                          failed=failed)), flush=True)
    return summary


def main(argv=None):
    summary = run(parse_args(argv))
    if summary["subjects_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
