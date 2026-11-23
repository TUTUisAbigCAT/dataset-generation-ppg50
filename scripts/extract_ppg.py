r"""Extract 50 Hz PPG, ECG/PPG beats, quality and 25 Hz IBI from DREAMT 2.2.0 (raw CSVs are only read).

See ppg_extraction.py for what is computed. Outputs (in --output-dir):
  <SID>.npz     per-epoch PPG (50 Hz), stages, events, beats, IBI (PPG / ECG / WatchSleepNet), quality
  report.json   every participant's full report (alignment, accuracy vs ECG, quality)
  report.csv    one row per participant with the key numbers
Participants already written are kept, so the job can be resubmitted.

Example:
  python scripts/extract_ppg.py --raw-dir /home/yxtao/links/scratch/dreamt/data_100Hz \
      --metadata /home/yxtao/links/scratch/dreamt/physionet.org/files/dreamt/2.2.0/participant_info.csv \
      --output-dir /home/yxtao/links/scratch/WatchSleepProcessedDataset/DREAMT22_PPG50 --workers 8
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

CSV_FIELDS = ("subject", "status", "epochs", "sleep_period_epochs", "epoch_runs", "stage_changes_on_grid_fraction",
              "aligned", "pulse_arrival_s", "pulse_arrival_concentration", "pulse_arrival_drift_s", "coarse_hr_lag_s",
              "ecg_inverted", "ppg50_sensitivity", "ppg50_ppv", "ppg50_ibi_mae_ms", "ppg50_ibi_bias_ms",
              "ppg50_unrefined_ibi_mae_ms", "ppg100_ibi_mae_ms", "ppg100_unrefined_ibi_mae_ms",
              "watchsleepnet_sensitivity", "watchsleepnet_ibi_mae_ms",
              "good_epoch_fraction", "good_epoch_fraction_sleep_period", "median_sqi", "median_acc_std_g",
              "ppg_artifact_beat_fraction", "ppg_ibi_zero_fraction", "ecg_ibi_zero_fraction", "wsn_ibi_zero_fraction",
              "event_sample_fraction", "participant_ahi", "seconds", "error")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True, help="DREAMT 2.2.0 data_100Hz folder")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None, help="participant_info.csv (for AHI in the report)")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ecg-column", default="ECG")
    parser.add_argument("--no-watchsleepnet", action="store_true", help="Skip the per-epoch WatchSleepNet IBI (faster)")
    for name, value in px.DEFAULT_THRESHOLDS.items():
        parser.add_argument("--min-" + name.replace("_", "-") if name in ("sqi", "ecg_match")
                            else "--max-" + name.replace("_", "-"), dest=name, type=float, default=value,
                            help=f"good-epoch threshold for {name} (default {value})")
    cli = parser.parse_args(argv)
    if cli.workers < 1:
        parser.error("--workers must be positive")
    return cli


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
    alignment, accuracy = report.get("alignment", {}), report.get("accuracy", {})
    get = lambda name, key: (accuracy.get(name) or {}).get(key)
    return dict(
        subject=report.get("subject"), status=report.get("status"), epochs=report.get("epochs"),
        sleep_period_epochs=report.get("sleep_period_epochs"), epoch_runs=report.get("epoch_runs"),
        ecg_inverted=alignment.get("ecg_inverted"), ppg50_ibi_bias_ms=get("ppg50_refined", "bias_ms"),
        ppg100_unrefined_ibi_mae_ms=get("ppg100_unrefined", "mae_ms"),
        stage_changes_on_grid_fraction=report.get("stage_changes_on_grid_fraction"),
        aligned=alignment.get("aligned"), pulse_arrival_s=alignment.get("pulse_arrival_s"),
        pulse_arrival_concentration=alignment.get("pulse_arrival_concentration"),
        pulse_arrival_drift_s=alignment.get("pulse_arrival_drift_s"), coarse_hr_lag_s=alignment.get("coarse_hr_lag_s"),
        ppg50_sensitivity=get("ppg50_refined", "sensitivity"), ppg50_ppv=get("ppg50_refined", "ppv"),
        ppg50_ibi_mae_ms=get("ppg50_refined", "mae_ms"), ppg50_unrefined_ibi_mae_ms=get("ppg50_unrefined", "mae_ms"),
        ppg100_ibi_mae_ms=get("ppg100_refined", "mae_ms"), watchsleepnet_sensitivity=get("watchsleepnet", "sensitivity"),
        watchsleepnet_ibi_mae_ms=get("watchsleepnet", "mae_ms"),
        **{key: report.get(key) for key in ("good_epoch_fraction", "good_epoch_fraction_sleep_period", "median_sqi",
                                            "median_acc_std_g", "ppg_artifact_beat_fraction", "ppg_ibi_zero_fraction",
                                            "ecg_ibi_zero_fraction", "wsn_ibi_zero_fraction", "event_sample_fraction",
                                            "participant_ahi",
                                            "seconds", "error")})


def process_subject(task):
    sid, csv_path, target, options = task
    started = time.time()
    try:
        arrays, report = px.extract_subject(csv_path, ecg_column=options["ecg_column"],
                                            watchsleepnet=options["watchsleepnet"],
                                            thresholds=options["thresholds"], ahi=options["ahi"].get(sid))
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
                   thresholds={name: getattr(cli, name) for name in px.DEFAULT_THRESHOLDS})
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
        print(json.dumps({key: row.get(key) for key in ("subject", "status", "aligned", "pulse_arrival_s",
                                                         "ppg50_ibi_mae_ms", "watchsleepnet_ibi_mae_ms",
                                                         "good_epoch_fraction", "seconds", "error")}), flush=True)

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
