r"""Generate whole-night 50 Hz PPG for every participant and compare its quality with the real PPG.

  python scripts/generate_ppg_nights.py --run RUN_DIR [--checkpoint best.pt] --output-dir OUT
  python scripts/generate_ppg_nights.py --run RUN_DIR --output-dir OUT --check-only
  ... --ibi          also derive IBI (each window on its own), the event counterfactual and night NPZ files
  ... --quality-threshold 0.8   regenerate windows whose template SQI is below 0.8 (off by default)

The checkpoint is copied into OUT/checkpoint.pt first and only that copy is used, so a training job that
keeps updating best.pt cannot mix two models in one output. Rerunning the same command continues
(participants already written with this checkpoint and these options are kept).

OUT/
  generation.json          checkpoint (source, epoch, sha256, its validation metrics) and options
  subjects/<SID>.npz       generated PPG [n, 1500] and per-window quality of generated and real PPG
                           (templatematch, disimilarity, skewness, kurtosis, relative_power, beats),
                           stages, events; with --ibi also IBI and the counterfactual windows
  subjects/<SID>.json      quality comparison (medians, W1, KS, share below templatematch 0.9/0.8,
                           by stage); with --ibi also IBI metrics and event effects
  night_npz/, night_npz_real/   (--ibi only) generated and real nights in the WatchSleepNet NPZ format
  subjects_summary.csv, summary.json
Plot: python scripts/plot_ppg_quality.py OUT
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from host_runtime import configure_host_threads
if __name__ == "__main__":
    configure_host_threads(2)

import numpy as np

import ppg_generation as pg
import ppg_training as pt

OPTION_KEYS = ("seed", "quality_threshold", "max_retries", "counterfactual_windows", "ibi")
QUALITY_FIELDS = tuple(f"{m}_{k}" for m in ("templatematch", "disimilarity", "skewness", "kurtosis", "relative_power",
                                             "beats") for k in ("median_gen", "median_real", "w1", "ks")) + (
    "templatematch_below_0.9_gen", "templatematch_below_0.9_real", "templatematch_below_0.8_gen",
    "templatematch_below_0.8_real", "W_templatematch_median_gen", "W_templatematch_median_real",
    "Light_templatematch_median_gen", "Light_templatematch_median_real", "REM_templatematch_median_gen",
    "REM_templatematch_median_real", "Deep_templatematch_median_gen", "Deep_templatematch_median_real")
CSV_FIELDS = ("subject", "status", "epochs", "validation_subject", "ecg_irregular", "lag_median_s") + QUALITY_FIELDS + (
              "w1_ms", "mean_gen_ms", "mean_real_ms", "sd_gen_ms", "sd_real_ms",
              "valid_fraction_gen", "valid_fraction_real", "beats_per_epoch_gen", "beats_per_epoch_real",
              "sqi_median_gen", "sqi_median_real", "sqi_below_threshold_first_attempt", "sqi_below_threshold_final",
              "sqi_below_threshold_real", "retried_epochs", "mean_attempts",
              "W_w1_ms", "Light_w1_ms", "REM_w1_ms", "Deep_w1_ms",
              "W_mean_gen_ms", "W_mean_real_ms", "Light_mean_gen_ms", "Light_mean_real_ms",
              "REM_mean_gen_ms", "REM_mean_real_ms", "Deep_mean_gen_ms", "Deep_mean_real_ms",
              "event_sample_fraction", "within_window_contrast_real_ms", "within_window_contrast_gen_ms",
              "counterfactual_ibi_change_event_samples_ms", "counterfactual_ibi_change_other_samples_ms",
              "counterfactual_mean_abs_ppg_change", "seconds", "error")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="training run folder")
    parser.add_argument("--checkpoint", default="best.pt", help="file name in the run folder, or a path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--quality-threshold", type=float, default=0.0,
                        help="regenerate windows whose template SQI is below this (default 0 = keep the raw output)")
    parser.add_argument("--ibi", action="store_true",
                        help="also derive IBI, the event counterfactual and night NPZ files")
    parser.add_argument("--quality-workers", type=int, default=min(12, os.cpu_count() or 1),
                        help="processes for the quality measures")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--counterfactual-windows", type=int, default=64,
                        help="windows with events generated again with the event channel zeroed")
    parser.add_argument("--data-dir", type=Path, default=None, help="override the extraction folder")
    parser.add_argument("--max-hours", type=float, default=None, help="stop between participants after this")
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-cpu", action="store_true", help="tests only")
    parser.add_argument("--check-only", action="store_true")
    cli = parser.parse_args(argv)
    if cli.batch_size < 1 or cli.max_retries < 0 or cli.counterfactual_windows < 0 or not 0 <= cli.quality_threshold < 1:
        parser.error("batch size >= 1, retries and counterfactual windows >= 0, threshold in [0, 1)")
    return cli


def flat_row(metrics, extra):
    row = dict(extra)
    if metrics:
        row.update({k: metrics.get(k) for k in CSV_FIELDS if k in metrics and not isinstance(metrics.get(k), dict)})
        for measure, values in (metrics.get("quality") or {}).items():
            if measure.startswith("templatematch_below_"):
                row[f"{measure}_gen"], row[f"{measure}_real"] = values["gen"], values["real"]
            else:
                row.update({f"{measure}_{k}": values.get(k) for k in ("median_gen", "median_real", "w1", "ks")})
        for stage, values in (metrics.get("quality_by_stage") or {}).items():
            for side in ("gen", "real"):
                row[f"{stage}_templatematch_median_{side}"] = values.get(f"templatematch_median_{side}")
        row.update({k: (metrics.get("overall") or {}).get(k) for k in ("w1_ms", "mean_gen_ms", "mean_real_ms",
                                                                       "sd_gen_ms", "sd_real_ms")})
        for stage, values in metrics.get("by_stage", {}).items():
            row[f"{stage}_w1_ms"] = values.get("w1_ms")
            row[f"{stage}_mean_gen_ms"] = values.get("mean_gen_ms")
            row[f"{stage}_mean_real_ms"] = values.get("mean_real_ms")
        row.update({k: v for k, v in metrics.get("events", {}).items() if k in CSV_FIELDS})
    row.update(extra)                       # status and participant flags of this run win over stored metrics
    return row


def write_summary(output, rows, expected=()):
    """subjects_summary.csv and summary.json; participants of `expected` not generated yet are listed as pending."""
    pending = sorted(s for s in expected if s not in rows)
    with (output / "subjects_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for sid in sorted(set(rows) | set(pending)):
            writer.writerow(rows.get(sid) or dict(subject=sid, status="pending"))
    done = [r for r in rows.values() if r.get("status") in ("written", "kept_existing")]
    def median(key, subset=done):
        values = [float(r[key]) for r in subset if r.get(key) not in (None, "")]
        return float(np.median(values)) if values else None
    regular = [r for r in done if not r.get("ecg_irregular")]
    summary = dict(schema=pg.SCHEMA, subjects=len(rows) + len(pending), written=len(done), pending=pending,
                   failed=sorted(s for s, r in rows.items() if r.get("status") == "failed"),
                   median={k: median(k) for k in CSV_FIELDS if k not in ("subject", "status", "error")
                           and any(isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool) for r in done)},
                   median_regular_rhythm_w1_ms=median("w1_ms", regular))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return summary


def save_npz(path, arrays):
    temporary = path.with_name("." + path.stem + ".partial.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def main(argv=None):
    import torch

    cli = parse_args(argv)
    run = cli.run.expanduser().resolve()
    source = Path(cli.checkpoint).expanduser()
    if not source.is_absolute():
        source = run / source if (run / source).exists() else source.resolve()
    output = cli.output_dir.expanduser().resolve()
    options = dict(seed=cli.seed, quality_threshold=cli.quality_threshold, max_retries=cli.max_retries,
                   counterfactual_windows=cli.counterfactual_windows, ibi=bool(cli.ibi))
    manifest_path = output / "generation.json"
    frozen = output / "checkpoint.pt"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if {k: manifest["options"].get(k) for k in OPTION_KEYS} != options:
            raise SystemExit(f"{output} was generated with other options {manifest['options']}; use a new --output-dir")
        if pg.sha256_file(frozen) != manifest["checkpoint"]["sha256"]:
            raise SystemExit(f"{frozen} changed since the first run; use a new --output-dir")
        first = Path(manifest["checkpoint"]["source"]).expanduser().resolve()
        if first != source.resolve():
            raise SystemExit(f"{output} was generated from {first} (epoch {manifest['checkpoint'].get('epoch')}), "
                             f"not {source}; use a new --output-dir (LABEL)")
    else:
        if not source.is_file():
            raise SystemExit(f"Checkpoint not found: {source}")
        manifest = None
    checkpoint = frozen if manifest else source
    device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if cli.check_only:
        device = torch.device("cpu")
    model, weights = pg.load_generator(checkpoint, device)
    config = dict(weights["config"])
    if cli.data_dir is not None:
        config["data_dir"] = str(cli.data_dir.expanduser().resolve())
    cohort = pt.PpgCohort(config)
    cohort.scaler = weights["scaler"]
    subjects = cohort.subjects if not cli.subjects else list(cli.subjects)
    unknown = sorted(set(subjects) - set(cohort.subjects))
    if unknown:
        raise SystemExit(f"No extraction file or participant info for {unknown}")
    run_summary = run / "run_summary.json"
    validation_subjects = set(json.loads(run_summary.read_text())["validation_subjects"]) if run_summary.is_file() else set()
    epochs_total = int(sum(cohort.records[s]["stages"].size for s in subjects))
    info = dict(checkpoint=str(source), checkpoint_epoch=weights.get("epoch"), checkpoint_validation=weights.get("validation"),
                subjects=len(subjects), epochs=epochs_total, output_directory=str(output), device=str(device))
    if cli.check_only:
        print(json.dumps(dict(status="check_passed", **info), indent=1, default=float), flush=True)
        return dict(status="checked", **info)
    if device.type != "cuda" and not cli.allow_cpu:
        raise SystemExit("CUDA is required (use --allow-cpu only for tests)")
    for folder in ("subjects",) + (("night_npz", "night_npz_real") if cli.ibi else ()):
        (output / folder).mkdir(parents=True, exist_ok=True)
    if manifest is None:
        shutil.copy2(source, frozen)
        manifest = dict(schema=pg.SCHEMA, checkpoint=dict(source=str(source), sha256=pg.sha256_file(frozen),
                                                          epoch=weights.get("epoch"), validation=weights.get("validation")),
                        run=str(run), options=dict(options, batch_size=cli.batch_size), data_dir=config["data_dir"])
        manifest_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    print(json.dumps(dict(status="generating", **dict(info, checkpoint_epoch=manifest["checkpoint"].get("epoch"),
                                                     checkpoint=manifest["checkpoint"]["source"]),
                          checkpoint_validation_ibi_w1_ms=(manifest["checkpoint"].get("validation") or {}).get("ibi_w1_ms")),
                     default=float), flush=True)
    schedule = pt.build_diffusion(config, device)
    report = cohort.report
    rows, started, durations = {}, time.time(), []
    for sid in sorted(cohort.subjects):
        alignment = (report.get(sid) or {}).get("alignment") or {}
        extra = dict(subject=sid, validation_subject=sid in validation_subjects,
                     ecg_irregular=alignment.get("ecg_irregular"), lag_median_s=alignment.get("lag_median_s"))
        meta_path = output / "subjects" / f"{sid}.json"
        if meta_path.is_file():
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            if (stored.get("status") == "written" and stored.get("schema") == pg.SCHEMA
                    and stored.get("checkpoint_sha256") == manifest["checkpoint"]["sha256"]):
                rows[sid] = flat_row(stored, dict(extra, status="kept_existing", epochs=stored.get("epochs"),
                                                  seconds=stored.get("seconds")))
                continue
        if sid not in subjects:
            continue
        if cli.max_hours is not None and durations and \
                (time.time() - started + 1.5 * max(durations)) / 3600.0 > cli.max_hours:
            left = [s for s in subjects if s not in rows]
            print(json.dumps(dict(event="paused_for_time", next_subject=sid, remaining=left,
                                  rerun_same_command=True)), flush=True)
            break
        t0 = time.time()
        try:
            arrays, metrics = pg.generate_subject(sid, cohort, model, schedule, device,
                                                  dict(options, batch_size=cli.batch_size,
                                                       checkpoint_epoch=weights.get("epoch"),
                                                       quality_workers=cli.quality_workers))
            record = cohort.records[sid]
            ahi = float(record["continuous"][0])
            save_npz(output / "subjects" / f"{sid}.npz", arrays)
            if cli.ibi:
                save_npz(output / "night_npz" / f"{sid}.npz",
                         pg.night_arrays(arrays["ibi_generated"], record, ahi, synthetic=True,
                                         epoch_start_s=record.get("epoch_start_s")))
                save_npz(output / "night_npz_real" / f"{sid}.npz",
                         pg.night_arrays(arrays["ibi_real_windowed"], record, ahi, synthetic=False,
                                         epoch_start_s=record.get("epoch_start_s")))
            seconds = round(time.time() - t0, 1)
            metrics.update(status="written", seconds=seconds, checkpoint_sha256=manifest["checkpoint"]["sha256"])
            meta_path.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
            rows[sid] = flat_row(metrics, dict(extra, status="written", epochs=metrics["epochs"], seconds=seconds))
            durations.append(time.time() - t0)
        except Exception as error:
            import traceback
            rows[sid] = dict(extra, status="failed", error=str(error))
            print(traceback.format_exc(), file=sys.stderr, flush=True)
        r = rows[sid]
        print(json.dumps({k: r.get(k) for k in ("subject", "status", "epochs", "templatematch_median_gen",
                                                 "templatematch_median_real", "templatematch_below_0.8_gen",
                                                 "templatematch_below_0.8_real", "skewness_median_gen",
                                                 "skewness_median_real", "w1_ms", "seconds", "error")},
                         default=float), flush=True)
        write_summary(output, rows, subjects)
    summary = write_summary(output, rows, subjects)
    remaining = [s for s in subjects if s not in rows]
    status = "completed" if not remaining and not summary["failed"] else ("paused" if remaining else "completed_with_failures")
    print(json.dumps(dict(event="done", status=status, written=summary["written"], failed=summary["failed"],
                          remaining=len(remaining),
                          templatematch_median_gen=summary["median"].get("templatematch_median_gen"),
                          templatematch_median_real=summary["median"].get("templatematch_median_real"),
                          output_directory=str(output)), default=float), flush=True)
    return dict(status=status, summary=summary)


if __name__ == "__main__":
    main()
