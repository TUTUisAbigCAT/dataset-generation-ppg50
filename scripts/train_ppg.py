r"""Train the conditional diffusion model on 50 Hz PPG epochs (see ppg_training.py).

New run (output: <output_root>/ppg50_<SLURM_JOB_ID or time>, or --output-dir):
  python scripts/train_ppg.py --config configs/ppg50.json --max-hours 23
Continue a paused or interrupted run (same configuration, state.pt in the run folder):
  python scripts/train_ppg.py --resume /path/to/run --max-hours 23
Check data, validation participants and model shapes without training (CPU is fine):
  python scripts/train_ppg.py --config configs/ppg50.json --check-only
Quick GPU test in a debugjob (2 short epochs, then a resumed third one):
  python scripts/train_ppg.py --output-dir RUNS/smoke --epochs 2 --max-steps-per-epoch 50 --validation-windows 4
  python scripts/train_ppg.py --resume RUNS/smoke --epochs 3

Run folder: config.json, run_summary.json (participants, windows, validation choice, scaler),
metrics.csv (one row per epoch), best.pt (lowest validation IBI W1), last.pt, state.pt (resume),
validation/epoch_XXX.npz (generated and real validation windows), source/ (code snapshot).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Single-thread BLIS before NumPy/PyTorch load (as in run_dreamt.py): the cluster's multi-threaded
# BLIS aborts in bli_l3_decor_openmp.c next to PyTorch's own OpenMP runtime.
from host_runtime import configure_host_threads
if __name__ == "__main__":
    configure_host_threads(2)

import ppg_training as pt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ppg50.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="exact new run folder")
    parser.add_argument("--resume", type=Path, default=None, help="run folder to continue (uses its config.json)")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument("--max-steps-per-epoch", type=int, default=None,
                        help="quick test: stop each epoch after this many updates")
    parser.add_argument("--validation-windows", type=int, default=None,
                        help="quick test: validation windows per participant")
    parser.add_argument("--max-hours", type=float, default=None, help="pause cleanly before this many hours")
    parser.add_argument("--device", default=None, help="cuda (default when available) or cpu")
    parser.add_argument("--allow-cpu", action="store_true", help="permit training without CUDA (tests only)")
    return parser.parse_args(argv)


def snapshot_source(destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("ppg_training.py", "ppg_extraction.py", "diffusion_schedule.py", "apnea_labels.py"):
        shutil.copy2(ROOT / name, destination / name)
    for name in ("TSImputers", "utils"):
        target = destination / name
        if not target.exists():
            shutil.copytree(ROOT / name, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def main(argv=None):
    cli = parse_args(argv)
    overrides = {}
    if cli.epochs:
        overrides.setdefault("training", {})["epochs"] = cli.epochs
    if cli.max_steps_per_epoch:
        overrides.setdefault("training", {})["max_steps_per_epoch"] = cli.max_steps_per_epoch
    if cli.validation_windows:
        overrides["validation"] = {"windows_per_subject": cli.validation_windows}
    overrides = overrides or None
    if cli.resume is not None:
        output = cli.resume.expanduser().resolve()
        if not (output / "config.json").is_file():
            raise SystemExit(f"{output} has no config.json: the first run there did not start, so there is "
                             "nothing to resume (start it again without --resume)")
        config = pt.load_config(output / "config.json", overrides)
        if overrides:                      # a longer run keeps the saved config otherwise identical
            (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    else:
        config = pt.load_config(cli.config, overrides)
        if cli.output_dir is not None:
            output = cli.output_dir.expanduser().resolve()
        else:
            tag = os.environ.get("SLURM_JOB_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output = Path(config["output_root"]).expanduser() / f"ppg50_{tag}"
    trainer = pt.Trainer(config, output, device=cli.device if not cli.check_only else "cpu",
                         resume=cli.resume is not None)
    summary = trainer.summary()
    if cli.check_only:
        import torch

        model = pt.build_model(config, len(trainer.channels))
        window = torch.from_numpy(trainer.cohort.window_array(*trainer.training_windows[0]))[None]
        out = model(window, torch.zeros(1, 1, dtype=torch.long))
        parameters = sum(p.numel() for p in model.parameters())
        print(json.dumps(dict(status="check_passed", output_directory=str(output), model_parameters=parameters,
                              model_output_shape=list(out.shape),
                              **{k: summary[k] for k in ("channels", "window_samples", "training_subjects",
                                                         "training_windows", "validation_subjects",
                                                         "validation_windows", "skipped_subjects")}),
                         indent=1), flush=True)
        return dict(status="checked", summary=summary)
    import torch

    if trainer.device.type != "cuda" and not cli.allow_cpu:
        raise SystemExit("CUDA is required for training (use --allow-cpu only for tests)")
    if cli.resume is None and (output / "config.json").exists():
        raise SystemExit(f"{output} already holds a run; continue it with --resume {output}")
    output.mkdir(parents=True, exist_ok=True)       # a folder left by a run that never started is reused
    snapshot_source(output / "source")
    print(json.dumps(dict(status="training", output_directory=str(output), device=str(trainer.device),
                          training_windows=summary["training_windows"],
                          validation_subjects=summary["validation_subjects"])), flush=True)
    result = trainer.run(max_hours=cli.max_hours)
    print(json.dumps(dict(event="done", output_directory=str(output), **result), default=float), flush=True)
    return result


if __name__ == "__main__":
    main()
