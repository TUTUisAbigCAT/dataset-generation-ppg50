"""Per-fold generator configurations (scripts/make_fold_configs.py) and submit_train_ppg.sh's RUN_DIR CONFIG mode."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_ppg_training import tiny_config, write_cohort   # noqa: E402

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_PANDAS = importlib.util.find_spec("pandas") is not None
FOLDS = [dict(fold=0, test=["S001", "S002"], val=["S003"], train=["S004", "S005", "S092"],
              generator_train=["S003", "S004", "S005", "S092"]),
         dict(fold=1, test=["S003", "S005"], val=["S004"], train=["S001", "S002", "S092"],
              generator_train=["S001", "S002", "S004", "S092"])]


@unittest.skipUnless(HAVE_PANDAS, "pandas is needed for the synthetic cohort")
class FoldConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.data, self.metadata = write_cohort(self.folder)
        self.base = self.folder / "base.json"
        self.base.write_text(json.dumps(tiny_config(self.data, self.metadata, self.folder / "runs",
                                                    stage_condition="n1n2")))
        self.folds = self.folder / "folds.json"
        self.folds.write_text(json.dumps(dict(folds=FOLDS)))

    def tearDown(self):
        self.temporary.cleanup()

    def make(self, output):
        from scripts import make_fold_configs

        with contextlib.redirect_stdout(io.StringIO()):
            return make_fold_configs.main(["--folds", str(self.folds), "--base", str(self.base),
                                           "--output-dir", str(output)])

    def test_each_fold_excludes_its_test_participants(self):
        import ppg_training as pt

        out = self.folder / "fold_configs"
        self.assertEqual(self.make(out)["folds"], [0, 1])
        base = json.loads(self.base.read_text())
        for fold in FOLDS:
            k = fold["fold"]
            config = json.loads((out / f"fold{k}.json").read_text())
            self.assertEqual(config["exclude_subjects"], fold["test"])
            self.assertEqual({key: v for key, v in config.items() if key != "exclude_subjects"},
                             {key: v for key, v in base.items() if key != "exclude_subjects"})
            self.assertEqual(pt.load_config(out / f"fold{k}.json")["stage_condition"], "n1n2")
            self.assertEqual((out / f"fold{k}_train.txt").read_text().split(), fold["train"])
        source = json.loads((out / "folds_source.json").read_text())
        self.assertEqual(source["folds_written"], [0, 1])
        self.assertEqual(len(source["folds_sha256"]), 64)
        with self.assertRaises(SystemExit):                                  # never overwritten
            self.make(out)

    def test_overlapping_fold_is_refused(self):
        bad = [dict(FOLDS[0], train=["S001", "S004"])]
        self.folds.write_text(json.dumps(dict(folds=bad)))
        with self.assertRaises(SystemExit) as refused:
            self.make(self.folder / "bad")
        self.assertIn("S001", str(refused.exception))
        self.assertFalse((self.folder / "bad").exists())

    @unittest.skipUnless(HAVE_TORCH, "torch is needed for training")
    def test_fold_generator_never_trains_on_its_test_participants(self):
        from scripts import train_ppg

        out = self.folder / "fold_configs"
        self.make(out)
        run = self.folder / "runs" / "ppg50n1n2_fold0"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            checked = train_ppg.main(["--config", str(out / "fold0.json"), "--output-dir", str(run), "--check-only"])
            summary = checked["summary"]
            self.assertEqual(summary["skipped_subjects"]["S001"], "excluded by configuration")
            self.assertEqual(summary["skipped_subjects"]["S002"], "excluded by configuration")
            self.assertFalse({"S001", "S002"} & set(summary["validation_subjects"]))
            first = train_ppg.main(["--config", str(out / "fold0.json"), "--output-dir", str(run),
                                    "--device", "cpu", "--allow-cpu"])
            again = train_ppg.main(["--resume", str(run), "--device", "cpu", "--allow-cpu"])
        self.assertEqual(first["status"], "completed")
        self.assertEqual(again["status"], "completed")                     # a spare queued job does nothing
        self.assertEqual(json.loads((run / "config.json").read_text())["exclude_subjects"], ["S001", "S002"])
        self.assertEqual(len((run / "metrics.csv").read_text().splitlines()), 2)


class SubmitScriptTests(unittest.TestCase):
    """Which train_ppg.py command submit_train_ppg.sh runs (module, the venv and python stubbed)."""

    def run_script(self, *args):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / ".venv" / "bin").mkdir(parents=True)
            (tmp / ".venv" / "bin" / "activate").write_text("")
            log = tmp / "calls.txt"
            stub = ('module() { :; }; python() { echo "$*" >> "$CALLS"; }; export -f module python; '
                    f'bash "{ROOT / "submit_train_ppg.sh"}" "$@"')
            env = dict(os.environ, SLURM_SUBMIT_DIR=str(tmp), CALLS=str(log))
            for key in ("RUN", "CONFIG", "EPOCHS", "MAX_HOURS"):
                env.pop(key, None)
            subprocess.run(["bash", "-c", stub, "submit"] + [str(a) for a in args], env=env, check=True,
                           capture_output=True, text=True)
            return [line for line in log.read_text().splitlines() if "train_ppg.py" in line]

    def test_run_dir_and_config_start_then_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "ppg50n1n2_fold0"
            config = Path(tmp) / "fold0.json"
            [call] = self.run_script(run, config)
            self.assertIn(f"--config {config} --output-dir {run} --max-hours 3.5", call)
            run.mkdir()
            (run / "config.json").write_text("{}")
            [call] = self.run_script(run, config)                             # the run exists: continue it
            self.assertIn(f"--resume {run} --max-hours 3.5", call)
            self.assertNotIn("--config", call)

    def test_existing_modes_unchanged(self):
        [call] = self.run_script("/runs/ppg50_1")
        self.assertIn("--resume /runs/ppg50_1 --max-hours 3.5", call)
        [call] = self.run_script()
        self.assertIn("--config configs/ppg50.json --max-hours 3.5", call)


if __name__ == "__main__":
    unittest.main()
