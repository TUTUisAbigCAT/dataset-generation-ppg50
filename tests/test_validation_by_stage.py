"""scripts/validation_by_stage.py: validation windows of a run, by sleep stage."""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_ppg_training import tiny_config, write_cohort   # noqa: E402

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_PANDAS = importlib.util.find_spec("pandas") is not None


class BeatTests(unittest.TestCase):
    def test_beat_intervals_and_hrv(self):
        from scripts import validation_by_stage as vs

        row = np.r_[np.zeros(5), np.repeat([0.8, 0.9, 1.0, 0.9, 0.8, 0.9], 22), np.zeros(3)].astype(np.float32)
        beats = vs.beat_intervals(row)
        np.testing.assert_allclose(beats, [0.8, 0.9, 1.0, 0.9, 0.8, 0.9], rtol=1e-6)
        hr, sdnn, rmssd = vs.window_hrv(row[None])[0]
        b = beats * 1000.0
        self.assertAlmostEqual(hr, 60000.0 / b.mean(), places=3)
        self.assertAlmostEqual(sdnn, b.std(), places=3)
        self.assertAlmostEqual(rmssd, np.sqrt(np.mean(np.diff(b) ** 2)), places=3)
        self.assertTrue(np.all(np.isnan(vs.window_hrv(np.zeros((1, 50), np.float32))[0])))   # no beats


@unittest.skipUnless(HAVE_TORCH and HAVE_PANDAS, "torch and pandas are needed for a training run")
class RunReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import train_ppg

        cls.temporary = tempfile.TemporaryDirectory()
        folder = Path(cls.temporary.name)
        data, metadata = write_cohort(folder)
        config = folder / "config.json"
        config.write_text(json.dumps(tiny_config(data, metadata, folder / "runs", stage_condition="n1n2")))
        cls.run_dir = folder / "runs" / "fold0"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            train_ppg.main(["--config", str(config), "--output-dir", str(cls.run_dir), "--device", "cpu", "--allow-cpu"])
        cls.folder = folder

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def report(self, output, *extra):
        from scripts import validation_by_stage as vs

        with contextlib.redirect_stdout(io.StringIO()) as printed, contextlib.redirect_stderr(io.StringIO()):
            result = vs.main([str(self.run_dir), "--output-dir", str(output), *extra])
        return result, printed.getvalue()

    def test_stage_table_matches_the_runs_validation_windows(self):
        import ppg_training as pt

        result, printed = self.report(self.folder / "report", "--epochs", "best", "last")
        rows = result["stage_rows"]
        epochs = sorted({r["epoch"] for r in rows})
        status = json.loads((self.run_dir / "status.json").read_text())
        self.assertEqual(epochs, sorted({status["best"]["epoch"], status["epoch"]}))
        trainer = pt.Trainer(pt.load_config(self.run_dir / "config.json"), self.run_dir, device="cpu", log=lambda *_: None)
        stages = [int(trainer.cohort.records[s]["stages"][f]) for s, f in trainer.validation_windows]
        last = {r["stage"]: r for r in rows if r["epoch"] == status["epoch"]}
        self.assertEqual(last["all"]["windows"], len(stages))
        for code, name in {0: "W", 1: "N1", 2: "N2", 3: "Deep", 4: "REM"}.items():
            self.assertEqual(last[name]["windows"], stages.count(code))
        # the whole-run W1 is the per-sample W1 the script computes from the saved windows
        z = np.load(self.run_dir / "validation" / f"epoch_{status['epoch']:03d}.npz")
        config = pt.load_config(self.run_dir / "config.json")
        gen, _ = pt.derived_ibi(z["generated"], config)
        real, _ = pt.derived_ibi(z["real"], config)
        from scripts import validation_by_stage as vs
        self.assertAlmostEqual(last["all"]["w1_ms"], vs.w1_ms(gen.ravel(), real.ravel()), places=6)
        for name in ("stage_metrics.csv", "subject_metrics.csv", "summary.txt"):
            self.assertTrue((self.folder / "report" / name).is_file())
        with open(self.folder / "report" / "subject_metrics.csv") as stream:
            subjects = {row["subject"] for row in csv.DictReader(stream)}
        self.assertEqual(subjects, set(trainer.validation_subjects))
        self.assertIn("epoch", printed)
        with self.assertRaises(SystemExit):                                      # never overwritten
            self.report(self.folder / "report")

    def test_changed_windows_are_refused(self):
        import shutil

        copy = self.folder / "changed"
        shutil.copytree(self.run_dir, copy)
        path = sorted((copy / "validation").glob("epoch_*.npz"))[-1]
        z = dict(np.load(path))
        z["real"] = z["real"] + 0.5
        np.savez_compressed(path, **z)
        from scripts import validation_by_stage as vs
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as refused:
            vs.main([str(copy), "--epochs", "last", "--output-dir", str(self.folder / "changed_report")])
        self.assertIn("differ", str(refused.exception))
        self.assertFalse((self.folder / "changed_report").exists())


if __name__ == "__main__":
    unittest.main()
