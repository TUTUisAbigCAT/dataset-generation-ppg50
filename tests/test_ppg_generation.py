"""Whole-night PPG generation with a tiny model trained for one epoch on a synthetic cohort (CPU)."""
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

HAVE_STACK = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("neurokit2") is not None


@unittest.skipUnless(HAVE_STACK, "torch and neurokit2 are needed")
class NightGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_ppg_training import tiny_config, write_cohort
        from scripts import train_ppg

        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        data, metadata = write_cohort(cls.folder)
        config = tiny_config(data, metadata, cls.folder / "runs")
        path = cls.folder / "tiny.json"
        path.write_text(json.dumps(config))
        cls.run_dir = cls.folder / "run"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            train_ppg.main(["--config", str(path), "--output-dir", str(cls.run_dir), "--device", "cpu", "--allow-cpu"])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def generate(self, out, *extra):
        from scripts import generate_ppg_nights

        args = ["--run", str(self.run_dir), "--output-dir", str(out), "--device", "cpu", "--allow-cpu",
                "--batch-size", "8", "--max-retries", "1", "--counterfactual-windows", "2", *extra]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return generate_ppg_nights.main(args)

    def test_quality_retry_keeps_the_best_attempt(self):
        import torch
        import ppg_generation as pg
        import ppg_training as pt

        model, weights = pg.load_generator(self.run_dir / "best.pt", torch.device("cpu"))
        cohort = pt.PpgCohort(weights["config"])
        cohort.scaler = weights["scaler"]
        conditions = np.stack([cohort.window_array("S001", k) for k in range(6)])
        schedule = pt.build_diffusion(weights["config"], torch.device("cpu"))
        ppg, sqi, first, attempts = pg.generate_with_quality(
            model, conditions, schedule, torch.device("cpu"), torch.Generator().manual_seed(3),
            batch_size=4, threshold=0.999, max_retries=2)
        self.assertEqual(ppg.shape, (6, 1500))
        self.assertTrue((sqi >= first).all())
        self.assertTrue((attempts == 3).all())                                   # nothing reaches 0.999
        np.testing.assert_allclose(pg.windowed(ppg)[2], sqi, atol=1e-6)          # stored SQI is the kept one's

    def test_nights_metrics_resume_and_frozen_checkpoint(self):
        out = self.folder / "nights"
        check = self.generate(out, "--check-only")
        self.assertEqual(check["subjects"], 6)
        first = self.generate(out, "--subjects", "S001", "S002")
        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["summary"]["written"], 2)
        with np.load(out / "subjects" / "S001.npz") as z:
            self.assertEqual(z["generated_ppg"].shape, (14, 1500))
            self.assertEqual(z["ibi_generated"].shape, (14, 750))
            self.assertEqual(z["counterfactual_ppg_with_events"].shape, (1, 1500))   # one epoch has events
            generated = z["generated_ppg"].copy()
        with np.load(out / "night_npz" / "S001.npz") as z:
            self.assertEqual(z["data"].shape, (14 * 750, 1))
            self.assertEqual(z["stages"].shape, (14 * 750,))
            self.assertEqual(int(z["fs"]), 25)
            self.assertTrue(bool(z["synthetic"]))
            self.assertEqual(z["apnea"].sum(), 350)
        with np.load(out / "night_npz_real" / "S001.npz") as z:
            self.assertFalse(bool(z["synthetic"]))
        metrics = json.loads((out / "subjects" / "S001.json").read_text())
        self.assertEqual(metrics["status"], "written")
        self.assertIn("Light", metrics["by_stage"])
        self.assertIn("counterfactual_ibi_change_event_samples_ms", metrics["events"])
        self.assertIsNotNone(metrics["overall"]["w1_ms"])
        rows = list(csv.DictReader((out / "subjects_summary.csv").read_text().splitlines()))
        self.assertEqual([r["subject"] for r in rows], ["S001", "S002"])
        # training goes on and replaces best.pt: the output keeps using its own frozen copy
        original = (self.run_dir / "best.pt").read_bytes()
        (self.run_dir / "best.pt").write_bytes(b"replaced by a later epoch")
        try:
            again = self.generate(out)                                          # continues with the others
        finally:
            (self.run_dir / "best.pt").write_bytes(original)
        self.assertEqual(again["summary"]["written"], 6)
        statuses = {r["subject"]: r["status"] for r in csv.DictReader((out / "subjects_summary.csv").read_text().splitlines())}
        self.assertEqual(statuses["S001"], "kept_existing")
        with np.load(out / "subjects" / "S001.npz") as z:
            np.testing.assert_array_equal(z["generated_ppg"], generated)
        with self.assertRaises(SystemExit):                                     # other options, same folder
            self.generate(out, "--seed", "7")
        twin = self.folder / "nights_twin"
        self.generate(twin, "--subjects", "S001")
        with np.load(twin / "subjects" / "S001.npz") as z:
            np.testing.assert_array_equal(z["generated_ppg"], generated)       # same seed, same night


if __name__ == "__main__":
    unittest.main()
