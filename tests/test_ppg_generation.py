"""Whole-night PPG generation with a tiny model trained for one epoch on a synthetic cohort (CPU)."""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import shutil
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

    def test_quality_measures_and_quality_only_run(self):
        import ppg_generation as pg
        from test_ppg_training import pulse_epochs

        clean = pulse_epochs(np.random.default_rng(5), 4, 0.85)
        noise = np.random.default_rng(6).normal(size=(4, 1500)).astype(np.float32)
        q_clean, q_noise = pg.quality_measures(clean), pg.quality_measures(noise, workers=2)
        self.assertGreater(np.nanmin(q_clean["templatematch"]), 0.9)
        self.assertLess(np.nanmedian(q_noise["templatematch"]), np.nanmedian(q_clean["templatematch"]))
        self.assertGreater(np.nanmedian(q_clean["relative_power"]), np.nanmedian(q_noise["relative_power"]))
        self.assertTrue(np.all(q_clean["beats"] >= 30))
        parallel = pg.quality_measures(np.tile(clean, (6, 1)), workers=3)
        np.testing.assert_allclose(parallel["templatematch"][:4], q_clean["templatematch"])
        scaled = pg.quality_measures(clean * 300.0)                               # raw E4-like amplitude
        for measure in pg.QUALITY_MEASURES:
            np.testing.assert_allclose(scaled[measure], q_clean[measure], rtol=1e-6, err_msg=measure)
        same = pg.compare_quality(q_clean, q_clean)
        self.assertEqual(same["templatematch"]["w1"], 0.0)
        self.assertEqual(same["templatematch_below_0.8"]["gen"], 0.0)

        out = self.folder / "quality_only"
        result = self.generate(out, "--subjects", "S001")
        self.assertEqual(result["status"], "completed")
        self.assertFalse((out / "night_npz").exists())                       # no IBI without --ibi
        with np.load(out / "subjects" / "S001.npz") as z:
            self.assertEqual(z["quality_gen_templatematch"].shape, (14,))
            self.assertEqual(z["quality_real_skewness"].shape, (14,))
            self.assertNotIn("ibi_generated", z.files)
            self.assertTrue((z["attempts"] == 1).all())                          # no regeneration by default
        metrics = json.loads((out / "subjects" / "S001.json").read_text())
        self.assertGreater(metrics["quality"]["templatematch"]["median_real"], 0.9)
        self.assertIn("Light", metrics["quality_by_stage"])
        row = next(csv.DictReader((out / "subjects_summary.csv").read_text().splitlines()))
        self.assertNotEqual(row["templatematch_median_gen"], "")
        self.assertEqual(row["w1_ms"], "")
        from scripts import plot_ppg_quality
        with contextlib.redirect_stdout(io.StringIO()):
            target = plot_ppg_quality.main([str(out)])
        self.assertTrue(Path(target).is_file())

    def test_n1n2_fold_run_generates_only_its_training_participants(self):
        from test_ppg_training import tiny_config
        from scripts import generate_ppg_nights, train_ppg

        data = self.folder / "ppg"
        config = tiny_config(data, self.folder / "participant_info.csv", self.folder / "runs",
                             stage_condition="n1n2", exclude_subjects=["S005"])
        path = self.folder / "tiny_n1n2.json"
        path.write_text(json.dumps(config))
        run = self.folder / "run_n1n2"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            train_ppg.main(["--config", str(path), "--output-dir", str(run), "--device", "cpu", "--allow-cpu"])
            out = self.folder / "nights_n1n2"
            args = ["--run", str(run), "--output-dir", str(out), "--device", "cpu", "--allow-cpu", "--batch-size", "8"]
            check = generate_ppg_nights.main(args + ["--check-only"])
            done = generate_ppg_nights.main(args + ["--subjects", "S001"])
        self.assertEqual(check["subjects"], 5)                                    # the test participant is left out
        self.assertEqual(done["status"], "completed")
        import torch
        weights = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        self.assertEqual(len(weights["channels"]), 12)
        self.assertIn("stage_N1", weights["channels"])
        with np.load(out / "subjects" / "S001.npz") as z:
            self.assertEqual(z["generated_ppg"].shape, (14, 1500))

    def test_nights_metrics_resume_and_frozen_checkpoint(self):
        out = self.folder / "nights"
        check = self.generate(out, "--check-only", "--ibi")
        self.assertEqual(check["subjects"], 6)
        first = self.generate(out, "--subjects", "S001", "S002", "--ibi", "--quality-threshold", "0.8")
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
            again = self.generate(out, "--ibi", "--quality-threshold", "0.8")    # continues with the others
        finally:
            (self.run_dir / "best.pt").write_bytes(original)
        self.assertEqual(again["summary"]["written"], 6)
        statuses = {r["subject"]: r["status"] for r in csv.DictReader((out / "subjects_summary.csv").read_text().splitlines())}
        self.assertEqual(statuses["S001"], "kept_existing")
        with np.load(out / "subjects" / "S001.npz") as z:
            np.testing.assert_array_equal(z["generated_ppg"], generated)
        with self.assertRaises(SystemExit):                                     # other options, same folder
            self.generate(out, "--seed", "7", "--ibi", "--quality-threshold", "0.8")
        other = self.run_dir / "epoch050.pt"
        shutil.copy2(self.run_dir / "best.pt", other)
        with self.assertRaises(SystemExit) as refused:                         # another checkpoint, same folder
            self.generate(out, "--checkpoint", "epoch050.pt", "--ibi", "--quality-threshold", "0.8")
        self.assertIn("use a new --output-dir", str(refused.exception))
        other.unlink()
        twin = self.folder / "nights_twin"
        self.generate(twin, "--subjects", "S001", "--ibi", "--quality-threshold", "0.8")
        with np.load(twin / "subjects" / "S001.npz") as z:
            np.testing.assert_array_equal(z["generated_ppg"], generated)       # same seed, same night


if __name__ == "__main__":
    unittest.main()
