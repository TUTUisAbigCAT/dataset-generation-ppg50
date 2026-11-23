"""50 Hz PPG diffusion training on a small synthetic cohort with a tiny model on CPU."""
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

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_NEUROKIT = importlib.util.find_spec("neurokit2") is not None
SUBJECTS = {"S001": 5.0, "S002": 12.0, "S003": 30.0, "S004": 50.0, "S005": 70.0, "S092": 21.0}
IRREGULAR = {"S004"}


def pulse_epochs(rng, epochs, rr):
    t = np.arange(epochs * 1500) / 50.0
    beats = np.cumsum(rr + rng.normal(0, 0.02, int(t[-1] / rr) + 5))
    signal = sum(np.exp(-0.5 * ((t - b) / 0.07) ** 2) + 0.35 * np.exp(-0.5 * ((t - b - 0.28) / 0.08) ** 2)
                 for b in beats if b < t[-1] + 1)
    return (40 * signal + rng.normal(0, 0.5, t.size)).reshape(epochs, 1500).astype(np.float32)


def write_cohort(folder, epochs=14):
    import pandas as pd

    rng = np.random.default_rng(0)
    data = folder / "ppg"
    data.mkdir()
    report = {}
    for k, (sid, ahi) in enumerate(SUBJECTS.items()):
        stages = np.array([0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 2, 2, 0, 0][:epochs], dtype=np.int8)
        epoch_index = np.arange(epochs) + 3
        epoch_index[8:] += 1                                   # a gap after the eighth epoch
        epoch_ok = np.ones(epochs, bool)
        epoch_ok[5] = False
        events = np.zeros((epochs, 1500, 4), np.uint8)
        events[3, 200:900, 2] = 1
        np.savez_compressed(data / f"{sid}.npz", schema=np.array("dreamt_ppg50_extraction_v4"),
                            ppg=pulse_epochs(rng, epochs, 0.8 + 0.05 * k), stages=stages, epoch_index=epoch_index,
                            epoch_start_s=epoch_index * 30.0, epoch_ok=epoch_ok, events=events)
        report[sid] = dict(status="written", median_sqi=0.98, epoch_ok_fraction=0.97,
                           alignment=dict(ecg_irregular=sid in IRREGULAR, lag_median_s=0.4))
    (data / "report.json").write_text(json.dumps(dict(subjects=report)))
    np.savez(data / ".S999.partial.npz", x=np.zeros(1))
    pd.DataFrame(dict(SID=list(SUBJECTS), AHI=list(SUBJECTS.values()), BMI=[22, 25, 28, 31, 34, 26],
                      AGE=[30, 40, 50, 60, 70, 45], GENDER=["F", "M", "F", "M", "F", "M"])).to_csv(
        folder / "participant_info.csv", index=False)
    return data, folder / "participant_info.csv"


def tiny_config(data, metadata, output_root, **overrides):
    import ppg_training as pt

    config = dict(data_dir=str(data), metadata_path=str(metadata), output_root=str(output_root),
                  validation=dict(n_subjects=2, windows_per_subject=2, batch_size=4),
                  training=dict(epochs=1, batch_size=8, num_workers=0, log_every=1000, ema_decay=0.9),
                  model=dict(num_res_layers=1, res_channels=8, skip_channels=8, s4_dstate=8,
                             diff_step_embed_in=8, diff_step_embed_mid=8, diff_step_embed_out=8),
                  diffusion=dict(timesteps=4))
    for key, value in overrides.items():
        config[key] = value
    return pt.load_config(None, config)


class ConfigAndDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        cls.data, cls.metadata = write_cohort(cls.folder)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_config_rejects_unknown_keys_and_merges_defaults(self):
        import ppg_training as pt

        with self.assertRaises(ValueError):
            pt.load_config(None, dict(training=dict(epocs=3)))
        config = pt.load_config(ROOT / "configs" / "ppg50.json")
        self.assertEqual(config["validation"]["never"], ["S092"])
        self.assertEqual(config["window_epochs"], 1)
        self.assertEqual(pt.channel_names(config)[-1], "apnea_event")

    def test_normalization_and_window_layout(self):
        import ppg_training as pt

        x = pt.normalize_window(np.r_[np.sin(np.linspace(0, 40, 1500)) * 300 + 50], dict(
            low_percentile=1.0, high_percentile=99.0, clip=3.0))
        self.assertLess(abs(float(np.median(x))), 0.05)
        self.assertAlmostEqual(float(np.percentile(x, 99) - np.percentile(x, 1)), 2.0, places=2)
        self.assertTrue((pt.normalize_window(np.ones(1500), dict(low_percentile=1, high_percentile=99, clip=3)) == 0).all())
        config = tiny_config(self.data, self.metadata, self.folder / "runs")
        cohort = pt.PpgCohort(config)
        self.assertEqual(sorted(cohort.subjects), sorted(SUBJECTS))
        windows = cohort.windows(["S001"])
        self.assertEqual(len(windows), 13)                                      # epoch 5 is not ok
        cohort.fit_scaler(cohort.subjects)
        array = cohort.window_array("S001", 3)
        self.assertEqual(array.shape, (1500, 11))
        np.testing.assert_array_equal(array[0, 1:5], [0, 1, 0, 0])              # N2 -> Light
        np.testing.assert_array_equal(cohort.window_array("S001", 8)[0, 1:5], [0, 0, 1, 0])   # R -> REM
        np.testing.assert_array_equal(cohort.window_array("S001", 6)[0, 1:5], [0, 0, 0, 1])   # N3 -> Deep
        self.assertEqual(array[:, 10].sum(), 700)
        self.assertEqual(array[0, 8] + array[0, 9], 1.0)
        two = pt.load_config(None, dict(config, window_epochs=2))
        pairs = pt.PpgCohort(two).windows(["S001"])
        self.assertNotIn(("S001", 4), pairs)                                    # contains the bad epoch 5
        self.assertTrue(all(first + 1 != 8 for _, first in pairs))              # never across the gap

    def test_validation_subjects_skip_s092_and_irregular_rhythm(self):
        import ppg_training as pt

        config = tiny_config(self.data, self.metadata, self.folder / "runs",
                             validation=dict(n_subjects=3, windows_per_subject=4))
        cohort = pt.PpgCohort(config)
        chosen, info = pt.select_validation_subjects(cohort, config)
        self.assertNotIn("S092", chosen)
        self.assertNotIn("S004", chosen)
        self.assertEqual(chosen, ["S001", "S003", "S005"])                      # spread over AHI
        self.assertEqual(info["not_eligible"]["S092"], "never used for validation")
        cohort.report["S003"]["alignment"]["lag_median_s"] = 55.0                # a shifted E4 clock
        chosen, info = pt.select_validation_subjects(cohort, config)
        self.assertNotIn("S003", chosen)
        cohort.report["S003"]["alignment"]["lag_median_s"] = 0.4
        windows = pt.select_validation_windows(cohort, chosen, 4)
        self.assertEqual(len(windows), 12)
        explicit = pt.load_config(None, dict(config, validation=dict(subjects=["S092"])))
        with self.assertRaises(ValueError):
            pt.select_validation_subjects(cohort, explicit)


@unittest.skipUnless(HAVE_TORCH and HAVE_NEUROKIT, "torch and neurokit2 are needed")
class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        cls.data, cls.metadata = write_cohort(cls.folder)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_loss_and_sampler_keep_conditions(self):
        import torch
        import ppg_training as pt

        config = tiny_config(self.data, self.metadata, self.folder / "runs")
        model = pt.build_model(config, 11)
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            self.assertNotIn(0, tensor.stride(), name)                      # no aliased storage left
        copy = pt.build_model(config, 11)
        copy.load_state_dict(model.state_dict())                            # checkpoints load back
        schedule = pt.build_diffusion(config, torch.device("cpu"))
        batch = torch.randn(3, 1500, 11)
        loss = pt.diffusion_loss(model, batch, schedule)
        self.assertTrue(torch.isfinite(loss))
        out = pt.sample(model.eval(), batch, schedule, generator=torch.Generator().manual_seed(1))
        self.assertEqual(tuple(out.shape), (3, 1500))
        again = pt.sample(model, batch, schedule, generator=torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(out, again))

    def test_metrics_are_zero_for_identical_windows(self):
        import ppg_training as pt

        config = tiny_config(self.data, self.metadata, self.folder / "runs")
        cohort = pt.PpgCohort(config)
        cohort.fit_scaler(cohort.subjects)
        windows = [("S001", 0), ("S001", 2), ("S002", 0), ("S002", 2)]
        real = np.stack([cohort.window_array(*w)[:, 0] for w in windows])
        summary, per_subject = pt.validation_metrics(real, real, [s for s, _ in windows], config)
        self.assertEqual(summary["ibi_w1_ms"], 0.0)
        self.assertEqual(summary["std_gen"], summary["std_real"])
        self.assertGreater(per_subject["S001"]["beats_per_window_real"], 30)
        noise = np.random.default_rng(1).normal(size=real.shape).astype(np.float32)
        worse, _ = pt.validation_metrics(noise, real, [s for s, _ in windows], config)
        self.assertGreater(worse["ibi_w1_ms"], 50.0)

    def test_run_resume_and_pause(self):
        from scripts import train_ppg

        config = tiny_config(self.data, self.metadata, self.folder / "runs")
        config_path = self.folder / "tiny.json"
        config_path.write_text(json.dumps(config))
        out = self.folder / "run"
        with self.assertRaises(SystemExit) as missing:                        # nothing to resume yet
            train_ppg.main(["--resume", str(self.folder / "never_started"), "--device", "cpu", "--allow-cpu"])
        self.assertIn("no config.json", str(missing.exception))
        (out / "source").mkdir(parents=True)                                  # left by a run that never started
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            checked = train_ppg.main(["--config", str(config_path), "--output-dir", str(out), "--check-only"])
            self.assertEqual(checked["summary"]["validation_subjects"], ["S001", "S005"])
            first = train_ppg.main(["--config", str(config_path), "--output-dir", str(out), "--device", "cpu",
                                    "--allow-cpu"])
        self.assertEqual(first["status"], "completed")
        with self.assertRaises(SystemExit):                                     # a second new run there
            train_ppg.main(["--config", str(config_path), "--output-dir", str(out), "--device", "cpu", "--allow-cpu"])
        for name in ("config.json", "run_summary.json", "metrics.csv", "best.pt", "last.pt", "state.pt",
                     "validation/epoch_001.npz", "source/ppg_training.py"):
            self.assertTrue((out / name).exists(), name)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            second = train_ppg.main(["--resume", str(out), "--epochs", "3", "--device", "cpu", "--allow-cpu",
                                     "--max-hours", "1e-9"])
        self.assertEqual(second["status"], "paused")                            # out of time inside epoch 2
        self.assertEqual(second["epoch"], 1)
        self.assertIn("longer than --max-hours", second["note"])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            third = train_ppg.main(["--resume", str(out), "--device", "cpu", "--allow-cpu"])
        self.assertEqual(third["status"], "completed")
        rows = list(csv.DictReader((out / "metrics.csv").read_text().splitlines()))
        self.assertEqual([int(r["epoch"]) for r in rows], [1, 2, 3])
        self.assertTrue(all(int(r["steps"]) == int(r["steps_per_full_epoch"]) for r in rows))
        quick = self.folder / "quick"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            train_ppg.main(["--config", str(config_path), "--output-dir", str(quick), "--device", "cpu",
                            "--allow-cpu", "--max-steps-per-epoch", "2", "--validation-windows", "1"])
        quick_rows = list(csv.DictReader((quick / "metrics.csv").read_text().splitlines()))
        self.assertEqual(int(quick_rows[0]["steps"]), 2)
        self.assertGreater(float(quick_rows[0]["seconds_per_step"]), 0)
        with np.load(quick / "validation" / "epoch_001.npz") as saved:
            self.assertEqual(saved["generated"].shape, (2, 1500))              # 1 window x 2 participants
        self.assertTrue(all(np.isfinite(float(r["ibi_w1_ms"])) for r in rows))
        with np.load(out / "validation" / "epoch_003.npz") as saved:
            self.assertEqual(saved["generated"].shape, (4, 1500))
        summary = json.loads((out / "run_summary.json").read_text())
        self.assertNotIn("S092", summary["validation_subjects"])
        self.assertEqual(summary["training_subjects"], 6)                       # validation is not held out


if __name__ == "__main__":
    unittest.main()
