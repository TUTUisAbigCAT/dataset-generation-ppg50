"""DREAMT PPG extraction on synthetic 100 Hz recordings with known beats, lag and artifacts."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ppg_extraction as px

HAVE_NEUROKIT = importlib.util.find_spec("neurokit2") is not None
FS = 100
OFFSET = 1234                     # rows before the first PSG epoch
PAT = 0.30                        # true pulse arrival time (s)
STAGE_SEQUENCE = ["W"] * 6 + ["N1"] * 4 + ["N2"] * 12 + ["Missing"] + ["N2"] * 6 + ["N3"] * 8 + ["R"] * 6 + ["N2"] * 6 + ["W"] * 4
MOTION_EPOCH = 15


def r_peaks(duration, rng):
    times, t = [], 0.5
    while t < duration:
        times.append(t)
        t += 0.9 + 0.06 * np.sin(2 * np.pi * t / 25.0) + rng.normal(0, 0.02)
    return np.asarray(times)


def synthetic_frame(rng, *, ppg_shift=0.0):
    epochs = len(STAGE_SEQUENCE)
    rows = OFFSET + epochs * 3000 + 777
    t = np.arange(rows) / FS
    beats = r_peaks(rows / FS, rng)
    ecg = np.zeros(rows)
    ppg = np.zeros(rows)
    for b in beats:
        window = slice(max(0, int((b - 0.5) * FS)), min(rows, int((b + 1.2) * FS)))
        tt = t[window]
        ecg[window] += np.exp(-0.5 * ((tt - b) / 0.012) ** 2) + 0.2 * np.exp(-0.5 * ((tt - b - 0.25) / 0.04) ** 2)
        p = b + PAT + ppg_shift
        window = slice(max(0, int((p - 0.5) * FS)), min(rows, int((p + 1.0) * FS)))
        tt = t[window]
        ppg[window] += np.exp(-0.5 * ((tt - p) / 0.07) ** 2) + 0.35 * np.exp(-0.5 * ((tt - p - 0.28) / 0.08) ** 2)
    ecg += rng.normal(0, 0.02, rows)
    ppg = 50 * ppg + rng.normal(0, 0.5, rows) + 5 * np.sin(2 * np.pi * t / 200.0)
    stages = np.array(["P"] * rows, dtype=object)
    for k, label in enumerate(STAGE_SEQUENCE):
        stages[OFFSET + 3000 * k: OFFSET + 3000 * (k + 1)] = label
    stages[OFFSET + 3000 * epochs:] = "W"
    acc = rng.normal(0, 0.1, (rows, 3)) + np.array([0, 0, 64.0])
    motion = slice(OFFSET + 3000 * MOTION_EPOCH, OFFSET + 3000 * (MOTION_EPOCH + 1))
    acc[motion] += rng.normal(0, 20.0, (3000, 3))
    events = {column: np.full(rows, np.nan) for column in px.EVENT_COLUMNS}
    events["Hypopnea"][OFFSET + 3000 * 12 + 500: OFFSET + 3000 * 12 + 2500] = 1.0
    frame = pd.DataFrame(dict(TIMESTAMP=t, BVP=ppg, ECG=ecg, ACC_X=acc[:, 0], ACC_Y=acc[:, 1], ACC_Z=acc[:, 2],
                              Sleep_Stage=stages, **events))
    return frame, beats


class PureFunctionTests(unittest.TestCase):
    def test_epoch_grid_finds_the_psg_offset_and_skips_unscored_epochs(self):
        labels = ["P"] * 50 + ["W"] * 30 + ["N2"] * 20 + ["Missing"] * 10 + ["R"] * 10 + ["W"] * 7
        grid = px.epoch_grid(labels, epoch_rows=10)
        self.assertEqual(grid["offset"], 0)
        self.assertEqual(grid["codes"][grid["valid"]].tolist(), [0, 0, 0, 2, 2, 4])
        self.assertEqual(grid["on_grid_fraction"], 1.0)

    def test_sample_and_hold_ibi(self):
        values, flag = px.ibi_at([0.5, 1.5, 2.6, 4.0, -1], [0.0, 1.0, 2.5, 4.9], flagged=[False, True, False, False])
        np.testing.assert_allclose(values, [1.0, 1.5, 0.0, 0.0, 0.0])      # 2.4 s is outside 0.3-2.0 s
        self.assertEqual(flag.tolist(), [1, 1, 0, 0, 0])

    def test_refinement_finds_the_true_peak_between_samples(self):
        fs, true = 50, 3.013
        t = np.arange(0, 6, 1 / fs)
        signal = np.exp(-0.5 * ((t - true) / 0.07) ** 2)
        peak = int(np.argmax(signal))
        self.assertLess(abs(px.refine_peaks(signal, [peak], fs)[0] - true), 0.003)
        self.assertGreater(abs(peak / fs - true), 0.005)

    def test_matching_and_errors(self):
        ecg = np.arange(0, 20, 1.0)
        ppg = np.r_[ecg[:10] + 0.3, ecg[11:] + 0.3 + 0.004]          # one missed beat, 4 ms later after it
        matched = px.match_beats(ecg, ppg, 0.3)
        self.assertEqual(int((matched >= 0).sum()), 19)
        errors = px.ibi_errors(ecg, ppg, matched)
        self.assertEqual(errors.size, 17)
        self.assertLess(np.abs(errors).max(), 0.0041)
        self.assertEqual(px.flat_fraction(np.r_[np.zeros(30), np.arange(1, 71)]), 0.3)

    def test_ecg_inversion_check(self):
        t = np.arange(0, 60, 0.01)
        ecg = sum(np.exp(-0.5 * ((t - b) / 0.012) ** 2) for b in np.arange(0.5, 60, 0.85)) \
            - 0.3 * sum(np.exp(-0.5 * ((t - b - 0.04) / 0.02) ** 2) for b in np.arange(0.5, 60, 0.85))
        self.assertFalse(px.ecg_is_inverted(ecg, 100))
        self.assertTrue(px.ecg_is_inverted(-ecg, 100))

    def test_lag_track_follows_a_drifting_clock(self):
        rng = np.random.default_rng(0)
        ecg = np.cumsum(rng.uniform(0.8, 1.0, 4000))
        drift = 0.25 + 0.2 * ecg / ecg[-1]                          # 0.25 s -> 0.45 s over the night
        ppg = np.sort(ecg + drift + rng.normal(0, 0.008, ecg.size))
        track = px.lag_track(ecg, ppg, coarse=0.3, block_s=600.0, min_beats=300)
        self.assertGreater(track["concentration"], 0.2)
        self.assertGreater(track["track_lag_s"][-1] - track["track_lag_s"][0], 0.15)
        matched = px.match_beats(ecg, ppg, px.lag_at(ecg, track), tolerance=0.05)
        self.assertGreater((matched >= 0).mean(), 0.97)

    def test_mode_lag_ignores_wrapped_pairs(self):
        lags = np.r_[np.full(600, 0.31), np.random.default_rng(1).uniform(-0.2, 0.8, 300)]
        estimate, concentration = px.mode_lag(lags)
        self.assertAlmostEqual(estimate, 0.31, delta=0.005)
        self.assertGreater(concentration, 0.6)


@unittest.skipUnless(HAVE_NEUROKIT, "neurokit2 is not installed")
class PpgToIbiTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(0, 120, 1 / 50)
        beats = np.arange(0.4, 120, 0.8)
        self.signal = sum(np.exp(-0.5 * ((t - b) / 0.07) ** 2) for b in beats)

    def test_consecutive_epochs_join_and_gaps_reset(self):
        epochs = self.signal.reshape(4, 1500)
        joined = px.ppg_to_ibi(epochs, 50, np.array([0.0, 30.0, 60.0, 90.0]))
        np.testing.assert_allclose(np.median(joined["ibi"]), 0.8, atol=0.003)
        self.assertLess((joined["ibi"] == 0).mean(), 0.02)               # zeros only before the first beat
        gapped = px.ppg_to_ibi(epochs, 50, np.array([0.0, 30.0, 300.0, 330.0]))
        self.assertTrue((gapped["ibi"][1, -5:] == 0).all())               # after the last beat before the gap
        self.assertTrue((gapped["ibi"][2, :5] == 0).all())
        self.assertEqual(int(joined["beats_per_epoch"].sum()), joined["beat_times_s"].size)
        self.assertGreater(float(joined["sqi"].min()), 0.95)

    def test_shapes_are_checked_and_flat_input_gives_zeros(self):
        with self.assertRaises(ValueError):
            px.ppg_to_ibi(np.zeros((2, 1000)), 50, np.array([0.0, 30.0]))
        flat = px.ppg_to_ibi(np.zeros((1, 1500)), 50, np.array([0.0]))
        self.assertTrue((flat["ibi"] == 0).all())


@unittest.skipUnless(HAVE_NEUROKIT, "neurokit2 is not installed")
class ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        rng = np.random.default_rng(4)
        cls.frame, cls.beats = synthetic_frame(rng)
        cls.raw = cls.folder / "raw"
        cls.raw.mkdir()
        cls.frame.to_csv(cls.raw / "S001_whole_df.csv", index=False)
        shifted, _ = synthetic_frame(np.random.default_rng(5), ppg_shift=7.0)
        shifted.to_csv(cls.raw / "S002_whole_df.csv", index=False)
        cls.frame.drop(columns="BVP").to_csv(cls.raw / "S003_whole_df.csv", index=False)
        inverted = cls.frame.copy()
        inverted["ECG"] = -inverted["ECG"]
        inverted.to_csv(cls.raw / "S004_whole_df.csv", index=False)
        with contextlib.redirect_stderr(io.StringIO()):
            cls.arrays, cls.report = px.extract_subject(cls.raw / "S001_whole_df.csv", ahi=12.0)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_grid_stages_events_and_shapes(self):
        a, r = self.arrays, self.report
        self.assertEqual(r["grid_offset_rows"], OFFSET)
        expected = [px.STAGE_CODES[s] for s in STAGE_SEQUENCE if s != "Missing"]
        self.assertEqual(a["stages"].tolist(), expected)
        self.assertEqual(a["ppg"].shape, (len(expected), 1500))
        self.assertEqual(a["ibi_ppg"].shape, (len(expected), 750))
        self.assertEqual(a["events"].shape, (len(expected), 1500, 4))
        self.assertTrue(a["events"][12, :, 2].any() and not a["events"][11].any())
        np.testing.assert_allclose(a["epoch_start_s"][:2], [OFFSET / FS, OFFSET / FS + 30])
        self.assertEqual(a["in_sleep_period"].tolist().index(True), 6)

    def test_alignment_matching_and_refinement_accuracy(self):
        r = self.report
        self.assertTrue(r["alignment"]["aligned"])
        self.assertAlmostEqual(r["alignment"]["pulse_arrival_s"], PAT, delta=0.03)
        acc = r["accuracy"]
        self.assertGreater(acc["ppg50_refined"]["sensitivity"], 0.97)
        self.assertGreater(acc["ppg50_refined"]["ppv"], 0.97)
        self.assertLess(acc["ppg50_refined"]["mae_ms"], 8.0)
        self.assertLess(acc["ppg50_refined"]["mae_ms"], acc["ppg50_unrefined"]["mae_ms"])
        self.assertIn("watchsleepnet", acc)

    def test_continuous_ibi_has_no_epoch_edge_zeros_unlike_watchsleepnet(self):
        a = self.arrays
        self.assertLess((a["ibi_ppg"] == 0).mean(), 0.01)
        self.assertGreater((a["ibi_wsn"][:, 0] == 0).mean(), 0.9)      # WatchSleepNet starts every epoch at 0
        true_rr = np.interp(a["epoch_start_s"] + 10.0, self.beats[1:], np.diff(self.beats))
        self.assertLess(np.median(np.abs(a["ibi_ecg"][:, 250] - true_rr)), 0.05)

    def test_motion_epoch_is_not_good(self):
        a = self.arrays
        position = MOTION_EPOCH                      # before the Missing epoch, so index unchanged
        self.assertGreater(a["acc_std_g"][position], 0.05)
        self.assertFalse(a["good_epoch"][position])
        self.assertGreater(a["good_epoch"].mean(), 0.8)
        self.assertGreater(float(np.median(a["sqi"])), 0.9)

    def test_script_flags_a_shifted_night_and_reports_failures(self):
        from scripts import extract_ppg
        out = self.folder / "out"
        cli = extract_ppg.parse_args(["--raw-dir", str(self.raw), "--output-dir", str(out), "--no-watchsleepnet"])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            summary = extract_ppg.run(cli)
        self.assertEqual(summary["subjects_failed"], ["S003"])
        self.assertTrue(summary["subjects"]["S001"]["alignment"]["aligned"])
        shifted = summary["subjects"]["S002"]
        self.assertFalse(shifted["alignment"]["aligned"])                 # 7.3 s is not a pulse arrival time
        self.assertAlmostEqual(shifted["alignment"]["coarse_hr_lag_s"], 7.0 + PAT, delta=0.5)
        self.assertAlmostEqual(shifted["alignment"]["pulse_arrival_s"], 7.0 + PAT, delta=0.03)
        self.assertGreater(shifted["accuracy"]["ppg50_refined"]["sensitivity"], 0.95)
        self.assertTrue(summary["subjects"]["S004"]["alignment"]["ecg_inverted"])
        self.assertAlmostEqual(summary["subjects"]["S004"]["alignment"]["pulse_arrival_s"], PAT, delta=0.03)
        rows = pd.read_csv(out / "report.csv")
        self.assertEqual(rows["subject"].tolist(), ["S001", "S002", "S003", "S004"])
        with np.load(out / "S001.npz") as saved:
            self.assertEqual(int(saved["fs_ppg"]), 50)
            json.loads(str(saved["report"]))
        stamp = (out / "S001.npz").stat().st_mtime_ns
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            again = extract_ppg.run(extract_ppg.parse_args(["--raw-dir", str(self.raw), "--output-dir", str(out),
                                                            "--subjects", "S001", "--no-watchsleepnet"]))
        self.assertEqual((out / "S001.npz").stat().st_mtime_ns, stamp)
        self.assertEqual(again["subjects"]["S001"]["status"], "kept_existing")
        (out / "report.json").unlink()                  # a job stopped before report.json: reuse the NPZ
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            third = extract_ppg.run(extract_ppg.parse_args(["--raw-dir", str(self.raw), "--output-dir", str(out),
                                                            "--subjects", "S001", "--no-watchsleepnet"]))
        self.assertEqual((out / "S001.npz").stat().st_mtime_ns, stamp)
        self.assertEqual(third["subjects"]["S001"]["status"], "kept_existing")
        self.assertTrue(third["subjects"]["S001"]["alignment"]["aligned"])


if __name__ == "__main__":
    unittest.main()
