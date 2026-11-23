"""DREAMT PPG extraction and the shared PPG -> IBI path on synthetic recordings with known beats."""
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
MOTION_EPOCH = 15                 # wrist motion in ACC
MISSED_EPOCH = 19                 # one PPG pulse removed (a missed beat) 10 s into this epoch
FLAT_EPOCH = 25                   # 8 s of flat BVP starting 10 s into this epoch (index 24 after Missing)


def r_peaks(duration, rng):
    times, t = [], 0.5
    while t < duration:
        times.append(t)
        t += 0.9 + 0.06 * np.sin(2 * np.pi * t / 25.0) + rng.normal(0, 0.02)
    return np.asarray(times)


def synthetic_frame(rng, *, ppg_shift=0.0, artifacts=True):
    epochs = len(STAGE_SEQUENCE)
    rows = OFFSET + epochs * 3000 + 777
    t = np.arange(rows) / FS
    beats = r_peaks(rows / FS, rng)
    missed_at = OFFSET / FS + 30 * MISSED_EPOCH + 10.0
    missed = int(np.argmin(np.abs(beats + PAT - missed_at))) if artifacts else -1
    ecg = np.zeros(rows)
    ppg = np.zeros(rows)
    for k, b in enumerate(beats):
        window = slice(max(0, int((b - 0.5) * FS)), min(rows, int((b + 1.2) * FS)))
        tt = t[window]
        ecg[window] += np.exp(-0.5 * ((tt - b) / 0.012) ** 2) + 0.2 * np.exp(-0.5 * ((tt - b - 0.25) / 0.04) ** 2)
        if k == missed:
            continue
        p = b + PAT + ppg_shift
        window = slice(max(0, int((p - 0.5) * FS)), min(rows, int((p + 1.0) * FS)))
        tt = t[window]
        ppg[window] += np.exp(-0.5 * ((tt - p) / 0.07) ** 2) + 0.35 * np.exp(-0.5 * ((tt - p - 0.28) / 0.08) ** 2)
    ecg += rng.normal(0, 0.02, rows)
    ppg = 50 * ppg + rng.normal(0, 0.5, rows) + 5 * np.sin(2 * np.pi * t / 200.0)
    if artifacts:
        flat = OFFSET + 3000 * FLAT_EPOCH + 1000
        ppg[flat:flat + 800] = ppg[flat]
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

    def test_parabolic_refinement(self):
        fs, true = 50, 3.013
        t = np.arange(0, 6, 1 / fs)
        signal = np.exp(-0.5 * ((t - true) / 0.07) ** 2)
        peak = int(np.argmax(signal))
        samples, times = px.refine_peaks_parabolic(signal, [peak, 0, len(t) - 1], fs)
        self.assertLess(abs(times[0] - true), 0.002)
        self.assertGreater(abs(peak / fs - true), 0.005)
        self.assertEqual(samples[1:].tolist(), [0.0, len(t) - 1.0])          # edges are left as they are
        flat = px.refine_peaks_parabolic(np.ones(10), [5], 100)[0]
        self.assertEqual(flat.tolist(), [5.0])

    def test_clean_intervals_fill_short_runs_and_zero_long_ones(self):
        options = px.cleaning_options()
        beats = np.r_[np.arange(0.0, 10.0, 1.0), np.arange(11.0, 20.0, 1.0)]   # beat at 10 s missed: 2-s interval
        ibi, status, fill = px.classify_intervals(beats, options)
        self.assertEqual(status.tolist().count(1), 1)
        times = np.arange(0, 19, 0.04)
        values, code = px.intervals_to_samples(times, beats, ibi, status, fill)
        inside = times < 18.96
        np.testing.assert_allclose(values[inside], 1.0)
        self.assertTrue((code[(times >= 9) & (times < 11)] == px.INTERPOLATED).all())
        self.assertTrue((code[(times < 9)] == px.MEASURED).all())
        # a 5-s gap is longer than 3 s: zero; three bad intervals in a row: zero
        long_gap = np.r_[np.arange(0.0, 10.0, 1.0), np.arange(15.0, 20.0, 1.0)]
        self.assertEqual(px.classify_intervals(long_gap, options)[1].tolist().count(2), 1)
        three = np.r_[np.arange(0.0, 10.0, 1.0), 9.1, 9.2, 9.3, np.arange(11.0, 20.0, 1.0)]
        status3 = px.classify_intervals(three, options)[1]
        self.assertGreaterEqual(status3.tolist().count(2), 3)
        self.assertEqual(status3.tolist().count(1), 0)
        # a bad run at the block edge has no good neighbour on one side: zero
        self.assertEqual(px.classify_intervals(np.r_[0.0, 0.1, np.arange(1.0, 6.0)], options)[1][0], 2)
        # no filling at all reproduces the WatchSleepNet rule
        none = px.cleaning_options(max_fill_intervals=0)
        self.assertEqual(px.classify_intervals(beats, none)[1].tolist().count(2), 1)

    def test_relative_rule_catches_in_range_extra_and_missed_beats(self):
        extra = np.sort(np.r_[np.arange(23) * 0.9, 9.45])                       # extra beat between 9.0 and 9.9
        range_only = px.classify_intervals(extra, px.cleaning_options())[1]
        self.assertEqual(int(range_only.sum()), 0)                              # 0.45-s halves are in range
        relative = px.classify_intervals(extra, px.cleaning_options(relative_threshold=0.3))
        self.assertEqual(relative[1].tolist().count(1), 2)
        missed = np.delete(np.arange(23) * 0.9, 10)                             # 1.8-s interval, in range
        self.assertEqual(int(px.classify_intervals(missed, px.cleaning_options())[1].sum()), 0)
        relative = px.classify_intervals(missed, px.cleaning_options(relative_threshold=0.3))
        self.assertEqual(relative[1].tolist().count(1), 1)
        times = np.arange(0, 19, 0.04)
        values, _ = px.intervals_to_samples(times, missed, *relative)
        np.testing.assert_allclose(values[times < missed[-1]], 0.9, atol=1e-9)

    def test_blocks_to_ibi_epoch_ok(self):
        beats = np.r_[np.arange(0.2, 12.0, 1.0), np.arange(40.0, 60.0, 1.0)]  # nothing from 12 to 40 s
        blocks = [dict(epochs=np.array([0, 1]), beat_times_s=beats)]
        out = px.blocks_to_ibi(blocks, np.array([0.0, 30.0]), px.cleaning_options())
        self.assertLess(out["valid_fraction"][0], 0.5)
        self.assertFalse(out["epoch_ok"][0])
        self.assertTrue(out["epoch_ok"][1])
        self.assertTrue((out["status"][0, :5] == px.NO_BEAT).all())          # before the first beat

    def test_matching_and_errors(self):
        ecg = np.arange(0, 20, 1.0)
        ppg = np.r_[ecg[:10] + 0.3, ecg[11:] + 0.3 + 0.004]
        matched = px.match_beats(ecg, ppg, 0.3)
        self.assertEqual(int((matched >= 0).sum()), 19)
        errors = px.ibi_errors(ecg, ppg, matched)
        self.assertEqual(errors.size, 17)
        self.assertLess(np.abs(errors).max(), 0.0041)
        self.assertEqual(px.flat_fraction(np.r_[np.zeros(30), np.arange(1, 71)]), 0.3)
        accuracy = px.series_accuracy(np.array([1.0, 0.0, 1.2, 1.0]), np.array([1.0, 1.0, 1.0, 0.0]))
        self.assertAlmostEqual(accuracy["coverage"], 2 / 3)
        self.assertAlmostEqual(accuracy["mae_ms"], 100.0)
        self.assertAlmostEqual(accuracy["big_error_fraction"], 0.5)

    def test_ecg_inversion_check(self):
        t = np.arange(0, 60, 0.01)
        ecg = sum(np.exp(-0.5 * ((t - b) / 0.012) ** 2) for b in np.arange(0.5, 60, 0.85)) \
            - 0.3 * sum(np.exp(-0.5 * ((t - b - 0.04) / 0.02) ** 2) for b in np.arange(0.5, 60, 0.85))
        self.assertFalse(px.ecg_is_inverted(ecg, 100))
        self.assertTrue(px.ecg_is_inverted(-ecg, 100))

    def test_tracking_follows_drift_and_fast_wander(self):
        rng = np.random.default_rng(0)
        ecg = np.cumsum(0.9 + 0.05 * np.sin(np.arange(6000) / 4.0) + rng.normal(0, 0.03, 6000))
        # slow drift, a back-and-forth of 0.4 s within 20 s, and a 2-s ramp over 2 minutes in the middle
        wander = 0.3 + 0.2 * ecg / ecg[-1] + 0.2 * np.sin(2 * np.pi * ecg / 40.0)
        wander += np.clip((ecg - 2000.0) / 60.0, 0, 2.0) - np.clip((ecg - 2300.0) / 60.0, 0, 2.0)
        ppg = ecg + wander + rng.normal(0, 0.01, ecg.size)
        track = px.track_beats(ecg, ppg)
        hit = track["matched"] >= 0
        self.assertGreater(hit.mean(), 0.95)
        self.assertGreater((track["matched"][hit] == np.flatnonzero(hit)).mean(), 0.95)   # the right beats
        np.testing.assert_allclose(track["lag"][hit], wander[hit], atol=0.05)
        accuracy = px.tracked_accuracy(ecg, ppg, track)
        self.assertLess(accuracy["median_abs_ms"], 25.0)
        times = np.linspace(ppg[10], ppg[-10], 1000)
        shift = px.lag_series(times, ppg, track)
        self.assertGreater(np.isfinite(shift).mean(), 0.9)

    def test_tracking_uses_a_far_prior_and_fixes_the_beat_identity(self):
        rng = np.random.default_rng(1)
        ecg = np.cumsum(0.9 + 0.08 * np.sin(np.arange(3000) / 3.0) + rng.normal(0, 0.04, 3000))
        ppg = ecg + 7.3 + rng.normal(0, 0.01, ecg.size)
        self.assertLess((px.track_beats(ecg, ppg)["matched"] >= 0).mean(), 0.5 + 0.5)   # runs, maybe off by beats
        track = px.track_beats(ecg, ppg, priors=(0.3, 7.2))
        hit = track["matched"] >= 0
        self.assertGreater(hit.mean(), 0.95)
        self.assertAlmostEqual(float(np.median(track["lag"][hit])), 7.3, delta=0.02)

    def test_ecg_rhythm_flags_bigeminy(self):
        sinus = np.cumsum(np.full(500, 0.9) + np.random.default_rng(2).normal(0, 0.02, 500))
        self.assertLess(px.ecg_rhythm(sinus), 0.05)
        bigeminy = np.cumsum(np.tile([0.49, 1.27], 250))
        self.assertGreater(px.ecg_rhythm(bigeminy), 0.95)

    def test_route_agreement(self):
        reference = np.arange(0, 100, 0.9)
        other = np.r_[reference[:50] + 0.002, reference[51:] - 0.001, 33.33]
        agreement = px.route_agreement(reference, other)
        self.assertAlmostEqual(agreement["reference_matched"], (reference.size - 1) / reference.size)
        self.assertLess(agreement["dt_p95_abs_ms"], 2.5)
        self.assertLess(agreement["ibi_p95_abs_ms"], 3.5)


@unittest.skipUnless(HAVE_NEUROKIT, "neurokit2 is not installed")
class PpgToIbiTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(0, 120, 1 / 50)
        self.beats = np.arange(0.4, 120, 0.8) + 0.0037
        self.signal = sum(np.exp(-0.5 * ((t - b) / 0.07) ** 2) for b in self.beats)

    def test_50_hz_input_is_detected_at_100_hz_and_joined_across_epochs(self):
        epochs = self.signal.reshape(4, 1500)
        joined = px.ppg_to_ibi(epochs, 50, np.array([0.0, 30.0, 60.0, 90.0]))
        found = joined["beat_times_s"]
        matched = px.match_beats(self.beats, found, 0.0, tolerance=0.05)
        self.assertGreater((matched >= 0).mean(), 0.98)
        self.assertLess(np.median(np.abs(found[matched[matched >= 0]] - self.beats[matched >= 0])), 0.002)
        np.testing.assert_allclose(np.median(joined["ibi"]), 0.8, atol=0.002)
        self.assertLess((joined["ibi"] == 0).mean(), 0.02)               # zeros only before the first beat
        self.assertTrue(joined["epoch_ok"].all())
        self.assertEqual(int(joined["beats_per_epoch"].sum()), found.size)
        self.assertGreater(float(joined["sqi"].min()), 0.95)
        gapped = px.ppg_to_ibi(epochs, 50, np.array([0.0, 30.0, 300.0, 330.0]))
        self.assertTrue((gapped["status"][1, -5:] == px.NO_BEAT).all())  # after the last beat before the gap
        self.assertTrue((gapped["ibi"][2, :5] == 0).all())

    def test_local_amplitude_keeps_beats_next_to_a_loud_artifact(self):
        rng = np.random.default_rng(3)
        fs, t = 100, np.arange(0, 600, 0.01)
        beats = np.cumsum(0.9 + rng.normal(0, 0.02, 700))
        beats = beats[beats < 599]
        signal = sum(np.exp(-0.5 * ((t - b) / 0.07) ** 2) + 0.3 * np.exp(-0.5 * ((t - b - 0.3) / 0.08) ** 2)
                     for b in beats) + 0.05 * rng.normal(size=t.size)
        loud = (t > 200) & (t < 300)
        signal[loud] += 40 * np.sin(2 * np.pi * 1.3 * t[loud]) * rng.uniform(0.5, 1.5, loud.sum())
        quiet = (beats < 190) | (beats > 310)
        found = {}
        for method in px.PEAK_METHODS:
            peaks = px.ppg_peaks(signal, fs, method=method)[1] / fs
            found[method] = (px.match_beats(beats[quiet], peaks, 0.0, tolerance=0.05) >= 0).mean()
        self.assertLess(found["elgendi"], 0.5)                 # one global threshold: quiet beats lost
        self.assertGreater(found["elgendi_local"], 0.98)
        self.assertGreater(found["charlton"], 0.98)
        with self.assertRaises(ValueError):
            px.ppg_peaks(signal, fs, method="other")

    def test_shapes_are_checked_and_flat_input_gives_zeros(self):
        with self.assertRaises(ValueError):
            px.ppg_to_ibi(np.zeros((2, 1000)), 50, np.array([0.0, 30.0]))
        with self.assertRaises(ValueError):
            px.ppg_to_ibi(np.zeros((1, 1500)), 50, np.array([0.0]), not_an_option=1)
        flat = px.ppg_to_ibi(np.zeros((1, 1500)), 50, np.array([0.0]))
        self.assertTrue((flat["ibi"] == 0).all())
        self.assertFalse(flat["epoch_ok"][0])


@unittest.skipUnless(HAVE_NEUROKIT, "neurokit2 is not installed")
class ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        cls.frame, cls.beats = synthetic_frame(np.random.default_rng(4))
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
        for key in ("ibi_ppg", "ibi_ppg_status", "ibi_ppg_relative", "ibi_ppg_raw", "ibi_ppg100", "ibi_wsn", "ibi_ecg"):
            self.assertEqual(a[key].shape, (len(expected), 750), key)
        self.assertEqual(a["events"].shape, (len(expected), 1500, 4))
        self.assertTrue(a["events"][12, :, 2].any() and not a["events"][11].any())
        np.testing.assert_allclose(a["epoch_start_s"][:2], [OFFSET / FS, OFFSET / FS + 30])
        self.assertEqual(a["in_sleep_period"].tolist().index(True), 6)
        self.assertEqual(r["epoch_runs"], 2)

    def test_alignment_and_beat_accuracy(self):
        r = self.report
        self.assertAlmostEqual(r["alignment"]["lag_median_s"], PAT, delta=0.03)
        self.assertGreater(r["alignment"]["tracked_fraction"], 0.95)
        self.assertFalse(r["alignment"]["ecg_irregular"])
        self.assertGreater(r["route_agreement_50_vs_100"]["reference_matched"], 0.97)
        self.assertLess(r["route_agreement_50_vs_100"]["ibi_median_abs_ms"], 3.0)
        beats = r["beat_accuracy"]
        self.assertGreater(beats["ppg50"]["sensitivity"], 0.97)
        self.assertLess(beats["ppg50"]["mae_ms"], 3.0)
        self.assertLess(beats["ppg50"]["mae_ms"], beats["ppg50_unrefined"]["mae_ms"])
        self.assertLess(beats["ppg100"]["mae_ms"], 3.0)
        self.assertIn("watchsleepnet", beats)
        for name in ("ppg50_elgendi", "ppg50_charlton"):
            self.assertGreater(beats[name]["sensitivity"], 0.95, name)
            self.assertIn(f"{name}_clean", r["series_accuracy"])
        self.assertEqual(r["peak_method"], "elgendi_local")

    def test_flat_segment_is_zeroed_and_missed_beat_only_fixed_by_the_relative_rule(self):
        a, series = self.arrays, self.report["series_accuracy"]
        start = OFFSET / FS
        t = a["epoch_start_s"][:, None] + np.arange(750)[None, :] / 25.0
        flat = (t > start + 30 * FLAT_EPOCH + 12.0) & (t < start + 30 * FLAT_EPOCH + 16.0)
        self.assertTrue((a["ibi_ppg_status"][flat] == px.ARTIFACT).all())
        self.assertTrue((a["ibi_ppg"][flat] == 0).all())
        missed = (t > start + 30 * MISSED_EPOCH + 9.7) & (t < start + 30 * MISSED_EPOCH + 9.9)
        self.assertGreater(float(np.median(a["ibi_ppg"][missed])), 1.5)          # 1.8 s is in range
        self.assertLess(float(np.median(a["ibi_ppg_relative"][missed])), 1.1)    # interpolated
        self.assertLess(series["ppg50_relative"]["big_error_fraction"], series["ppg50_clean"]["big_error_fraction"])
        self.assertLess(series["ppg50_clean"]["mae_ms"], 25.0)
        self.assertLess(series["reference_coverage"], 1.0)                     # no beats to track in the flat part
        self.assertGreater(series["ppg50_clean"]["coverage"], series["watchsleepnet"]["coverage"])
        self.assertTrue(a["epoch_ok"].all())

    def test_continuous_ibi_has_no_epoch_edge_zeros_unlike_watchsleepnet(self):
        a = self.arrays
        self.assertLess((a["ibi_ppg_status"][:, 0] == px.NO_BEAT).mean(), 0.1)
        self.assertGreater((a["ibi_wsn"][:, 0] == 0).mean(), 0.9)      # WatchSleepNet starts every epoch at 0
        true_rr = np.interp(a["epoch_start_s"] + 10.0, self.beats[1:], np.diff(self.beats))
        self.assertLess(np.median(np.abs(a["ibi_ecg"][:, 250] - true_rr)), 0.05)

    def test_quality_flags(self):
        a = self.arrays
        self.assertGreater(a["acc_std_g"][MOTION_EPOCH], 0.05)
        self.assertGreater(a["flat_fraction"][FLAT_EPOCH - 1], 0.2)          # index shifted by the Missing epoch
        self.assertGreater(float(np.median(a["sqi"])), 0.9)

    def test_script_flags_a_shifted_night_and_reports_failures(self):
        from scripts import extract_ppg
        out = self.folder / "out"
        cli = extract_ppg.parse_args(["--raw-dir", str(self.raw), "--output-dir", str(out), "--no-watchsleepnet"])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            summary = extract_ppg.run(cli)
        self.assertEqual(summary["subjects_failed"], ["S003"])
        self.assertAlmostEqual(summary["subjects"]["S001"]["alignment"]["lag_median_s"], PAT, delta=0.03)
        shifted = summary["subjects"]["S002"]
        self.assertAlmostEqual(shifted["alignment"]["coarse_hr_lag_s"], 7.0 + PAT, delta=0.5)
        self.assertAlmostEqual(shifted["alignment"]["lag_median_s"], 7.0 + PAT, delta=0.03)
        self.assertGreater(shifted["beat_accuracy"]["ppg50"]["sensitivity"], 0.95)
        self.assertTrue(summary["subjects"]["S004"]["alignment"]["ecg_inverted"])
        self.assertAlmostEqual(summary["subjects"]["S004"]["alignment"]["lag_median_s"], PAT, delta=0.03)
        rows = pd.read_csv(out / "report.csv")
        self.assertEqual(rows["subject"].tolist(), ["S001", "S002", "S003", "S004"])
        self.assertTrue(np.isfinite(rows.loc[0, "ppg50_clean_mae_ms"]))
        with np.load(out / "S001.npz") as saved:
            self.assertEqual(int(saved["fs_ppg"]), 50)
            self.assertEqual(json.loads(str(saved["report"]))["cleaning"]["max_fill_intervals"], 2)
        stamp = (out / "S001.npz").stat().st_mtime_ns
        args = ["--raw-dir", str(self.raw), "--output-dir", str(out), "--subjects", "S001", "--no-watchsleepnet"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            again = extract_ppg.run(extract_ppg.parse_args(args))
        self.assertEqual((out / "S001.npz").stat().st_mtime_ns, stamp)
        self.assertEqual(again["subjects"]["S001"]["status"], "kept_existing")
        (out / "report.json").unlink()                  # a job stopped before report.json: reuse the NPZ
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            third = extract_ppg.run(extract_ppg.parse_args(args))
        self.assertEqual((out / "S001.npz").stat().st_mtime_ns, stamp)
        self.assertEqual(third["subjects"]["S001"]["status"], "kept_existing")
        self.assertAlmostEqual(third["subjects"]["S001"]["alignment"]["lag_median_s"], PAT, delta=0.03)
        with np.load(out / "S001.npz") as saved:                  # an older version's file is redone
            arrays = {key: saved[key] for key in saved.files}
        arrays["schema"] = np.array("dreamt_ppg50_extraction_v3")
        np.savez_compressed(out / "S001.npz", **arrays)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fourth = extract_ppg.run(extract_ppg.parse_args(args))
        self.assertEqual(fourth["subjects"]["S001"]["status"], "written")
        with np.load(out / "S001.npz") as saved:
            self.assertEqual(str(saved["schema"]), px.SCHEMA)


if __name__ == "__main__":
    unittest.main()
