from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ppg_extraction as px


def windows(stages, hr, rsa, rng):
    """25 Hz IBI of independent 30-s windows from beat times, as ppg_nights_to_ibi.py stores them."""
    blocks, starts = [], np.arange(len(stages)) * 1000.0
    for k, stage in enumerate(stages):
        t = [starts[k] - 1.0]
        while t[-1] < starts[k] + 31.0:
            t.append(t[-1] + 60.0 / hr[stage] + rsa * np.sin(2 * np.pi * 0.25 * t[-1]) + rng.normal(0, 0.008))
        blocks.append(dict(epochs=np.array([k]), beat_times_s=np.array(t)))
    out = px.blocks_to_ibi(blocks, starts, px.cleaning_options())
    return out["ibi"], out["status"], out["epoch_ok"]


class IbiKdeSpectrumTest(unittest.TestCase):
    def test_kde_spectra_and_stage_metrics(self):
        from scripts import ibi_kde_spectrum as ks

        rng = np.random.default_rng(1)
        stages = np.array([0] * 4 + [1] * 6 + [2] * 30 + [3] * 10 + [4] * 6 + [2] * 10)    # 66 epochs, 33 min
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "ibi"
            (folder / "subjects").mkdir(parents=True)
            for sid in ("S001", "S002", "S003"):
                real = windows(stages, {0: 70, 1: 64, 2: 60, 3: 54, 4: 66}, 0.05, rng)   # RSA at 0.25 Hz
                gen = windows(stages, {0: 70, 1: 64, 2: 60, 3: 60, 4: 66}, 0.0, rng)    # no RSA, no deep drop
                np.savez_compressed(folder / "subjects" / f"{sid}.npz", ibi_generated=gen[0],
                                    ibi_generated_status=gen[1], ibi_real=real[0], ibi_real_status=real[1],
                                    epoch_ok_generated=gen[2], epoch_ok_real=real[2], stages=stages,
                                    epoch_index=np.arange(len(stages)), apnea_25hz=np.zeros((len(stages), 750), np.uint8),
                                    peak_method=np.array("elgendi_local_fix"))
            with contextlib.redirect_stdout(io.StringIO()):
                summary = ks.main(["--ibi", str(folder), "--workers", "2"])
            out = folder / "kde_spectrum"
            for name in ("kde_ibi_hr.png", "spectrum_night.png", "spectrum_stage.png", "stage_hr_hrv.png",
                         "stage_contrast.png", "participant_stage_metrics.csv", "night_band_power.csv",
                         "psd_stage.csv", "kde_spectrum_summary.json"):
                self.assertTrue((out / name).is_file(), name)
            self.assertEqual(summary["participants"], 3)
            deep = summary["stage_metrics"]["Deep"]["hr_bpm"]
            self.assertAlmostEqual(deep["real_median"], 54, delta=1.5)
            self.assertAlmostEqual(deep["gen_median"], 60, delta=1.5)
            n2 = summary["stage_metrics"]["N2"]
            self.assertLess(n2["hf_ms2"]["median_diff"], -6)                     # generated lacks the RSA
            self.assertAlmostEqual(n2["hf_centre_bpm"]["real_median"], 15, delta=1.5)
            self.assertGreater(n2["rmssd_ms"]["real_median"], n2["rmssd_ms"]["gen_median"])
            self.assertLess(summary["night_spectrum"]["HF"]["median_diff"], -3)
            contrast = summary["stage_contrasts"]["Deep"]["hr_bpm"]
            self.assertAlmostEqual(contrast["real_median"], -6, delta=1.5)
            self.assertAlmostEqual(contrast["gen_median"], 0, delta=1.5)
            kde = summary["kde"]["ALL"]["ibi_ms"]
            self.assertGreater(kde["w1"], 0)

    def test_rmssd_uses_successive_measured_intervals(self):
        from scripts import ibi_kde_spectrum as ks

        values = np.repeat([0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9], 20).astype(np.float32)
        status = np.zeros(values.size, np.uint8)
        self.assertAlmostEqual(ks.window_rmssd(values, status), 0.1, places=5)
        status[100:140] = 1                                     # filled samples break the pairs around them
        self.assertTrue(np.isnan(ks.window_rmssd(values, status, min_pairs=10)))


if __name__ == "__main__":
    unittest.main()
