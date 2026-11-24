from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import neurokit2  # noqa: F401
    HAVE_NEUROKIT = True
except ImportError:
    HAVE_NEUROKIT = False


def pulses(n_epochs, period, rng):
    t = np.arange(n_epochs * 1500) / 50.0
    beats = np.cumsum(period + rng.normal(0, 0.02, int(n_epochs * 30 / period) + 5))
    beats = beats[beats < t[-1]]
    signal = sum(np.exp(-0.5 * ((t - b) / 0.07) ** 2) for b in beats)
    return signal.reshape(n_epochs, 1500)


@unittest.skipUnless(HAVE_NEUROKIT, "neurokit2 is not installed")
class NightsToIbiTest(unittest.TestCase):
    def test_conversion_comparison_and_night_files(self):
        from scripts import ppg_nights_to_ibi as conv

        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            nights, data = Path(tmp) / "nights", Path(tmp) / "data"
            (nights / "subjects").mkdir(parents=True)
            data.mkdir()
            (nights / "generation.json").write_text(json.dumps(dict(data_dir=str(data), checkpoint=dict(epoch=146))))
            for k, sid in enumerate(("S001", "S002", "S003")):
                n = 12
                real = pulses(n + 2, 0.9, rng)                      # extraction keeps two extra epochs
                gen = pulses(n, 0.8, rng)                           # generated: faster heart rate
                index = np.arange(2, n + 2)
                stages = np.array([0, 1, 2, 2, 3, 4] * 2)
                apnea = np.zeros((n, 750), np.uint8)
                apnea[3, 200:500] = 1
                np.savez_compressed(nights / "subjects" / f"{sid}.npz", generated_ppg=gen.astype(np.float32),
                                    epoch_index=index, stages=stages, apnea_25hz=apnea)
                np.savez_compressed(data / f"{sid}.npz", ppg=real.astype(np.float32), epoch_index=np.arange(n + 2),
                                    epoch_start_s=np.arange(n + 2) * 30.0,
                                    report=np.array(json.dumps(dict(participant_ahi=10.0 + k))))
            with contextlib.redirect_stdout(io.StringIO()):
                summary = conv.main(["--nights", str(nights), "--workers", "2"])
            out = nights / "ibi_elgendi_local_fix"
            self.assertEqual(summary["participants"], 3)
            self.assertEqual(summary["failed"], [])
            pooled = summary["pooled"]["ALL"]
            self.assertAlmostEqual(pooled["median_real_ms"], 900, delta=40)
            self.assertAlmostEqual(pooled["median_gen_ms"], 800, delta=40)
            self.assertAlmostEqual(pooled["w1_ms"], 100, delta=40)
            with np.load(out / "subjects" / "S001.npz") as z:
                self.assertEqual(z["ibi_generated"].shape, (12, 750))
                self.assertEqual(z["ibi_real"].shape, (12, 750))
            with np.load(out / "night_npz" / "S002.npz") as z:
                self.assertEqual(z["data"].shape, (12 * 750, 1))
                self.assertTrue(bool(z["synthetic"]))
                self.assertEqual(float(z["ahi"]), 11.0)
                self.assertEqual(z["stages"].size, 12 * 750)
            with np.load(out / "night_npz_real" / "S002.npz") as z:
                self.assertFalse(bool(z["synthetic"]))
            self.assertTrue((out / "ibi_comparison.png").is_file())
            self.assertEqual(len((out / "ibi_summary.csv").read_text().splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
