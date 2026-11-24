from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import analyze_ppg_quality as aq  # noqa: E402


def real_quality(rng, n):
    return dict(templatematch=np.clip(1 - rng.gamma(1.0, 0.06, n), 0.3, 1.0),
                disimilarity=rng.lognormal(np.log(0.003), 0.8, n), skewness=rng.normal(-0.25, 0.3, n),
                kurtosis=rng.normal(-0.9, 0.5, n), relative_power=np.clip(rng.normal(0.75, 0.12, n), 0, 1),
                beats=rng.integers(25, 40, n).astype(float))


def write_folder(folder, generated, seed=1, subjects=8, epochs=150):
    rng = np.random.default_rng(seed)
    (folder / "subjects").mkdir(parents=True)
    for k in range(subjects):
        stages = rng.choice([0, 1, 2, 3, 4], size=epochs, p=[0.2, 0.1, 0.45, 0.1, 0.15])
        real = real_quality(np.random.default_rng(100 + k), epochs)
        gen = generated(np.random.default_rng(200 + k), real, stages)
        gen["templatematch"][:3] = np.nan                                   # windows without enough beats
        arrays = dict(stages=stages.astype(np.int8), epoch_index=np.arange(epochs))
        arrays.update({f"quality_real_{m}": v.astype(np.float32) for m, v in real.items()})
        arrays.update({f"quality_gen_{m}": v.astype(np.float32) for m, v in gen.items()})
        np.savez_compressed(folder / "subjects" / f"S{k + 1:03d}.npz", **arrays)


def same_distribution(rng, real, stages):
    return real_quality(rng, stages.size)


def worse(rng, real, stages):
    gen = real_quality(rng, stages.size)
    gen["templatematch"] = np.clip(gen["templatematch"] - 0.15, 0.3, 1.0)
    gen["beats"] = gen["beats"] + 4
    return gen


class QualityAnalysisTest(unittest.TestCase):
    def test_tables_by_stage_and_c2st(self):
        with tempfile.TemporaryDirectory() as tmp:
            same, shifted = Path(tmp) / "same", Path(tmp) / "shifted"
            write_folder(same, same_distribution)
            write_folder(shifted, worse)
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                result = aq.main([str(same), str(shifted), "--labels", "same", "shifted"])
            first, second = result["results"]
            self.assertEqual(set(first), {"ALL", "W", "N1", "N2", "Light", "Deep", "REM"})
            self.assertEqual(first["ALL"]["windows"], 8 * 150)
            self.assertEqual(first["ALL"]["windows"], sum(first[s]["windows"] for s in ("W", "N1", "N2", "Deep", "REM")))
            self.assertLess(abs(first["ALL"]["c2st"]["auc"] - 0.5), 0.06)       # same distributions
            self.assertGreater(second["ALL"]["c2st"]["auc"], 0.8)                 # shifted distributions
            self.assertGreater(second["ALL"]["bands"]["gen"]["poor"], first["ALL"]["bands"]["gen"]["poor"])
            self.assertAlmostEqual(first["ALL"]["bands"]["gen"]["no_score"], 24 / 1200)
            agreement = first["ALL"]["same_epoch"]
            self.assertAlmostEqual(sum(agreement[k] for k in ("both_ok", "added", "removed", "both_poor")), 1.0)
            self.assertEqual(first["ALL"]["participants"]["subjects"], 8)
            self.assertIn("C2ST AUC", printed.getvalue())
            self.assertTrue((same / "quality_by_stage.png").is_file())
            rows = list(csv.DictReader((same / "quality_by_stage.csv").read_text().splitlines()))
            self.assertEqual(len(rows), 14)
            self.assertEqual({r["run"] for r in rows}, {"same", "shifted"})
            report = json.loads((same / "quality_analysis.json").read_text())
            self.assertEqual(set(report["runs"]), {"same", "shifted"})

    def test_same_epoch_agreement(self):
        gen = np.array([0.95, 0.70, 0.95, 0.70, np.nan])
        real = np.array([0.95, 0.95, 0.70, 0.70, 0.95])
        s = aq.same_epoch(gen, real)
        self.assertAlmostEqual(s["added"], 2 / 5)                                 # no score counts as poor
        self.assertAlmostEqual(s["removed"], 1 / 5)
        self.assertAlmostEqual(s["added_given_real_ok"], 2 / 3)
        self.assertAlmostEqual(s["removed_given_real_poor"], 1 / 2)


if __name__ == "__main__":
    unittest.main()
