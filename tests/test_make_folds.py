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


def write_people(folder, n=23):
    import pandas as pd

    data = folder / "ppg"
    data.mkdir()
    rng = np.random.default_rng(3)
    sids = [f"S{i:03d}" for i in range(1, n + 1)]
    for k, sid in enumerate(sids):
        stages = rng.choice([0, 1, 2, 4], 60)
        if k % 3 == 0:
            stages[10:25] = 3                                      # a third with deep sleep
        np.savez_compressed(data / f"{sid}.npz", schema=np.array("dreamt_ppg50_extraction_v4"),
                            stages=stages.astype(np.int8), ppg=np.zeros((60, 4), np.float32))
    np.savez(data / "S999.npz", schema=np.array("something else"), stages=np.zeros(3))
    ahi = rng.uniform(0, 80, n)
    pd.DataFrame(dict(SID=sids, AHI=ahi, BMI=rng.uniform(20, 40, n), AGE=rng.uniform(30, 80, n),
                      GENDER=["F", "M"] * (n // 2) + ["F"] * (n % 2))).to_csv(folder / "info.csv", index=False)
    config = json.loads((ROOT / "configs" / "ppg50.json").read_text())
    config.update(data_dir=str(data), metadata_path=str(folder / "info.csv"), output_root=str(folder / "runs"))
    path = folder / "base.json"
    path.write_text(json.dumps(config))
    return sids, path


class MakeFoldsTest(unittest.TestCase):
    def test_folds_are_disjoint_balanced_and_configs_exclude_the_test_fold(self):
        import ppg_training as pt
        from scripts import make_folds

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            sids, base = write_people(folder)
            out = folder / "folds"
            with contextlib.redirect_stdout(io.StringIO()):
                summary = make_folds.main(["--config", str(base), "--output-dir", str(out)])
            folds = summary["folds"]
            self.assertEqual(len(folds), 5)
            tests = [s for f in folds for s in f["test"]]
            self.assertEqual(sorted(tests), sids)                              # every participant tested once
            sizes = [len(f["test"]) for f in folds]
            self.assertLessEqual(max(sizes) - min(sizes), 1)
            deep = [sum(summary["participants"][s]["deep_epochs"] >= 10 for s in f["test"]) for f in folds]
            self.assertLessEqual(max(deep) - min(deep), 1)
            for f in folds:
                self.assertFalse(set(f["test"]) & set(f["generator_train"]))
                self.assertFalse(set(f["val"]) & set(f["train"]))
                self.assertEqual(sorted(f["test"] + f["generator_train"]), sids)
                self.assertEqual(sorted(f["train"] + f["val"]), f["generator_train"])
                self.assertIn(len(f["val"]), (2, 3))                           # about 10 % of 23
                config = pt.load_config(out / f"fold{f['fold']}.json")
                self.assertEqual(config["stage_condition"], "n1n2")
                self.assertEqual(config["exclude_subjects"], f["test"])
                self.assertEqual(config["training"]["epochs"], 150)
            with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
                make_folds.main(["--config", str(base), "--output-dir", str(out)])   # never overwritten
            again = folder / "again"
            with contextlib.redirect_stdout(io.StringIO()):
                same = make_folds.main(["--config", str(base), "--output-dir", str(again)])
            self.assertEqual(same["folds"], folds)                             # seeded


if __name__ == "__main__":
    unittest.main()
