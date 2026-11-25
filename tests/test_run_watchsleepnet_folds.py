"""Needs WatchSleepNet's modeling folder: set WATCHSLEEPNET_DIR (skipped otherwise)."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WSN = os.environ.get("WATCHSLEEPNET_DIR")
TINY = dict(num_channels=8, hidden_dim=8, num_heads=2, num_layers=1, tcn_layers=1, kernel_size=3)


@unittest.skipUnless(WSN and Path(WSN, "engine.py").is_file(), "set WATCHSLEEPNET_DIR to WatchSleepNet's modeling folder")
class RunWatchSleepNetFoldsTest(unittest.TestCase):
    def test_folds_runs_and_summary(self):
        from scripts import make_replacement_sets, run_watchsleepnet_folds
        from test_make_replacement_sets import night

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            wsn = folder / "modeling"                      # a 4-class copy, like the one used for DREAMT
            shutil.copytree(WSN, wsn, ignore=shutil.ignore_patterns("__pycache__"))
            setup = (wsn / "data_setup.py").read_text()
            setup = setup.replace("label_map = {0:0, 1:1, 2:1, 3:1, 4:2,", "label_map = {0:0, 1:1, 2:1, 3:3, 4:2,", 1)
            (wsn / "data_setup.py").write_text(setup)
            rng = np.random.default_rng(0)
            sids = [f"S{i:03d}" for i in range(1, 9)]
            real, gen = folder / "real", folder / "gen_fold0"
            real.mkdir(); gen.mkdir()
            for sid in sids:
                stages = rng.choice([0, 1, 2, 3, 4], 30)
                np.savez_compressed(real / f"{sid}.npz", **night(stages, 0.0, rng))
                np.savez_compressed(gen / f"{sid}.npz", **night(stages, 0.05, rng))
            (folder / "gen_fold1").symlink_to(gen)
            folds = [dict(fold=0, train=sids[:4], val=sids[4:6], test=sids[6:]),
                     dict(fold=1, train=sids[2:6], val=sids[6:], test=sids[:2])]
            (folder / "folds.json").write_text(json.dumps(dict(folds=folds)))
            sets = folder / "sets"
            with contextlib.redirect_stdout(io.StringIO()):
                make_replacement_sets.main(["--folds", str(folder / "folds.json"), "--real", str(real),
                                            "--generated", str(folder / "gen_fold{fold}"), "--output-dir", str(sets),
                                            "--arms", "real", "gen:REM"])
            common = ["--wsn-dir", str(wsn), "--folds", str(folder / "folds.json"), "--sets", str(sets),
                      "--output-dir", str(folder / "results"), "--num-workers", "0", "--device", "cpu"]
            with contextlib.redirect_stdout(io.StringIO()):
                check = run_watchsleepnet_folds.main(common + ["--check", "--batch-size", "2"])
                summary = run_watchsleepnet_folds.main(common + ["--seeds", "0", "1", "--epochs", "2", "--patience", "1",
                                                                 "--batch-size", "2", "--quiet",
                                                                 "--model-params", json.dumps(TINY)])
            sys.path.remove(str(wsn))
            for name in ("config", "data_setup", "engine"):
                sys.modules.pop(name, None)
            self.assertEqual(check["classes"], ["W", "Light", "REM", "Deep"])
            self.assertEqual(sorted(summary), ["gen_REM", "real"])
            self.assertIn("diff_vs_real_mean", summary["gen_REM"]["kappa"])
            result = json.loads((folder / "results" / "gen_REM" / "seed1" / "results.json").read_text())
            self.assertEqual(result["pooled"]["epochs"], 4 * 30)                 # two test nights per fold
            self.assertEqual(len(result["pooled"]["f1"]), 4)
            self.assertTrue(all(f["checkpoint_found"] for f in result["per_fold"]))
            self.assertTrue((folder / "results" / "summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
