"""Needs MambaSleepNet's modeling folder: set MAMBASLEEPNET_DIR (skipped otherwise).

The CUDA Mamba kernel is replaced by a small causal stand-in (a fake mamba_ssm package); everything else,
including the CNN and BiMamba blocks of models/mamba_sleepnet.py, is the real code.
"""
from __future__ import annotations

import contextlib
import csv
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
SOURCE = os.environ.get("MAMBASLEEPNET_DIR")
FAKE_MAMBA = '''
import torch
from torch import nn


class Mamba(nn.Module):
    """Causal stand-in with Mamba's constructor: depthwise causal conv + gated projection."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, d_conv, groups=d_model, padding=d_conv - 1)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x):
        y = self.conv(x.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        return y * torch.sigmoid(self.gate(x))
'''


@unittest.skipUnless(SOURCE and Path(SOURCE, "models", "mamba_sleepnet.py").is_file(),
                     "set MAMBASLEEPNET_DIR to the modeling folder holding models/mamba_sleepnet.py")
class RunMambaSleepNetFoldsTest(unittest.TestCase):
    def test_folds_runs_and_summary(self):
        from scripts import make_replacement_sets, run_mambasleepnet_folds as run
        from test_make_replacement_sets import night

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            modeling = folder / "modeling" / "models"
            modeling.mkdir(parents=True)
            (modeling / "__init__.py").write_text("")
            shutil.copy2(Path(SOURCE, "models", "mamba_sleepnet.py"), modeling / "mamba_sleepnet.py")
            fake = folder / "fake" / "mamba_ssm"
            fake.mkdir(parents=True)
            (fake / "__init__.py").write_text(FAKE_MAMBA)
            sys.path.insert(0, str(folder / "fake"))
            rng = np.random.default_rng(0)
            sids = [f"S{i:03d}" for i in range(1, 9)]
            real, gen = folder / "real", folder / "gen_fold0"
            real.mkdir(); gen.mkdir()
            for k, sid in enumerate(sids):
                stages = rng.choice([0, 1, 2, 3, 4], 24 + k)                     # nights of different length
                np.savez_compressed(real / f"{sid}.npz", **night(stages, 0.0, rng))
                np.savez_compressed(gen / f"{sid}.npz", **night(stages, 0.05, rng))
            (folder / "gen_fold1").symlink_to(gen)
            folds = [dict(fold=0, train=sids[:4], val=sids[4:6], test=sids[6:]),
                     dict(fold=1, train=sids[2:6], val=sids[6:], test=sids[:2])]
            (folder / "folds.json").write_text(json.dumps(dict(folds=folds)))
            sets = folder / "sets"
            common = ["--mamba-dir", str(folder / "modeling"), "--folds", str(folder / "folds.json"),
                      "--sets", str(sets), "--output-dir", str(folder / "results"), "--num-workers", "0",
                      "--device", "cpu", "--hidden-dim", "16", "--num-layers", "1", "--d-state", "4",
                      "--batch-size", "2"]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    make_replacement_sets.main(["--folds", str(folder / "folds.json"), "--real", str(real),
                                                "--generated", str(folder / "gen_fold{fold}"),
                                                "--output-dir", str(sets), "--arms", "real", "gen:REM"])
                    check = run.main(common + ["--check"])
                    summary = run.main(common + ["--seeds", "0", "1", "--epochs", "3", "--patience", "2"])
                    again = run.main(common + ["--seeds", "0", "--arms", "real"])      # finished runs are kept
            finally:
                sys.path.remove(str(folder / "fake"))
                sys.path.remove(str(folder / "modeling"))
                for name in ("mamba_ssm", "models", "models.mamba_sleepnet"):
                    sys.modules.pop(name, None)
            self.assertEqual(check["arms"], ["gen_REM", "real"])
            self.assertEqual(sorted(summary), ["gen_REM", "real"])
            self.assertIn("diff_vs_real_mean", summary["gen_REM"]["kappa"])
            self.assertEqual(again["real"]["seeds"], [0, 1])
            out = folder / "results" / "gen_REM" / "seed1"
            result = json.loads((out / "results.json").read_text())
            self.assertEqual(result["classes"], ["W", "Light", "REM", "Deep"])
            self.assertEqual(result["pooled"]["epochs"], (24 + 6) + (24 + 7) + 24 + 25)   # every test epoch
            self.assertEqual(len(result["pooled"]["f1"]), 4)
            self.assertEqual(np.sum(result["pooled"]["confusion"]), result["pooled"]["epochs"])
            log = list(csv.DictReader((out / "fold0_training.csv").read_text().splitlines()))
            self.assertEqual(log[0]["saved"], "True")                            # the first epoch always
            self.assertTrue((out / "fold1.pt").is_file())
            people = list(csv.DictReader((out / "per_participant.csv").read_text().splitlines()))
            self.assertEqual(sorted(p["subject"] for p in people), sorted(sids[6:] + sids[:2]))

    def test_padding_does_not_change_predictions(self):
        import torch
        from scripts import run_mambasleepnet_folds as run

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "modeling" / "models").mkdir(parents=True)
            (folder / "modeling" / "models" / "__init__.py").write_text("")
            shutil.copy2(Path(SOURCE, "models", "mamba_sleepnet.py"), folder / "modeling" / "models" / "mamba_sleepnet.py")
            (folder / "fake" / "mamba_ssm").mkdir(parents=True)
            (folder / "fake" / "mamba_ssm" / "__init__.py").write_text(FAKE_MAMBA)
            sys.path.insert(0, str(folder / "fake"))
            try:
                blocks = run.import_blocks(folder / "modeling")
                torch.manual_seed(0)
                model = run.build_model(blocks, hidden_dim=16, num_layers=2, d_state=4).eval()
                short, long = torch.randn(1, 5, 750), torch.randn(1, 9, 750)
                batch = torch.zeros(2, 9, 750)
                batch[0, :5], batch[1] = short[0], long[0]
                with torch.no_grad():
                    alone = model(short, torch.tensor([5]))
                    together = model(batch, torch.tensor([5, 9]))
                self.assertEqual(sum(p.numel() for n, p in model.named_parameters() if "projector" in n), 0)
            finally:
                sys.path.remove(str(folder / "fake"))
                sys.path.remove(str(folder / "modeling"))
                for name in ("mamba_ssm", "models", "models.mamba_sleepnet"):
                    sys.modules.pop(name, None)
            torch.testing.assert_close(together[0, :5], alone[0], rtol=1e-5, atol=1e-5)

    def test_nights_read_like_watchsleepnet(self):
        from scripts import run_mambasleepnet_folds as run
        from test_make_replacement_sets import night

        with tempfile.TemporaryDirectory() as tmp:
            stages = np.array([0, 1, 2, 3, 4, 5, 2])
            np.savez_compressed(Path(tmp) / "S001.npz", **night(stages, 0.0, np.random.default_rng(0)))
            x, y, n = run.Nights(tmp, ["S001"])[0]
            self.assertEqual(tuple(x.shape), (7, 750))
            self.assertEqual(n, 7)
            self.assertEqual(y.tolist(), [0, 1, 1, 3, 2, -1, 1])             # W L L Deep REM unknown L


if __name__ == "__main__":
    unittest.main()
