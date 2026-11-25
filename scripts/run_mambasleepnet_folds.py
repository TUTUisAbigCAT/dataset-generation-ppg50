r"""Train and test MambaSleepNet (CE only) on our fixed participant folds: DREAMT only, from scratch, per arm and seed.

  python scripts/run_mambasleepnet_folds.py --mamba-dir ~/WatchSleepNet_public/modeling --folds FOLDS/folds.json \
      --sets SETS --output-dir RESULTS [--arms real gen_Deep ...] [--seeds 0 1 2] [--epochs 200] [--patience 20]

The network is the CNN + bidirectional Mamba of MambaSleepNet: the DeepFeatureExtractor and BiMambaBlock
classes are imported from <mamba-dir>/models/mamba_sleepnet.py (which needs mamba_ssm and CUDA), and this
script puts only a linear classifier on top: no projection head, no contrastive loss. Per 30-s epoch:
CNN features of the real (unpadded) epochs, then the Mamba stack over each night's true length, then the
class. Parameter names (feature_extractor, mamba_layers, classifier) are MambaSleepNet's.

Training follows the settings of MambaSleepNetConfig and its engine, with cross-entropy only: Adam
(lr 1e-3, weight decay 1e-4), batch 32 nights, at most 200 epochs; a checkpoint is saved when validation
kappa does not fall and validation loss does not rise (the first epoch always), and training stops after
20 epochs without one. Nights are read as WatchSleepNet's SSDataset reads them: `data` cut into 30-s
epochs of fs*30 samples, `stages` per epoch mapped W 0, N1/N2 1 (Light), N3 3 (Deep), REM 2, anything else
-1 (ignored), at most 1100 epochs.

SETS is the output of scripts/make_replacement_sets.py (training participants with the arm's replaced
windows, validation and test participants real). For every arm and seed, each fold trains on its "train"
participants, stops on its "val" participants and is tested on its "test" participants. Scores: per fold
and pooled over the five test folds, accuracy, Cohen's kappa, macro F1, F1 per class, confusion matrix; per
test participant, accuracy and kappa. Output: RESULTS/<arm>/seed<s>/ (fold checkpoints, training logs,
results.json, per_fold.csv, per_participant.csv) and RESULTS/summary.csv / summary.json over every finished
run: mean and SD over seeds, and the difference to the "real" arm per seed. A finished run is not repeated.
--check loads one batch of every loader of the first arm and runs the model on it (no training).
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

CLASSES = ("W", "Light", "REM", "Deep")
LABEL_MAP = {0: 0, 1: 1, 2: 1, 3: 3, 4: 2}            # W, N1, N2, N3, REM -> W, Light, Light, Deep, REM
MAX_EPOCHS_PER_NIGHT = 1100
DEFAULTS = dict(hidden_dim=256, num_layers=4, d_state=64, batch_size=32, learning_rate=1e-3, weight_decay=1e-4,
                epochs=200, patience=20)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mamba-dir", type=Path, required=True,
                        help="the modeling folder that holds models/mamba_sleepnet.py")
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--sets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="*", default=None, help="default: every arm folder in --sets")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    for key, value in DEFAULTS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- data
class Nights:
    """Night NPZ files -> (epochs [n, fs*30], labels [n]) as WatchSleepNet's SSDataset."""

    def __init__(self, folder, sids):
        self.files = [Path(folder) / f"{s}.npz" for s in sids]
        missing = [str(p) for p in self.files if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing nights: {missing[:5]}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        import torch

        with np.load(self.files[index]) as z:
            ibi = z["data"].reshape(-1).astype(np.float32)
            stages = z["stages"].reshape(-1)
            fs = int(np.asarray(z["fs"]).item())
        samples = fs * 30
        count = min(len(ibi) // samples, MAX_EPOCHS_PER_NIGHT)
        x = ibi[:count * samples].reshape(count, samples)
        y = np.array([LABEL_MAP.get(int(v), -1) for v in stages[::samples][:count]], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(y), count

    @staticmethod
    def collate(batch):
        import torch
        from torch.nn.utils.rnn import pad_sequence

        xs, ys, lengths = zip(*batch)
        return (pad_sequence(xs, batch_first=True, padding_value=0.0),
                pad_sequence(ys, batch_first=True, padding_value=-1), torch.tensor(lengths))


def loader(folder, sids, batch_size, shuffle, workers):
    from torch.utils.data import DataLoader

    return DataLoader(Nights(folder, sids), batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      collate_fn=Nights.collate)


# --------------------------------------------------------------------------- model
def import_blocks(mamba_dir):
    mamba_dir = Path(mamba_dir).expanduser().resolve()
    if not (mamba_dir / "models" / "mamba_sleepnet.py").is_file():
        raise SystemExit(f"{mamba_dir} has no models/mamba_sleepnet.py (give the modeling folder)")
    if str(mamba_dir) not in sys.path:
        sys.path.insert(0, str(mamba_dir))
    module = importlib.import_module("models.mamba_sleepnet")
    if getattr(module, "Mamba", None) is None:
        raise SystemExit("mamba_ssm is not importable in this environment (MambaSleepNet needs it, with CUDA)")
    return module


def build_model(blocks, *, hidden_dim, num_layers, d_state, num_classes=len(CLASSES)):
    import torch
    from torch import nn
    import torch.nn.functional as F

    class MambaSleepNetCE(nn.Module):
        """CNN per 30-s epoch -> bidirectional Mamba over the night -> linear classifier (CE only)."""

        def __init__(self):
            super().__init__()
            self.feature_extractor = blocks.DeepFeatureExtractor(initial_channels=1, target_dim=hidden_dim)
            self.mamba_layers = nn.Sequential(*[blocks.BiMambaBlock(d_model=hidden_dim, d_state=d_state)
                                                for _ in range(num_layers)])
            self.classifier = nn.Linear(hidden_dim, num_classes)

        def forward(self, x, lengths):
            batch, steps, samples = x.shape
            lengths = torch.as_tensor(lengths, device=x.device).long()
            valid = torch.arange(steps, device=x.device)[None, :] < lengths[:, None]
            features = self.feature_extractor(x[valid].reshape(-1, 1, samples))     # padding never enters the CNN
            sequence = features.new_zeros(batch, steps, features.size(-1))
            sequence[valid] = features
            out = torch.zeros_like(sequence)
            for length in lengths.unique().tolist():                                 # each night at its own length
                rows = torch.where(lengths == length)[0]
                encoded = self.mamba_layers(sequence.index_select(0, rows)[:, :length].contiguous())
                out = out.index_copy(0, rows, F.pad(encoded, (0, 0, 0, steps - length)))
            return self.classifier(out)

    return MambaSleepNetCE()


# --------------------------------------------------------------------------- training and scores
def set_seed(seed):
    import torch

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def scores(true, pred, n_classes=len(CLASSES)):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

    true, pred = np.asarray(true, dtype=int), np.asarray(pred, dtype=int)
    keep = true >= 0
    true, pred = true[keep], pred[keep]
    labels = list(range(n_classes))
    if true.size == 0:
        return dict(epochs=0)
    return dict(epochs=int(true.size), accuracy=float(accuracy_score(true, pred)),
                kappa=float(cohen_kappa_score(true, pred, labels=labels)),
                macro_f1=float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
                f1=[float(v) for v in f1_score(true, pred, labels=labels, average=None, zero_division=0)],
                confusion=confusion_matrix(true, pred, labels=labels).tolist())


def predict(model, data, device, loss_fn=None):
    """(true labels, predicted labels, mean batch loss) over valid epochs."""
    import torch

    model.eval()
    true, pred, losses = [], [], []
    with torch.no_grad():
        for x, y, lengths in data:
            x, y = x.to(device), y.to(device)
            logits = model(x, lengths)
            mask = torch.arange(y.shape[1], device=device)[None, :] < lengths.to(device)[:, None]
            if loss_fn is not None:
                losses.append(float(loss_fn(logits[mask], y[mask])))
            true.extend(y[mask].cpu().tolist())
            pred.extend(logits[mask].argmax(-1).cpu().tolist())
    return true, pred, (float(np.mean(losses)) if losses else math.nan)


def train_fold(model, train_dl, val_dl, args, device, checkpoint, log_path):
    """CE training with MambaSleepNet's checkpoint rule: val kappa >= best and val loss <= best."""
    import torch

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    best = None
    waited = 0
    rows = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        total, batches = 0.0, 0
        for x, y, lengths in train_dl:
            x, y = x.to(device), y.to(device)
            mask = torch.arange(y.shape[1], device=device)[None, :] < lengths.to(device)[:, None]
            if not (y[mask] >= 0).any():
                continue
            optimizer.zero_grad()
            loss = loss_fn(model(x, lengths)[mask], y[mask])
            if torch.isnan(loss):
                print(f"epoch {epoch}: NaN loss, ending the epoch")
                break
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        true, pred, val_loss = predict(model, val_dl, device, loss_fn)
        val = scores(true, pred)
        kappa = val.get("kappa")
        kappa = -1.0 if kappa is None or not math.isfinite(kappa) else kappa
        row = dict(epoch=epoch, train_loss=total / max(batches, 1), val_loss=val_loss, val_kappa=kappa,
                   val_accuracy=val.get("accuracy"), seconds=round(time.time() - started, 1), saved=False)
        if best is None or (val_loss <= best[1] and kappa >= best[0]):
            best = (kappa, val_loss) if best is None else (max(best[0], kappa), min(best[1], val_loss))
            torch.save(model.state_dict(), checkpoint)
            row["saved"], waited = True, 0
        else:
            waited += 1
        rows.append(row)
        print(f"  epoch {epoch}: train loss {row['train_loss']:.4f}, val loss {val_loss:.4f}, "
              f"val kappa {kappa:.3f}{' (saved)' if row['saved'] else ''}", flush=True)
        if waited >= args.patience:
            print(f"  early stop after {epoch} epochs")
            break
    with open(log_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_one(arm, seed, folds, args, blocks):
    import torch

    output = args.output_dir / arm / f"seed{seed}"
    if (output / "results.json").is_file():
        print(f"{arm} seed {seed}: finished before, kept")
        return json.loads((output / "results.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    per_fold, per_participant, pooled_true, pooled_pred = [], [], [], []
    for fold in folds:
        set_seed(seed * 1000 + fold["fold"])
        folder = args.sets / arm / f"fold{fold['fold']}"
        print(f"{arm} seed {seed} fold {fold['fold']}: train {len(fold['train'])}, val {len(fold['val'])}, "
              f"test {len(fold['test'])}", flush=True)
        model = build_model(blocks, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                            d_state=args.d_state).to(device)
        checkpoint = output / f"fold{fold['fold']}.pt"
        log = train_fold(model, loader(folder, fold["train"], args.batch_size, True, args.num_workers),
                         loader(folder, fold["val"], args.batch_size, False, args.num_workers), args, device,
                         checkpoint, output / f"fold{fold['fold']}_training.csv")
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        fold_true, fold_pred = [], []
        for sid in fold["test"]:
            true, pred, _ = predict(model, loader(folder, [sid], 1, False, 0), device)
            s = scores(true, pred)
            per_participant.append(dict(fold=fold["fold"], subject=sid, epochs=s["epochs"],
                                        accuracy=s.get("accuracy"), kappa=s.get("kappa")))
            fold_true.extend(true)
            fold_pred.extend(pred)
        saved = [r["epoch"] for r in log if r["saved"]]
        per_fold.append(dict(fold=fold["fold"], trained_epochs=len(log), best_epoch=saved[-1] if saved else None,
                             **scores(fold_true, fold_pred)))
        pooled_true.extend(fold_true)
        pooled_pred.extend(fold_pred)
    settings = {k: getattr(args, k) for k in DEFAULTS}
    result = dict(arm=arm, seed=seed, model="MambaSleepNet CNN + BiMamba, cross-entropy only", classes=list(CLASSES),
                  pooled=scores(pooled_true, pooled_pred), per_fold=per_fold, settings=settings)
    with (output / "per_fold.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["fold", "epochs", "best_epoch", "accuracy", "kappa", "macro_f1"] + [f"f1_{n}" for n in CLASSES])
        for row in per_fold:
            writer.writerow([row["fold"], row["epochs"], row["best_epoch"], row.get("accuracy"), row.get("kappa"),
                             row.get("macro_f1")] + list(row.get("f1", [None] * len(CLASSES))))
    with (output / "per_participant.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["fold", "subject", "epochs", "accuracy", "kappa"])
        writer.writeheader()
        writer.writerows(per_participant)
    (output / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    p = result["pooled"]
    print(f"{arm} seed {seed}: accuracy {p['accuracy']:.3f}, kappa {p['kappa']:.3f}, macro F1 {p['macro_f1']:.3f}, "
          + ", ".join(f"F1 {n} {v:.3f}" for n, v in zip(CLASSES, p["f1"])), flush=True)
    return result


def summarize(output_dir):
    runs = {}
    for path in sorted(output_dir.glob("*/seed*/results.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        runs.setdefault(r["arm"], {})[r["seed"]] = r
    metrics = ["accuracy", "kappa", "macro_f1"] + [f"f1_{n}" for n in CLASSES]

    def value(r, m):
        return r["pooled"]["f1"][CLASSES.index(m[3:])] if m.startswith("f1_") else r["pooled"][m]

    rows, summary = [], {}
    for arm, seeds in runs.items():
        entry = dict(seeds=sorted(seeds))
        for m in metrics:
            values = np.array([value(seeds[s], m) for s in sorted(seeds)])
            entry[m] = dict(mean=float(values.mean()), sd=float(values.std(ddof=1)) if values.size > 1 else 0.0)
            common = sorted(set(seeds) & set(runs.get("real", {}))) if arm != "real" else []
            if common:
                diff = np.array([value(seeds[s], m) - value(runs["real"][s], m) for s in common])
                entry[m].update(diff_vs_real_mean=float(diff.mean()),
                                diff_vs_real_sd=float(diff.std(ddof=1)) if diff.size > 1 else 0.0)
        summary[arm] = entry
        rows.append([arm, len(seeds)] + [f"{entry[m]['mean']:.4f} ± {entry[m]['sd']:.4f}" for m in metrics]
                    + [f"{entry[m].get('diff_vs_real_mean', float('nan')):+.4f}" for m in metrics])
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arm", "seeds"] + metrics + [f"{m}_minus_real" for m in metrics])
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(dict(classes=list(CLASSES), arms=summary), indent=1),
                                             encoding="utf-8")
    return summary


def main(argv=None):
    args = parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.sets = args.sets.expanduser().resolve()
    folds = json.loads(args.folds.read_text(encoding="utf-8"))["folds"]
    arms = args.arms or sorted(p.name for p in args.sets.iterdir() if p.is_dir() and (p / "fold0").is_dir())
    missing = [f"{a}/fold{f['fold']}" for a in arms for f in folds if not (args.sets / a / f"fold{f['fold']}").is_dir()]
    if missing:
        raise SystemExit(f"missing in {args.sets}: {missing}")
    blocks = import_blocks(args.mamba_dir)
    if args.check:
        import torch

        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = build_model(blocks, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                            d_state=args.d_state).to(device).eval()
        for f in folds:
            folder = args.sets / arms[0] / f"fold{f['fold']}"
            for part in ("train", "val", "test"):
                x, y, lengths = next(iter(loader(folder, f[part], args.batch_size, False, 0)))
                with torch.no_grad():
                    out = model(x.to(device), lengths)
                print(f"{arms[0]} fold {f['fold']} {part}: {len(f[part])} nights, batch {tuple(x.shape)} -> "
                      f"{tuple(out.shape)}, labels {sorted(set(y[y >= 0].tolist()))}")
        parameters = sum(p.numel() for p in model.parameters())
        print(f"classes {list(CLASSES)}; model parameters {parameters}; arms {arms}")
        return dict(status="checked", arms=arms, parameters=parameters)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        for seed in args.seeds:
            run_one(arm, seed, folds, args, blocks)
    summary = summarize(args.output_dir)
    print(f"summary: {args.output_dir / 'summary.csv'}")
    return summary


if __name__ == "__main__":
    main()
