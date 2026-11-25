r"""Participant-level folds for the stage-replacement experiment, and one generator config per fold.

  python scripts/make_folds.py [--config configs/ppg50.json] [--output-dir DIR] [--folds 5] [--seed 20261124]

Participants are the ones a training cohort would load (an extraction NPZ in data_dir and complete
AHI/BMI/AGE/GENDER in participant_info.csv). They are split into K folds balanced on deep sleep (at least
`--deep-epochs` N3 epochs, default 10 = 5 min) and on AHI: within each of the two groups, participants
sorted by AHI are dealt in blocks of K, each block in random order to the folds with the fewest members.
In every fold, `--val-fraction` of all participants (default 0.1) are taken from the training side, the
same way, as WatchSleepNet's validation participants.

Output in --output-dir (default <output_root>/../folds<K>_n1n2, never overwritten):
  folds.json        per fold: test (all real), val and train (WatchSleepNet), generator_train = train + val;
                    per participant: AHI, deep epochs, fold
  fold<k>.json      the training config of fold k's generator: --config with stage_condition "n1n2",
                    exclude_subjects = fold k's test participants, training.epochs = --epochs (default 150)
A fold's generator never sees its test participants; its night generation covers exactly generator_train.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import ppg_training as pt

DEEP_CODE = 3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ppg50.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--deep-epochs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=150, help="generator training epochs")
    parser.add_argument("--stage-condition", default="n1n2", choices=sorted(pt.STAGE_LAYOUTS))
    parser.add_argument("--seed", type=int, default=20261124)
    return parser.parse_args(argv)


def participants(config):
    """{sid: dict(ahi, deep_epochs, epochs)} of the participants a training cohort would load."""
    info = pt.read_participants(config["metadata_path"])
    excluded = set(config["exclude_subjects"])
    out = {}
    for path in sorted(Path(config["data_dir"]).glob("S*.npz")):
        sid = path.stem
        if path.name.startswith(".") or sid in excluded or sid not in info.index:
            continue
        with np.load(path) as z:
            schema = str(z["schema"]) if "schema" in z.files else ""
            if not schema.startswith("dreamt_ppg50_extraction_"):
                continue
            stages = z["stages"].astype(np.int64)
        row = info.loc[sid]
        values = [float(row[c]) for c in pt.CONTINUOUS]
        if str(row["GENDER"]).strip().upper() not in ("F", "M") or not np.isfinite(values).all():
            continue
        out[sid] = dict(ahi=values[0], deep_epochs=int((stages == DEEP_CODE).sum()), epochs=int(stages.size))
    return out


def deal(sids, people, k, rng, sizes, deep_epochs):
    """Assign sids to k bins: deep-sleep groups, AHI order, blocks of k dealt to the smallest bins."""
    assignment = {}
    for group in (True, False):
        members = sorted((s for s in sids if (people[s]["deep_epochs"] >= deep_epochs) == group),
                         key=lambda s: (people[s]["ahi"], s))
        for start in range(0, len(members), k):
            block = list(rng.permutation(members[start:start + k]))
            order = sorted(range(k), key=lambda b: (sizes[b], rng.random()))
            for sid, b in zip(block, order):
                assignment[sid] = b
                sizes[b] += 1
    return assignment


def make_folds(people, k, val_fraction, deep_epochs, seed):
    rng = np.random.default_rng(seed)
    sids = sorted(people)
    if len(sids) < 2 * k:
        raise ValueError(f"{len(sids)} participants are too few for {k} folds")
    test_of = deal(sids, people, k, rng, [0] * k, deep_epochs)
    n_val = max(1, int(round(val_fraction * len(sids))))
    folds = []
    for fold in range(k):
        test = sorted(s for s in sids if test_of[s] == fold)
        rest = [s for s in sids if test_of[s] != fold]
        bins = max(2, int(round(len(rest) / n_val)))
        val_of = deal(rest, people, bins, rng, [0] * bins, deep_epochs)
        val = sorted(s for s in rest if val_of[s] == 0)
        train = sorted(s for s in rest if val_of[s] != 0)
        folds.append(dict(fold=fold, test=test, val=val, train=train, generator_train=sorted(train + val)))
    return folds, test_of


def main(argv=None):
    args = parse_args(argv)
    config = pt.load_config(args.config)
    output = (args.output_dir or Path(config["output_root"]).expanduser().parent
              / f"folds{args.folds}_{args.stage_condition}").expanduser().resolve()
    if (output / "folds.json").exists():
        raise SystemExit(f"{output / 'folds.json'} exists; folds are never overwritten (give another --output-dir)")
    people = participants(config)
    folds, test_of = make_folds(people, args.folds, args.val_fraction, args.deep_epochs, args.seed)
    output.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        fold_config = json.loads(json.dumps(config))
        fold_config.update(stage_condition=args.stage_condition, exclude_subjects=sorted(fold["test"]))
        fold_config["training"]["epochs"] = args.epochs
        pt.load_config(None, fold_config)                                     # still a valid config
        (output / f"fold{fold['fold']}.json").write_text(json.dumps(fold_config, indent=2), encoding="utf-8")
    summary = dict(schema="ppg50_folds_v1", folds_count=args.folds, seed=args.seed, val_fraction=args.val_fraction,
                   deep_epochs_threshold=args.deep_epochs, stage_condition=args.stage_condition,
                   base_config=str(args.config.resolve()), data_dir=config["data_dir"], folds=folds,
                   participants={s: dict(people[s], fold=test_of[s]) for s in sorted(people)})
    (output / "folds.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{len(people)} participants -> {args.folds} folds in {output}")
    for fold in folds:
        deep = sum(people[s]["deep_epochs"] >= args.deep_epochs for s in fold["test"])
        ahi = np.median([people[s]["ahi"] for s in fold["test"]])
        print(f"  fold {fold['fold']}: test {len(fold['test'])} (deep sleep {deep}, median AHI {ahi:.1f}), "
              f"val {len(fold['val'])}, train {len(fold['train'])}")
    return summary


if __name__ == "__main__":
    main()
