r"""One generator configuration per fold of the stage-replacement experiment (reads folds.json; new files only).

  python scripts/make_fold_configs.py [--folds FOLDS_JSON] [--base configs/ppg50.json] [--output-dir DIR]

FOLDS_JSON is the folds.json of ibi-sleep-staging's experiments/make_folds.py (default: the one on Trillium
scratch). The classifier of fold k is tested on that fold's test participants, so fold k's generator must
never see them. Written to --output-dir:

  fold<k>.json         --base (default configs/ppg50.json: N1/N2 stage condition, 300 epochs) with
                       exclude_subjects = the base's exclusions + fold k's test participants
  fold<k>_train.txt    fold k's classifier training participants, one line: the nights this fold's
                       generator generates (generate_ppg_nights.py --subjects $(cat fold<k>_train.txt))
  folds_source.json    which folds.json (path, sha256) and base configuration these came from

Existing files are never overwritten (a generator trained from them would no longer match). Train each fold:
  sbatch submit_train_ppg.sh RUNS/ppg50n1n2_fold<k> DIR/fold<k>.json    (repeat to continue; see that script)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ppg_training as pt

DEFAULT_FOLDS = Path("/home/yxtao/links/scratch/ibi-sleep-staging/folds.json")
DEFAULT_OUTPUT = Path("/home/yxtao/links/scratch/dataset-generation-ppg50/fold_configs")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS, help="ibi-sleep-staging folds.json")
    parser.add_argument("--base", type=Path, default=ROOT / "configs" / "ppg50.json", help="base configuration")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def fold_configs(base, folds):
    """{fold: (config, training participants, test participants)}; no test participant may be trained on."""
    out = {}
    for fold in folds:
        k = int(fold["fold"])
        test, train = sorted(fold["test"]), sorted(fold["train"])
        seen = set(fold.get("generator_train", train + fold.get("val", [])))
        overlap = sorted(set(test) & (seen | set(train)))
        if overlap:
            raise SystemExit(f"fold {k}: test participants also in training: {overlap}")
        config = copy.deepcopy(base)
        config["exclude_subjects"] = sorted(set(base.get("exclude_subjects", [])) | set(test))
        out[k] = (config, train, test)
    return out


def main(argv=None):
    args = parse_args(argv)
    folds_path = args.folds.expanduser().resolve()
    raw = folds_path.read_bytes()
    folds = json.loads(raw.decode("utf-8"))["folds"]
    base_given = json.loads(args.base.read_text(encoding="utf-8-sig"))
    base = pt.load_config(args.base)                          # validated, with defaults filled in
    configs = fold_configs(base_given, folds)
    output = args.output_dir.expanduser().resolve()
    names = [f"fold{k}{suffix}" for k in configs for suffix in (".json", "_train.txt")] + ["folds_source.json"]
    taken = [n for n in names if (output / n).exists()]
    if taken:
        raise SystemExit(f"never overwritten: {taken} in {output} (use a new --output-dir)")
    output.mkdir(parents=True, exist_ok=True)
    for k, (config, train, _) in configs.items():
        pt.load_config(None, config)                          # the fold configuration loads as is
        (output / f"fold{k}.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (output / f"fold{k}_train.txt").write_text(" ".join(train) + "\n", encoding="utf-8")
    (output / "folds_source.json").write_text(json.dumps(dict(
        folds=str(folds_path), folds_sha256=hashlib.sha256(raw).hexdigest(), base=str(args.base.resolve()),
        stage_condition=base["stage_condition"], epochs=base["training"]["epochs"],
        folds_written=sorted(configs)), indent=1) + "\n", encoding="utf-8")
    print(f"{len(configs)} fold configurations from {folds_path}")
    print(f"  base {args.base}: stage condition {base['stage_condition']}, {base['training']['epochs']} epochs")
    for k, (config, train, test) in configs.items():
        print(f"  fold {k}: generator never sees the {len(test)} test participants; "
              f"generates the {len(train)} training nights")
    print(f"output: {output}")
    return dict(output=str(output), folds=sorted(configs))


if __name__ == "__main__":
    main()
