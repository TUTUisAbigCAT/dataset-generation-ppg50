#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=train-fold
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/train_fold_%j.txt
# Train one fold's generator (GPU; submit from trig-login01, in /project/def-plago/yxtao/dataset-generation-ppg50):
#   sbatch submit_train_fold.sh FOLD_CONFIG RUN_DIR
# A new RUN_DIR starts from FOLD_CONFIG (scripts/make_folds.py writes fold<k>.json); a RUN_DIR that already
# holds a run is continued, and a completed one ends at once. So queue a few jobs per fold right away:
#   D=/home/yxtao/links/scratch/dataset-generation-ppg50/folds5_n1n2
#   for f in 0 1 2 3 4; do
#     R=$D/runs/ppg50n1n2_fold$f
#     J=$(sbatch --parsable submit_train_fold.sh $D/fold$f.json $R)
#     for i in 1 2; do J=$(sbatch --parsable --dependency=afterany:$J submit_train_fold.sh $D/fold$f.json $R); done
#   done
# Arguments, not environment variables: Trillium starts jobs with --export=NONE.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load python/3.11
module load scipy-stack/2024a
source ./.venv/bin/activate
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/cache" MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
export BLIS_THREAD_IMPL=single BLIS_NUM_THREADS=1
export BLIS_JC_NT=1 BLIS_PC_NT=1 BLIS_IC_NT=1 BLIS_JR_NT=1 BLIS_IR_NT=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

CONFIG="${1:-}"
RUN="${2:-}"
MAX_HOURS="${3:-3.5}"
if [ -z "$CONFIG" ] || [ -z "$RUN" ] || [ ! -f "$CONFIG" ]; then
  echo "Usage: sbatch submit_train_fold.sh FOLD_CONFIG RUN_DIR [MAX_HOURS]" >&2
  exit 1
fi
if [ -f "$RUN/status.json" ] && grep -q '"completed"' "$RUN/status.json"; then
  echo "$RUN is completed: nothing to do"
  exit 0
fi
if [ -f "$RUN/config.json" ]; then
  python -B -u scripts/train_ppg.py --resume "$RUN" --max-hours "$MAX_HOURS"
else
  python -B -u scripts/train_ppg.py --config "$CONFIG" --output-dir "$RUN" --max-hours "$MAX_HOURS"
fi
