#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=wsn-folds
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/wsn_folds_%j.txt
# WatchSleepNet on our 5 folds, DREAMT only, from scratch: one arm and one seed per job (GPU; trig-login01).
#   sbatch submit_watchsleepnet_folds.sh WSN_MODELING_DIR FOLDS_JSON SETS RESULTS ARM SEED
# e.g. every arm with seeds 0-2:
#   W=/home/yxtao/WatchSleepNet_public/modeling      # the copy with the 4-class label map
#   D=/home/yxtao/links/scratch/dataset-generation-ppg50/folds5_n1n2
#   for A in real gen_Deep gen_REM gen_Deep-REM gen_random20 shuffled_Deep shuffled_REM shuffled_random20; do
#     for S in 0 1 2; do sbatch submit_watchsleepnet_folds.sh $W $D/folds.json $D/sets $D/wsn_results $A $S; done
#   done
# A finished arm/seed is skipped; summary.csv in RESULTS is rewritten by every job over what is finished.
# Arguments, not environment variables: Trillium starts jobs with --export=NONE.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load python/3.11
module load scipy-stack/2024a
source ./.venv/bin/activate
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/cache" MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
export BLIS_THREAD_IMPL=single BLIS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

if [ "$#" -lt 6 ]; then
  echo "Usage: sbatch submit_watchsleepnet_folds.sh WSN_MODELING_DIR FOLDS_JSON SETS RESULTS ARM SEED [more options]" >&2
  exit 1
fi
WSN="$1"; FOLDS="$2"; SETS="$3"; RESULTS="$4"; ARM="$5"; SEED="$6"; shift 6
python -c "import torch, sklearn, tqdm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || {
  echo "WatchSleepNet needs torch, scikit-learn and tqdm in .venv (pip install tqdm)" >&2; exit 1; }
python -B -u scripts/run_watchsleepnet_folds.py --wsn-dir "$WSN" --folds "$FOLDS" --sets "$SETS" \
  --output-dir "$RESULTS" --arms "$ARM" --seeds "$SEED" --num-workers 4 "$@"
