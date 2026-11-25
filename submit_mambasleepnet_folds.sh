#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=mamba-folds
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/mamba_folds_%j.txt
# MambaSleepNet (CE only) on our 5 folds, DREAMT only, from scratch: one arm and one seed per job (GPU).
#   sbatch submit_mambasleepnet_folds.sh MAMBA_MODELING_DIR FOLDS_JSON SETS RESULTS ARM SEED [more options]
# MAMBA_MODELING_DIR is the modeling folder of the WatchSleepNet checkout with models/mamba_sleepnet.py; the job
# activates that checkout's venv (../venv, which has mamba_ssm), as its own mamba.sh does; without one, this
# repository's .venv (which then needs mamba_ssm). e.g. every arm with seeds 0-2:
#   M=/project/def-plago/yxtao/WatchSleepNet_public/modeling
#   D=/home/yxtao/links/scratch/dataset-generation-ppg50/folds5_n1n2
#   for A in real gen_Deep gen_REM gen_Deep-REM gen_random20 shuffled_Deep shuffled_REM shuffled_random20; do
#     for S in 0 1 2; do sbatch submit_mambasleepnet_folds.sh $M $D/folds.json $D/sets $D/mamba_results $A $S; done
#   done
# A finished arm/seed is skipped; summary.csv in RESULTS is rewritten by every job over what is finished.
# Arguments, not environment variables: Trillium starts jobs with --export=NONE.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
if [ "$#" -lt 6 ]; then
  echo "Usage: sbatch submit_mambasleepnet_folds.sh MAMBA_MODELING_DIR FOLDS_JSON SETS RESULTS ARM SEED [more options]" >&2
  exit 1
fi
MAMBA="$1"; FOLDS="$2"; SETS="$3"; RESULTS="$4"; ARM="$5"; SEED="$6"; shift 6
VENV="$(cd "$MAMBA/.." && pwd)/venv"
if [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
else
  module load StdEnv/2023 gcc/12.3 python/3.11 scipy-stack/2024a
  source ./.venv/bin/activate
fi
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/cache" MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -c "import torch, sklearn, mamba_ssm; print('python', __import__('sys').executable, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())" || {
  echo "the environment needs torch, scikit-learn and mamba_ssm (with CUDA)" >&2; exit 1; }
python -B -u scripts/run_mambasleepnet_folds.py --mamba-dir "$MAMBA" --folds "$FOLDS" --sets "$SETS" \
  --output-dir "$RESULTS" --arms "$ARM" --seeds "$SEED" --num-workers 4 "$@"
