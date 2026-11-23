#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=gen-ppg50
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/gen_ppg50_%j.txt
# Whole-night 50 Hz PPG for every participant and its quality vs the real PPG (GPU; trig-login01).
# Add IBI etc. later with EXTRA="--ibi" (a new LABEL/output folder).
# Run from /project/def-plago/yxtao/dataset-generation-ppg50:
#   sbatch submit_generate_ppg_nights.sh RUN_DIR [LABEL]
#   CHECKPOINT=checkpoints/epoch_050.pt sbatch submit_generate_ppg_nights.sh RUN_DIR e050
# Output: /home/yxtao/links/scratch/dataset-generation-ppg50/nights/<run name>_<LABEL> (or OUT=...).
# The checkpoint is copied into the output first; resubmitting the same command continues.

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

RUN="${1:-${RUN:-}}"
if [ -z "$RUN" ] || [ ! -d "$RUN" ]; then
  echo "Give the training run folder: sbatch submit_generate_ppg_nights.sh RUN_DIR [LABEL]" >&2
  exit 1
fi
LABEL="${2:-${LABEL:-best}}"
CHECKPOINT="${CHECKPOINT:-best.pt}"
OUT="${OUT:-/home/yxtao/links/scratch/dataset-generation-ppg50/nights/$(basename "$RUN")_${LABEL}}"
# shellcheck disable=SC2086
python -B -u scripts/generate_ppg_nights.py --run "$RUN" --checkpoint "$CHECKPOINT" --output-dir "$OUT" \
  --max-hours "${MAX_HOURS:-3.6}" --quality-workers "$(nproc)" ${EXTRA:-}
python -B scripts/plot_ppg_quality.py "$OUT" || true
echo "nights: $OUT  (summary: $OUT/subjects_summary.csv, figure: $OUT/quality_comparison.png)"
