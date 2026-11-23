#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=train-ppg50
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/train_ppg50_%j.txt
# Train the 50 Hz PPG diffusion model (GPU; submit from trig-login01).
# Run from /project/def-plago/yxtao/dataset-generation-ppg50:
#   sbatch submit_train_ppg.sh                 # new run: <output_root>/ppg50_<job id>
#   sbatch submit_train_ppg.sh RUN_DIR         # continue a paused or interrupted run
#   sbatch --export=ALL,EPOCHS=150 submit_train_ppg.sh RUN_DIR   # train a finished run further, to 150 epochs
# The job pauses cleanly before MAX_HOURS (default 3.5 h of the 4 h limit) and keeps every finished
# epoch; resubmit with the run folder to continue. A run folder is <output_root>/ppg50_<first job id>,
# so continuations can be queued right away (each starts when the previous one ends):
#   J=$(sbatch --parsable submit_train_ppg.sh)
#   R=/home/yxtao/links/scratch/dataset-generation-ppg50/runs/ppg50_$J
#   for i in 1 2 3; do J=$(sbatch --parsable --dependency=afterany:$J submit_train_ppg.sh $R); done
# Progress: the log (one line per epoch) and RUN_DIR/metrics.csv.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load python/3.11
module load scipy-stack/2024a
source ./.venv/bin/activate
# $HOME and $PROJECT are read-only inside Trillium jobs: keep caches on the job's local disk.
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/cache" MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
# single-thread BLIS (the multi-threaded one aborts next to PyTorch's OpenMP runtime)
export BLIS_THREAD_IMPL=single BLIS_NUM_THREADS=1
export BLIS_JC_NT=1 BLIS_PC_NT=1 BLIS_IC_NT=1 BLIS_JR_NT=1 BLIS_IR_NT=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

RUN="${1:-${RUN:-}}"
CONFIG="${CONFIG:-configs/ppg50.json}"
MAX_HOURS="${MAX_HOURS:-3.5}"
python -c "import torch, neurokit2, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'neurokit2', neurokit2.__version__, 'numpy', numpy.__version__)"
EPOCH_ARGS=()
if [ -n "${EPOCHS:-}" ]; then
  EPOCH_ARGS=(--epochs "$EPOCHS")
fi
if [ -n "$RUN" ]; then
  python -B -u scripts/train_ppg.py --resume "$RUN" --max-hours "$MAX_HOURS" "${EPOCH_ARGS[@]}"
else
  python -B -u scripts/train_ppg.py --config "$CONFIG" --max-hours "$MAX_HOURS" "${EPOCH_ARGS[@]}"
fi
