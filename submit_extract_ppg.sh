#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=00:30:00
#SBATCH --job-name=extract-ppg50
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/extract_ppg_%j.txt
# DREAMT 2.2.0 -> 50 Hz PPG epochs + 25 Hz IBI variants + accuracy against the PSG ECG.
# CPU only: submit from a Trillium CPU login node (ssh trillium.scinet.utoronto.ca), not from
# trig-login01. CPU jobs always get a whole node (192 cores); every participant runs in parallel.
# Run from /project/def-plago/yxtao/dataset-generation-ppg50:
#   sbatch submit_extract_ppg.sh [RAW_DIR] [OUT]
#   SUBJECTS="S002 S092" sbatch submit_extract_ppg.sh        # a quick test on two participants
# The raw CSVs are only read. Resubmitting keeps participants already written.
# Check afterwards: OUT/report.csv (alignment, IBI error of each variant vs ECG, epoch_ok fraction).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load python/3.11
module load scipy-stack/2024a
source ./.venv/bin/activate
# $HOME and $PROJECT are read-only inside Trillium jobs: keep caches on the job's local disk.
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/cache" MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
# one thread per worker process
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if ! python -c "import neurokit2" 2>/dev/null; then
  echo "neurokit2 is missing from .venv. On a login node run:" >&2
  echo "  source .venv/bin/activate && (pip install --no-index neurokit2 || pip install neurokit2)" >&2
  exit 1
fi

RAW_DIR="${1:-${RAW_DIR:-/home/yxtao/links/scratch/dreamt/data_100Hz}}"
OUT="${2:-${OUT:-/home/yxtao/links/scratch/WatchSleepProcessedDataset/DREAMT22_PPG50}}"
METADATA="${METADATA:-/home/yxtao/links/scratch/dreamt/physionet.org/files/dreamt/2.2.0/participant_info.csv}"
if [ ! -d "$RAW_DIR" ]; then
  echo "Raw DREAMT folder not found: $RAW_DIR" >&2
  exit 1
fi
# each participant needs about 1-2 GB; a CPU node has 192 cores and 755 GiB
WORKERS="${WORKERS:-$(nproc)}"
SUBJECT_ARGS=()
if [ -n "${SUBJECTS:-}" ]; then
  # shellcheck disable=SC2206
  SUBJECT_ARGS=(--subjects ${SUBJECTS})
fi

python -c "import neurokit2, scipy, numpy; print('neurokit2', neurokit2.__version__, 'scipy', scipy.__version__, 'numpy', numpy.__version__)"
echo "workers: $WORKERS"
# plain python (not srun): one process that starts its own workers on this node's cores
python -B -u scripts/extract_ppg.py --raw-dir "$RAW_DIR" --output-dir "$OUT" --metadata "$METADATA" \
  --workers "$WORKERS" "${SUBJECT_ARGS[@]}"
echo "PPG50: $OUT  (report: $OUT/report.csv)"
