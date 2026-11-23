#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --job-name=extract-ppg50
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/extract_ppg_%j.txt
# DREAMT 2.2.0 -> 50 Hz PPG epochs + ECG/PPG beats + quality + 25 Hz IBI (CPU work only;
# the GPU allocation is only the usual way to get CPUs from this login node).
# Run from /project/def-plago/yxtao/dataset-generation-30s-apnea:
#   sbatch submit_extract_ppg.sh [RAW_DIR] [OUT]
#   SUBJECTS="S002 S092" sbatch submit_extract_ppg.sh        # a quick test on two participants
# The raw CSVs are only read. Resubmitting keeps participants already written.
# Check afterwards: OUT/report.csv (alignment, beat accuracy vs ECG, good-epoch fraction).

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
  echo "neurokit2 is missing from .venv. On the login node run:" >&2
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
WORKERS="${WORKERS:-$(nproc)}"
[ "$WORKERS" -gt 16 ] && WORKERS=16
SUBJECT_ARGS=()
if [ -n "${SUBJECTS:-}" ]; then
  # shellcheck disable=SC2206
  SUBJECT_ARGS=(--subjects ${SUBJECTS})
fi

python -c "import neurokit2, scipy, numpy; print('neurokit2', neurokit2.__version__, 'scipy', scipy.__version__, 'numpy', numpy.__version__)"
srun python -B -u scripts/extract_ppg.py --raw-dir "$RAW_DIR" --output-dir "$OUT" --metadata "$METADATA" \
  --workers "$WORKERS" "${SUBJECT_ARGS[@]}"
echo "PPG50: $OUT  (report: $OUT/report.csv)"
