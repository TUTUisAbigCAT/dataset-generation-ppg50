#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=00:30:00
#SBATCH --job-name=compare-peaks
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/compare_peaks_%j.txt
# Beat-detection routes against the PSG ECG on the existing extraction (its files are only read):
# the current route, plain Elgendi, the current route + correct_artifacts, and the teacher's route
# (nk.ppg_clean at 50 Hz, cleaned wave upsampled to 500 Hz, nk.ppg_peaks with correct_artifacts=True).
# CPU only: submit from a Trillium CPU login node (ssh trillium.scinet.utoronto.ca), not trig-login01.
# Run from /project/def-plago/yxtao/dataset-generation-ppg50:
#   sbatch submit_compare_peaks.sh [DATA_DIR] [OUT]
# Results: OUT/summary.json, OUT/compare.csv and the table at the end of the log.

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
  echo "neurokit2 is missing from .venv. On a login node run (never without -c: neurokit2 0.3 pulls numpy 2):" >&2
  echo "  source .venv/bin/activate && pip install 'neurokit2==0.2.12' -c constraints_trillium.txt" >&2
  exit 1
fi
if ! python -c "import numpy, sys; sys.exit(0 if numpy.__version__.startswith('1.') else 1)"; then
  echo "numpy $(python -c 'import numpy; print(numpy.__version__)') in .venv: torch 2.3.1 needs numpy 1.26." >&2
  echo "  Remove it on a login node: pip uninstall -y numpy pandas   (scipy-stack's numpy 1.26.4 is used again)" >&2
  exit 1
fi

DATA_DIR="${1:-/home/yxtao/links/scratch/WatchSleepProcessedDataset/DREAMT22_PPG50}"
OUT="${2:-/home/yxtao/links/scratch/dataset-generation-ppg50/peak_methods}"
if [ ! -d "$DATA_DIR" ]; then
  echo "Extraction folder not found: $DATA_DIR" >&2
  exit 1
fi
python -c "import neurokit2, scipy, numpy; print('neurokit2', neurokit2.__version__, 'scipy', scipy.__version__, 'numpy', numpy.__version__)"
# plain python (not srun): one process that starts its own workers on this node's cores
python -B -u scripts/compare_peak_methods.py --data-dir "$DATA_DIR" --output-dir "$OUT" --workers "$(nproc)"
