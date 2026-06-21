#!/bin/bash
#SBATCH -A bsc114
#SBATCH -q acc_debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:10:00
#SBATCH --job-name=kvasir_split
#SBATCH --output=split_%j.log

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ "$(basename "${SUBMIT_DIR}")" = "sbatch" ]; then
  PROJECT_ROOT="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  PROJECT_ROOT="$(cd "${SUBMIT_DIR}" && pwd)"
fi

if [ ! -d "${PROJECT_ROOT}/scripts" ]; then
  echo "Could not locate project root from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
  echo "Submit from the project root or from the sbatch/ directory." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p logs outputs/splits
exec > >(tee -a "logs/split_${SLURM_JOB_ID:-manual}.log") 2>&1

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Submit dir: ${SUBMIT_DIR}"
echo "Project root: ${PROJECT_ROOT}"
echo "Working dir: $(pwd)"
echo "Started at: $(date)"

module load anaconda 2>/dev/null || true
if [ -d "$HOME/BSC_jobs" ]; then
  source "$HOME/BSC_jobs/bin/activate"
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
echo "Python: $(command -v python)"
python --version

python scripts/split_dataset.py \
  --data-root data \
  --output outputs/splits/kvasir_split.json \
  --seed 999

echo "Finished at: $(date)"
