#!/bin/bash
#SBATCH -A bsc114
#SBATCH -q acc_interactive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH --job-name=kvasir_inference_jupyter
#SBATCH --output=jupyter_%j.log

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ "$(basename "${SUBMIT_DIR}")" = "sbatch" ]; then
  PROJECT_ROOT="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  PROJECT_ROOT="$(cd "${SUBMIT_DIR}" && pwd)"
fi

if [ ! -d "${PROJECT_ROOT}/notebooks" ]; then
  echo "Could not locate project root from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
  echo "Submit from the project root or from the sbatch/ directory." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p logs outputs/predictions
exec > >(tee -a "logs/jupyter_${SLURM_JOB_ID:-manual}.log") 2>&1

PORT="${PORT:-8888}"
IP="${IP:-0.0.0.0}"
NODE="$(hostname)"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Submit dir: ${SUBMIT_DIR}"
echo "Project root: ${PROJECT_ROOT}"
echo "Working dir: $(pwd)"
echo "Compute node: ${NODE}"
echo "Port: ${PORT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo
echo "From your local machine, create the SSH tunnel with:"
echo "ssh -L ${PORT}:${NODE}:${PORT} <your-bsc-user>@alogin4.bsc.es"
echo
echo "Then open the Jupyter URL printed below in your local browser."
echo

module load anaconda 2>/dev/null || true
if [ -d "$HOME/BSC_jobs" ]; then
  source "$HOME/BSC_jobs/bin/activate"
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
echo "Python: $(command -v python)"
python --version

jupyter lab --no-browser --ip="${IP}" --port="${PORT}" --notebook-dir="${PROJECT_ROOT}"
