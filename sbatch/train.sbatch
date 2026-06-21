#!/bin/bash
#SBATCH -A bsc114
#SBATCH -q acc
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40
#SBATCH --gres=gpu:2
#SBATCH -t 01:00:00
#SBATCH --job-name=kvasir_train
#SBATCH --output=train_%j.log

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
mkdir -p logs outputs/checkpoints

module load anaconda 2>/dev/null || true
if [ -d "$HOME/BSC_jobs" ]; then
  source "$HOME/BSC_jobs/bin/activate"
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

python scripts/train.py \
  --data-root data \
  --split-file outputs/splits/kvasir_split.json \
  --checkpoint outputs/checkpoints/unet_kvasir.pt \
  --history outputs/checkpoints/history.json \
  --image-size 256 \
  --batch-size 8 \
  --epochs 30 \
  --learning-rate 0.001 \
  --num-workers 4 \
  --gpus 2
