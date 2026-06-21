#!/bin/bash
#SBATCH -A bsc114
#SBATCH -q acc
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40
#SBATCH --gres=gpu:2
#SBATCH -t 01:00:00
#SBATCH --job-name=kvasir_train
#SBATCH --output=logs/train_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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
