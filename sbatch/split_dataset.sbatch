#!/bin/bash
#SBATCH -A bsc114
#SBATCH -q acc_debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:10:00
#SBATCH --job-name=kvasir_split
#SBATCH --output=logs/split_%j.log

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs/splits

module load anaconda 2>/dev/null || true
if [ -d "$HOME/BSC_jobs" ]; then
  source "$HOME/BSC_jobs/bin/activate"
fi

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"

python scripts/split_dataset.py \
  --data-root data \
  --output outputs/splits/kvasir_split.json \
  --seed 999

