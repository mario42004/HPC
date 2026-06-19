#!/bin/bash
#SBATCH -A bsc114
#SBATCH -q acc_interactive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40
#SBATCH --gres=gpu:2
#SBATCH -t 01:00:00
#SBATCH --job-name=unet_kvasir_jupyter
#SBATCH --output=jupyter_%j.log

set -euo pipefail

export PORT="${PORT:-8888}"
export IP="${IP:-0.0.0.0}"

echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "GPUs visible to this job: ${CUDA_VISIBLE_DEVICES:-not set}"
echo
echo "From your local machine, create an SSH tunnel to this node/port if needed."
echo "Then open the Jupyter URL printed below."
echo

module load anaconda 2>/dev/null || true

if [ -d "$HOME/BSC_jobs" ]; then
  # Optional environment used during the training preparation.
  # It is ignored when the directory does not exist.
  source "$HOME/BSC_jobs/bin/activate"
fi

jupyter lab --no-browser --ip="$IP" --port="$PORT"
