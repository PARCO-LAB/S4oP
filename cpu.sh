#!/bin/bash
#SBATCH --job-name=retrieval
#SBATCH --partition=gpuRTX
#SBATCH --nodelist=node008
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=cpu/%x_%j.out
#SBATCH --error=cpu/%x_%j.err

set -euo pipefail

MODEL="mamba"
DATASET="retrieval"
CKPT_FOLDER="checkpoints_pruned1"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Host: $(hostname)   Job: ${SLURM_JOB_ID}   $(date)"
python3 -c "import torch; print('cuda available:', torch.cuda.is_available())"

for P in 50 70 90; do
    echo ">>> pruning ${P}%"
    srun python3 latency.py -m "${MODEL}" -d "${DATASET}" -f "${CKPT_FOLDER}" -p "mamba_${DATASET}_pruned_${P}%"
done

echo ">>> FINITO (exit code $?)"