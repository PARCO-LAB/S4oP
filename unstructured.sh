#!/bin/bash
#SBATCH --job-name=retrieval
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuRTX     
#SBATCH --cpus-per-task=16 
#SBATCH --mem=128G
#SBATCH --time=120:00:00      
#SBATCH --output=unstructured/%x_%j.out   
#SBATCH --error=unstructured/%x_%j.err    

set -euo pipefail

MODEL="mamba"
DATASET="retrieval"
CKPT_FOLDER="checkpoints3"
CKPT_PRUNED="checkpoints_unstructured3"
mkdir -p unstructured
module load cuda/12.8 

echo "==================== INFO NODO ===================="
echo "Host:    $(hostname)"
echo "Data:    $(date)"
echo "Job ID:  ${SLURM_JOB_ID:-N/A}"
echo "==================== GPU ==========================="
nvidia-smi || echo "ATTENZIONE: nvidia-smi non disponibile"
echo "==================== TORCH / CUDA =================="
python3 - << 'PYEOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("cuda (torch build):", torch.version.cuda)
PYEOF
echo "==================================================="

echo ">>> python3 unstructured_pruning.py -d ${DATASET} -c ${CKPT_FOLDER} -o ${CKPT_PRUNED} --importance_batches 5 --prune_A_log --test"
srun python3 unstructured_pruning.py -d "${DATASET}" -c "${CKPT_FOLDER}" -o "${CKPT_PRUNED}" --importance_batches 5 --prune_A_log --test

echo ">>> FINITO (exit code $?)"

