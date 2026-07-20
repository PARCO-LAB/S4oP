#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuRTX     
#SBATCH --cpus-per-task=16 
#SBATCH --mem=128G
#SBATCH --time=96:00:00      
#SBATCH --output=test/%x_%j.out   
#SBATCH --error=test/%x_%j.err    

set -euo pipefail

MODEL="mamba"
DATASET="image"
CKPT_FOLDER="checkpoints1"
CKPT_PRUNED="checkpoints_unstructured1"

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

PRUNED_MODELS=(
    "mamba_${DATASET}_pruned_10%"
    "mamba_${DATASET}_pruned_30%"
    "mamba_${DATASET}_pruned_50%"
    "mamba_${DATASET}_pruned_70%"
    "mamba_${DATASET}_pruned_90%"
)

for PRUNED_MODEL in "${PRUNED_MODELS[@]}"; do
    echo ">>> python3 test.py --dataset ${DATASET} --checkpoint ./${CKPT_PRUNED}/${PRUNED_MODEL}.pth"
    srun python3 test.py --dataset "${DATASET}" --checkpoint "./${CKPT_PRUNED}/${PRUNED_MODEL}.pth"
done

echo ">>> FINITO (exit code $?)"