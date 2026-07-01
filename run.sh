#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodelist=node005   
#SBATCH --partition=gpuRTX     
#SBATCH --gres=gpu:1  
#SBATCH --cpus-per-task=16 
#SBATCH --mem=128G
#SBATCH --time=96:00:00      
#SBATCH --output=test/%x_%j.out   
#SBATCH --error=test/%x_%j.err    

set -euo pipefail

MODEL="mamba"
DATASET="image"
CKPT_FOLDER="checkpoints"
CKPT_PRUNED="checkpoints_pruned"
mkdir -p test
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

#echo ">>> python3 exec_pruning.py -m ${MODEL} -d ${DATASET} -b ${CKPT_FOLDER} -c ${CKPT_PRUNED}"
#srun python3 exec_pruning.py -m "${MODEL}" -d "${DATASET}" -b "${CKPT_FOLDER}" -c "${CKPT_PRUNED}"
echo ">>> python3 starting_model.py -m ${MODEL} -d ${DATASET} -f ${CKPT_FOLDER}"
srun python3 starting_model.py -m "${MODEL}" -d "${DATASET}" -f "${CKPT_FOLDER}"

echo ">>> FINITO (exit code $?)"