#!/bin/bash
#SBATCH --job-name=image
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuRTX     
#SBATCH --cpus-per-task=16 
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=mamba2/%x_%j.out   
#SBATCH --error=mamba2/%x_%j.err    

set -euo pipefail

MODEL="mamba2"
DATASET="image"
CKPT_FOLDER="mamba2"
CKPT_PRUNED="mamba2"
mkdir -p mamba2
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

# PRUNING
echo ">>> python3 s4op.py -m ${MODEL} -d ${DATASET} -b ${CKPT_FOLDER} -c ${CKPT_PRUNED}"
srun python3 s4op.py -m "${MODEL}" -d "${DATASET}" -b "${CKPT_FOLDER}" -c "${CKPT_PRUNED}"

# TRAIN/TEST BASE MODEL
# echo ">>> python3 starting_model.py -m ${MODEL} -d ${DATASET} -f ${CKPT_FOLDER}"
# srun python3 starting_model.py -m "${MODEL}" -d "${DATASET}" -f "${CKPT_FOLDER}"

# TEST PRUNED MODEL
# PRUNED_MODEL="mamba_ecg_pruned_10%"
# echo ">>> python3 starting_model.py -m ${MODEL} -d ${DATASET} -f ${CKPT_PRUNED} -p ${PRUNED_MODEL}"
# srun python3 starting_model.py -m "${MODEL}" -d "${DATASET}" -f "${CKPT_PRUNED}" -p "${PRUNED_MODEL}"

# TEST LATENCY/MEMORY FOOTPRINT BASE MODEL
# echo ">>> python3 latency.py -m ${MODEL} -d ${DATASET} -f ${CKPT_FOLDER}"
# srun python3 latency.py -m "${MODEL}" -d "${DATASET}" -f "${CKPT_FOLDER}"

# TEST LATENCY/MEMORY FOOTPRINT PRUNED MODEL
# PRUNED_MODELS=(
#     "mamba_${DATASET}_pruned_10%"
#     "mamba_${DATASET}_pruned_30%"
#     "mamba_${DATASET}_pruned_50%"
#     "mamba_${DATASET}_pruned_70%"
#     "mamba_${DATASET}_pruned_90%"
# )

# for PRUNED_MODEL in "${PRUNED_MODELS[@]}"; do
#     echo ">>> python3 latency.py -m ${MODEL} -d ${DATASET} -f ${CKPT_PRUNED} -p ${PRUNED_MODEL}"
#     srun python3 latency.py -m "${MODEL}" -d "${DATASET}" -f "${CKPT_PRUNED}" -p "${PRUNED_MODEL}"
# done

# DATASETS=(
#     "imdb"
#     "image"
#     "ecg"
#     "listops"
#     "retrieval"
# )

# for DAT in "${DATASETS[@]}"; do
#     echo ">>> python3 latency.py -m ${MODEL} -d ${DAT} -f ${CKPT_FOLDER}"
#     srun python3 latency.py -m "${MODEL}" -d "${DAT}" -f "${CKPT_FOLDER}"
# done

# S4oP RANKING
# echo ">>> python3 s4op_ranking.py -m ${MODEL} -d ${DATASET} -b ${CKPT_FOLDER} -c ${CKPT_PRUNED}"
# srun python3 s4op_ranking.py -m "${MODEL}" -d "${DATASET}" -b "${CKPT_FOLDER}" -c "${CKPT_PRUNED}"

echo ">>> FINITO (exit code $?)"