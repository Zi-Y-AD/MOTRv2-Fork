#!/bin/bash

# -----------------------------
#  Configuration GPU
# -----------------------------
# Pour un seul GPU, pas besoin de CUDA_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=0   # facultatif

# -----------------------------
#  Chemins des fichiers
# -----------------------------
CONFIG_FILE="configs/motrv2_deeptracel.py"          # fichier de config custom
WORK_DIR="$HOME/micromamba_envs/motrv2/work_dirs/motrv2-fork"  # dossier pour checkpoints/logs
DET_DB="DeepTracel/Transformer_Based/MOTRv2-fork/det_db_motrv2.json"           # ton fichier de détection
DATASET_ROOT="Data/MyDataset/post_split"            # racine dataset (train/val)

# -----------------------------
#  Lancer l'entraînement
# -----------------------------
python tools/motr_train.py \
    --cfg $CONFIG_FILE \
    --work-dir $WORK_DIR \
    --det-db $DET_DB \
    --dataset-root $DATASET_ROOT \
    --gpus 1