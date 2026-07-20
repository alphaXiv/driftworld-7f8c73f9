#!/usr/bin/env bash
set -euo pipefail

python -m pip install --quiet --disable-pip-version-check -r requirements.txt

DATASET=/tmp/pusht_cchi_v7_replay.zarr.zip
if [[ ! -s "$DATASET" ]]; then
  echo "DATASET_DOWNLOAD source=DiffusionPolicy_PushT file_id=1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq"
  gdown --fuzzy 'https://drive.google.com/file/d/1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq/view?usp=sharing' -O "$DATASET"
fi

echo "RUN_COMMAND bash run.sh"
echo "KUBERNETES_GPU_EXPECTED NVIDIA RTX PRO 6000 Blackwell"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
torchrun --standalone --nproc_per_node=8 reproduce.py --dataset "$DATASET"
