#!/bin/bash
# Single-node launch (Vast.ai / local). One training run.
#
# Usage:  bash run.sh [config.yaml] [ngpus] [extra args...]
#   bash run.sh configs/base_adamw.yaml 4 --lr 1e-3 --wandb
#
# ngpus defaults to 4 (override with arg 2, or "all" for every visible GPU).

set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

CONFIG="${1:-configs/base_adamw.yaml}"; shift || true
NGPUS="${1:-4}"; shift || true
[ "$NGPUS" = "all" ] && NGPUS="$(python3 -c 'import torch;print(torch.cuda.device_count())')"

mkdir -p logs/results

torchrun --standalone --nproc_per_node="$NGPUS" \
    train_gpt2.py --config "$CONFIG" "$@"
