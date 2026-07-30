#!/bin/bash
# Single-node sweep runner for Vast.ai (replaces the ALCF PBS qsub sweeps).
# Runs each config in the grid sequentially with torchrun on this box.
# Skips runs already finished (log contains "Done. Total time:").
# W&B logging is on by default (entity = vaseline555, set in train_gpt2.py).
#
# Usage:
#   bash scripts/sweep.sh <preset> [ngpus]
#
# Presets:
#   small_adamw            lr grid, GPT-2 small, AdamW baseline
#   small_muon             lr grid, GPT-2 small, Muon+AdamW baseline
#   small_theta_adamw      config x grad_clip, GPT-2 small, Theta AdamW
#   small_theta_muon       config x grad_clip, GPT-2 small, Theta Muon+AdamW
#   medium_adamw           lr grid, GPT-2 medium, AdamW baseline
#   medium_theta_muon      config x grad_clip, GPT-2 medium, Theta Muon+AdamW
#
# ngpus defaults to 4 (pass "all" to use every visible GPU).
# Example:
#   nohup bash scripts/sweep.sh small_theta_muon 4 > sweep.log 2>&1 &

set -e
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

PRESET="${1:?usage: sweep.sh <preset> [ngpus]}"
NGPUS="${2:-4}"
[ "$NGPUS" = "all" ] && NGPUS="$(python3 -c 'import torch;print(torch.cuda.device_count())')"

LOG_DIR="logs/results"
mkdir -p "$LOG_DIR"

# ---- Best baseline lr for theta sweeps (update after baseline sweep) ----
BEST_LR="1e-3"

# ---- Shared per-preset settings ----
case "$PRESET" in
  small_adamw|small_muon|small_theta_adamw|small_theta_muon)
      MODEL_ARGS="--batch-tokens 65536 --val-interval 100 --ckpt-interval 500 --log-interval 50 --train-steps 5000"
      SIZE_ARGS="" ;;                          # small = train_gpt2.py defaults (12/12/768)
  medium_adamw|medium_theta_muon)
      MODEL_ARGS="--batch-tokens 524288 --val-interval 100 --ckpt-interval 500 --log-interval 50 --train-steps 5000"
      SIZE_ARGS="--n-layer 24 --n-head 16 --n-embd 1024" ;;
  *) echo "Unknown preset: $PRESET"; exit 1 ;;
esac

is_completed() {  # $1 = run name
    for f in "${LOG_DIR}/${1}_"*.log; do
        [ -f "$f" ] && grep -q "Done\. Total time:" "$f" 2>/dev/null && return 0
    done
    return 1
}

run_one() {  # $1 = name, $2 = config, $3... = extra args
    local NAME="$1" CONFIG="$2"; shift 2
    if is_completed "$NAME"; then echo "  [done] $NAME"; return 0; fi
    local CKPT="${LOG_DIR}/${NAME}_latest.pt"
    local RESUME=""; [ -f "$CKPT" ] && RESUME="--resume $CKPT"
    local LOG="${LOG_DIR}/${NAME}_$(date +%Y%m%d_%H%M%S).log"
    echo "  [run]  $NAME  (ngpus=$NGPUS)"
    torchrun --standalone --nproc_per_node="$NGPUS" \
        train_gpt2.py --config "$CONFIG" $SIZE_ARGS $MODEL_ARGS "$@" \
        --ckpt-path "$CKPT" $RESUME \
        --wandb --wandb-run-name "$NAME" --wandb-group-name "$WANDB_GROUP" \
        2>&1 | tee "$LOG"
}

LRS=("4e-4" "8e-4" "1e-3" "2e-3")
GRAD_CLIPS=("0.1" "0.25" "0.5" "1.0")

case "$PRESET" in
  small_adamw)
      WANDB_GROUP="small_adamw"
      for LR in "${LRS[@]}"; do
          run_one "small_adamw_lr${LR}" configs/base_adamw.yaml --lr "$LR"
      done ;;
  small_muon)
      WANDB_GROUP="small_muon"
      for LR in "${LRS[@]}"; do
          run_one "small_muon_lr${LR}" configs/base_muon_adam.yaml --lr "$LR"
      done ;;
  small_theta_adamw)
      WANDB_GROUP="small_theta_adamw"
      for CFG in theta_sb1_adamw theta_rademacher_adamw; do
          for GC in "${GRAD_CLIPS[@]}"; do
              run_one "small_${CFG}_gc${GC}" "configs/${CFG}.yaml" --lr "$BEST_LR" --grad-clip "$GC"
          done
      done ;;
  small_theta_muon)
      WANDB_GROUP="small_theta_muon"
      for CFG in theta_sb1_muon_adam theta_rademacher_muon_adam; do
          for GC in "${GRAD_CLIPS[@]}"; do
              run_one "small_${CFG}_gc${GC}" "configs/${CFG}.yaml" --lr "$BEST_LR" --grad-clip "$GC"
          done
      done ;;
  medium_adamw)
      WANDB_GROUP="medium_adamw"
      for LR in "${LRS[@]}"; do
          run_one "medium_adamw_lr${LR}" configs/base_adamw.yaml --lr "$LR"
      done ;;
  medium_theta_muon)
      WANDB_GROUP="medium_theta_muon"
      for CFG in theta_sb1_muon_adam theta_rademacher_muon_adam; do
          for GC in "${GRAD_CLIPS[@]}"; do
              run_one "medium_${CFG}_gc${GC}" "configs/${CFG}.yaml" --lr "$BEST_LR" --grad-clip "$GC"
          done
      done ;;
esac

echo "Sweep '${PRESET}' complete."
