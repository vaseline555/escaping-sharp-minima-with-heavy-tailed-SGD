#!/bin/bash
# One-time setup on a fresh Vast.ai instance (run from a Jupyter terminal).
# Assumes a CUDA + PyTorch base image. Creates data/ and logs/, installs deps.
#
#   bash scripts/setup_vast.sh

set -e
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

mkdir -p data logs/results

# If torch is missing (non-pytorch image), install a CUDA build first:
python3 -c "import torch" 2>/dev/null || \
    pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install -e ./Theta

# W&B: authenticate (entity vaseline555 is set in train_gpt2.py defaults)
wandb login || echo "Run 'wandb login' before launching, or export WANDB_API_KEY."

python3 -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available(), 'GPUs', torch.cuda.device_count())"
python3 -c "from theta import Theta; print('Theta OK')"

echo ""
echo "Setup done. Next:"
echo "  bash download_data.sh 9        # ~2 GB smoke test (or 103 for full ~21 GB)"
echo "  bash run.sh configs/base_adamw.yaml 4 --wandb"
