#!/bin/bash
# Download FineWeb-10B (GPT-2 tokenized) shards into ./data.
#
# Usage:  bash download_data.sh [num_train_shards]
#   Default:  9 shards  (~900M tokens, ~1.8 GB on disk)
#   Full:   103 shards  (~10.3B tokens, ~21 GB on disk)
#
# Each shard is ~200 MB. Budget disk = num_shards * 0.2 GB + ~0.2 GB (val).
# Runs on any machine with internet (Vast.ai, laptop). No proxy/module setup.

set -e

NUM_SHARDS="${1:-9}"
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${PROJ_DIR}/data"

mkdir -p "$DATA_DIR"

python3 -c "
import sys
from huggingface_hub import hf_hub_download

local_dir = '${DATA_DIR}'
num_shards = int(sys.argv[1])

def get(fname):
    hf_hub_download(repo_id='kjj0/fineweb10B-gpt2', filename=fname,
                    repo_type='dataset', local_dir=local_dir)
    print(f'  downloaded {fname}')

print('Downloading validation shard...')
get('fineweb_val_%06d.bin' % 0)

print(f'Downloading {num_shards} training shards...')
for i in range(1, num_shards + 1):
    get('fineweb_train_%06d.bin' % i)

print('Done.')
" "$NUM_SHARDS"

echo ""
echo "Data in: $DATA_DIR"
echo "train_gpt2.py defaults to data/fineweb_train_*.bin — no extra flags needed."
