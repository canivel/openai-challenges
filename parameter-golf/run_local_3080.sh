#!/bin/bash
# Local run script for RTX 3080 (10GB VRAM, single GPU)
# Usage: bash run_local_3080.sh
#
# PROVEN SETTINGS (tested 2026-03-24):
# - 17M param model fits in 6.7GB VRAM (headroom to spare)
# - 118 steps in 3 min, ~1.5 sec/step without torch.compile
# - val_bpb: 2.82 after 118 steps (would improve dramatically with more time)
# - Artifact: 6.35MB int8+zlib (room for larger model)
#
# NOTE: Uses train_gpt_local.py (torch.compile disabled for Windows/no-Triton)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Download data if not present
if [ ! -f "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ]; then
    echo "=== Downloading FineWeb data... ==="
    uv pip install huggingface_hub --quiet 2>/dev/null || pip install huggingface_hub --quiet
    python data/cached_challenge_fineweb.py --train-shards 10
    echo "=== Data download complete ==="
fi

# Step 2: Install dependencies
echo "=== Checking dependencies ==="
uv pip install sentencepiece numpy --quiet 2>/dev/null || pip install sentencepiece numpy --quiet

# Step 3: Run training (single GPU, no distributed)
echo ""
echo "=== Starting training on RTX 3080 ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo ""

# Proven config for 10GB VRAM:
# - TRAIN_BATCH_TOKENS=131072 with grad_accum=8 → 16K tokens per micro-batch
# - VAL_BATCH_SIZE=16384 to avoid OOM during validation
# - TRAIN_SEQ_LEN=512 for faster throughput on consumer GPU
export TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-131072}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16384}
export TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-512}
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-600}
export VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-200}
export TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY:-50}
export WARMUP_STEPS=${WARMUP_STEPS:-2}

echo "Config: batch=${TRAIN_BATCH_TOKENS} val_batch=${VAL_BATCH_SIZE} seq_len=${TRAIN_SEQ_LEN} time=${MAX_WALLCLOCK_SECONDS}s"
echo ""

python train_gpt_local.py 2>&1 | tee "logs/local_3080_$(date +%Y%m%d_%H%M%S).log"
