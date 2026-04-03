#!/bin/bash
# RunPod 8×H100 SXM launch script
# Usage: bash records/our_submission/run_runpod_8xh100.sh
#
# Architecture: 11L, 512d, 8 heads (4 KV), LeakyReLU(0.5)², XSA-all, QK-Gain
# Quant: Full Hessian GPTQ (Int6 MLP/attn, Int8 embed), AR self-gen calib, lzma preset=9
# Eval: Sliding window (stride=64) + SLOT (64 steps, lr=0.010→0.001, warmstart=0.85)
# Target: ~1.10 BPB (SOTA: 1.1147)

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_gpt.py"
DATA_DIR="${DATA_PATH:-/workspace/data/datasets/fineweb10B_sp1024}"
TOKENIZER="${TOKENIZER_PATH:-/workspace/data/tokenizers/fineweb_1024_bpe.model}"
LOG_DIR="$REPO_ROOT/logs"
RUN_SEED="${SEED:-1337}"
RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)_seed${RUN_SEED}}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_ID}.log"

echo "============================================================"
echo "  Parameter Golf — RunPod 8×H100 SXM Launch"
echo "  Run ID: $RUN_ID  Seed: $RUN_SEED"
echo "  Log:    $LOG_FILE"
echo "============================================================"

# ── GPU check ─────────────────────────────────────────────────────────────────
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPUs detected: $GPU_COUNT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "$GPU_COUNT" -lt 8 ]; then
    echo "WARNING: Expected 8 GPUs, found $GPU_COUNT. Proceeding anyway."
fi

# ── Dependencies ───────────────────────────────────────────────────────────────
echo ""
echo "=== Installing dependencies ==="
pip install sentencepiece numpy --quiet

# Flash Attention 3 (optional — skip if build fails)
if ! python -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    echo "flash_attn_interface not found — training will use standard attention (slower)"
fi

# ── Data check ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Checking data ==="
if [ ! -d "$DATA_DIR" ]; then
    echo "Data directory not found at $DATA_DIR — downloading FineWeb..."
    pip install huggingface_hub --quiet
    python "$REPO_ROOT/data/cached_challenge_fineweb.py" --train-shards 10
fi
TRAIN_SHARDS=$(ls "$DATA_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
VAL_SHARDS=$(ls "$DATA_DIR"/fineweb_val_*.bin 2>/dev/null | wc -l)
echo "Train shards: $TRAIN_SHARDS  Val shards: $VAL_SHARDS"
if [ "$TRAIN_SHARDS" -eq 0 ] || [ "$VAL_SHARDS" -eq 0 ]; then
    echo "ERROR: Missing data shards in $DATA_DIR"
    exit 1
fi
if [ ! -f "$TOKENIZER" ]; then
    echo "ERROR: Tokenizer not found at $TOKENIZER"
    exit 1
fi
echo "Data OK."

# ── Training config ────────────────────────────────────────────────────────────
# 8×H100 SXM (80GB each) — 600s wall clock budget
# TRAIN_BATCH_TOKENS=786432 → 786432/8/2048 = 48 tokens/GPU/step (grad_accum handles the rest)
export DATA_PATH="$DATA_DIR"
export TOKENIZER_PATH="$TOKENIZER"
export RUN_ID="$RUN_ID"
export SEED="$RUN_SEED"

# Architecture (matches our submission)
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export VOCAB_SIZE=1024
export MLP_MULT=3.0
export TIE_EMBEDDINGS=1
export ROPE_BASE=10000.0
export LOGIT_SOFTCAP=30.0

# BigramHash (new SOTA config)
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112

# XSA — all layers
export XSA_LAST_N=11

# Training schedule (8×H100, 600s budget → ~1200-1500 steps)
export MAX_WALLCLOCK_SECONDS=595     # 5s buffer
export TRAIN_BATCH_TOKENS=786432    # 786K tokens/step across all GPUs
export TRAIN_SEQ_LEN=2048
export EVAL_SEQ_LEN=2048
export VAL_BATCH_SIZE=524288
export VAL_LOSS_EVERY=4000          # rarely hit; final eval at end
export TRAIN_LOG_EVERY=100
export WARMUP_STEPS=20
export WARMDOWN_ITERS=3500
export WARMDOWN_SHAPE=sqrt

# Optimizer
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export EMBED_LR=0.6
export HEAD_LR=0.008
export TIED_EMBED_LR=0.035
export TIED_EMBED_INIT_STD=0.005
export MUON_MOMENTUM=0.99
export MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_WD=0.04
export ADAM_WD=0.04
export BETA1=0.9
export BETA2=0.95
export ADAM_EPS=1e-8
export GRAD_CLIP_NORM=0.3
export QK_GAIN_INIT=4.0

# Focal loss (enabled)
export FOCAL_GAMMA=1.0

# Depth recurrence (disabled — costs 18% compute)
export RECUR_LAYERS=""

# SWA
export SWA_ENABLED=1
export SWA_EVERY=50

# QAT (disabled — using GPTQ post-training)
export QAT_ENABLED=0

# GPTQ calibration (AR self-generated)
export GPTQ_CALIB_BATCHES=32

# SLOT evaluation (Score-First TTT)
export SLOT_ENABLED=1
export SLOT_STEPS=64
export SLOT_LR=0.010
export SLOT_LR_MIN=0.001
export SLOT_WARMSTART=0.85

# Sliding window eval
export EVAL_STRIDE=64

# VE (Value Estimator)
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"

# ── Launch ─────────────────────────────────────────────────────────────────────
# cd to repo root so final_model.pt/.int6.ptz and logs/ land in the right place
cd "$REPO_ROOT"

echo ""
echo "=== Starting training (CWD: $REPO_ROOT) ==="
echo "Batch tokens: $TRAIN_BATCH_TOKENS | Seq len: $TRAIN_SEQ_LEN | Max time: ${MAX_WALLCLOCK_SECONDS}s"
echo "SLOT: ${SLOT_STEPS} steps, lr=${SLOT_LR}→${SLOT_LR_MIN}, warmstart=${SLOT_WARMSTART}"
echo "BigramHash: ${BIGRAM_VOCAB_SIZE}×${BIGRAM_DIM} | Focal: gamma=${FOCAL_GAMMA}"
echo ""

torchrun \
    --standalone \
    --nproc_per_node="$GPU_COUNT" \
    "$TRAIN_SCRIPT" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# ── Extract results ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  RESULTS"
echo "============================================================"
if [ -f "$LOG_FILE" ]; then
    echo "--- Final BPB lines ---"
    grep -E "(final_slot_exact|final_int6_sliding_window_exact|val_bpb)" "$LOG_FILE" | tail -20 || true
    echo ""
    echo "--- Last 5 training lines ---"
    grep "step:" "$LOG_FILE" | tail -5 || true
fi

echo ""
echo "Log saved to: $LOG_FILE"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
