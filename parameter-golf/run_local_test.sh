#!/bin/bash
# Local A/B testing script for RTX 3080 (10GB VRAM, single GPU)
#
# Usage:
#   bash run_local_test.sh                   # baseline (our submission config)
#   bash run_local_test.sh --name "exp1"     # named experiment
#   SLOT_STEPS=32 bash run_local_test.sh     # override any hyperparameter
#
# This runs our ACTUAL submission (records/our_submission/train_gpt.py) with
# reduced batch/seq_len to fit in 10GB VRAM. Train for 3 min, then GPTQ + SLOT.
#
# Compare RELATIVE BPB between runs — if A beats B locally, it'll likely
# transfer to H100. Don't chase absolute numbers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Parse --name argument
EXP_NAME="baseline"
while [[ $# -gt 0 ]]; do
    case $1 in
        --name) EXP_NAME="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

LOG_DIR="logs/local"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${EXP_NAME}_${TIMESTAMP}.log"

echo "============================================================"
echo "  Local A/B Test — $EXP_NAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Log: $LOG_FILE"
echo "============================================================"

# ── 3080-safe overrides (everything else inherits submission defaults) ─────────
# Batch: 131K tokens (vs 786K on H100) — fits in 10GB with grad_accum=8
export TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-131072}
# Val batch: 16K (vs 524K) — avoids OOM during eval
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16384}
# Seq len: 512 (vs 2048) — 4x less VRAM per sequence
export TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-512}
export EVAL_SEQ_LEN=${EVAL_SEQ_LEN:-512}
# Train 3 min (180s) — enough for ~100 steps to see relative differences
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-180}
# Log frequently for short runs
export VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-0}
export TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY:-10}
# Fewer warmup steps for short run
export WARMUP_STEPS=${WARMUP_STEPS:-5}
# Fewer warmdown iters (proportional to shorter run)
export WARMDOWN_ITERS=${WARMDOWN_ITERS:-500}

# SLOT: use fewer steps locally (faster, still shows relative gain)
export SLOT_STEPS=${SLOT_STEPS:-8}
export SLOT_ENABLED=${SLOT_ENABLED:-1}
export SLOT_LR=${SLOT_LR:-0.010}
export SLOT_LR_MIN=${SLOT_LR_MIN:-0.001}
export SLOT_WARMSTART=${SLOT_WARMSTART:-0.85}

# GPTQ: fewer AR-gen seqs for speed (16 instead of 64)
export GPTQ_CALIB_BATCHES=${GPTQ_CALIB_BATCHES:-8}

# Sliding window eval
export EVAL_STRIDE=${EVAL_STRIDE:-64}

# Disable torch.compile on Windows (no Triton) — runs ~2x slower but works
export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
# Enable all SDP backends for consumer GPUs (3080 doesn't support flash-only)
export LOCAL_GPU=${LOCAL_GPU:-1}

# Cap val tokens for fast eval (1M tokens vs 62M full set — ~60x faster)
export VAL_TOKENS_LIMIT=${VAL_TOKENS_LIMIT:-1000000}

# Data paths
export DATA_PATH=${DATA_PATH:-./data/datasets/fineweb10B_sp1024}
export TOKENIZER_PATH=${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}

# Run ID
export RUN_ID="local_${EXP_NAME}_${TIMESTAMP}"
export SEED=${SEED:-1337}

# ── All other params use submission defaults (from train_gpt.py) ──────────────
# NUM_LAYERS=11, MODEL_DIM=512, NUM_HEADS=8, NUM_KV_HEADS=4, etc.
# Override any of them by setting env vars before running this script.

echo ""
echo "Config: batch=${TRAIN_BATCH_TOKENS} seq=${TRAIN_SEQ_LEN} time=${MAX_WALLCLOCK_SECONDS}s slot_steps=${SLOT_STEPS}"
echo ""

# ── Run (single GPU, no torchrun) ─────────────────────────────────────────────
python records/our_submission/train_gpt.py 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  RESULTS: $EXP_NAME"
echo "============================================================"
if [ -f "$LOG_FILE" ]; then
    grep -E "final_slot_exact|final_int6_sliding_window_exact|final_int6_roundtrip_exact|Total submission|stopping_early|peak memory" "$LOG_FILE" | tail -10 || true
fi
echo ""
echo "Log: $LOG_FILE"
echo "Exit: $EXIT_CODE"
