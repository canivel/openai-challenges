#!/bin/bash
# ============================================================
# H100 1-Hour Experiment Plan
# ============================================================
# Budget: 60 minutes on 8xH100 SXM (~$25 quick grant)
# Strategy: 6 fast screens (3 min each) → 2 full runs (10 min) → 1 multi-seed
#
# Each "screen" uses MAX_WALLCLOCK_SECONDS=180 (3 min) to quickly
# compare configs. Best config gets full 10-min runs.
# ============================================================

set -e
cd "$(dirname "$0")/../../parameter-golf"

# Download full dataset (80 shards) if not present
if [ ! -f "data/datasets/fineweb10B_sp1024/fineweb_train_000079.bin" ]; then
    echo "=== Downloading full 80-shard dataset ==="
    python data/cached_challenge_fineweb.py --train-shards 80
fi

COMMON="VAL_LOSS_EVERY=9999 TRAIN_LOG_EVERY=100"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="../research-framework/results/h100_${TIMESTAMP}.tsv"
echo -e "run\tconfig\tval_bpb\tval_loss\tsteps\tartifact_bytes\ttime_sec" > "$RESULTS_FILE"

log_result() {
    local run=$1 config=$2 logfile=$3
    local bpb=$(grep "stopping_early\|^step.*val_bpb" "$logfile" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+')
    local loss=$(grep "stopping_early\|^step.*val_loss" "$logfile" | tail -1 | grep -oP 'val_loss:\K[0-9.]+')
    local steps=$(grep "stopping_early" "$logfile" | grep -oP 'step:\K[0-9]+')
    local artifact=$(grep "int8+zlib:" "$logfile" | grep -oP '^\S+ \K[0-9]+' | head -1)
    local time=$(grep "stopping_early\|train_time" "$logfile" | tail -1 | grep -oP 'train_time:\K[0-9]+')
    echo -e "${run}\t${config}\t${bpb}\t${loss}\t${steps}\t${artifact}\t${time}" >> "$RESULTS_FILE"
    echo ">>> Run $run ($config): BPB=$bpb steps=$steps"
}

# ============================================================
# PHASE 1: SCREENING RUNS (3 min each, ~30 min total)
# Goal: Find best architecture config
# ============================================================
echo ""
echo "========== PHASE 1: SCREENING (3-min runs) =========="
echo ""

# Run 1: Our v3 baseline (11L + all innovations)
echo "--- Run 1: v3 full stack (11L/3x/XSA4/DiffAttn/LeakyReLU²/SmearGate/BigramHash) ---"
RUN_ID=screen_v3_full MAX_WALLCLOCK_SECONDS=180 $COMMON \
    torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/run1_v3_full.log
log_result 1 "v3_full" logs/run1_v3_full.log

# Run 2: v3 WITHOUT Differential Attention (ablation)
echo "--- Run 2: v3 minus DiffAttn (ablation) ---"
RUN_ID=screen_no_diff DIFF_ATTN_START=0 MAX_WALLCLOCK_SECONDS=180 $COMMON \
    torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/run2_no_diff.log
log_result 2 "v3_no_diff" logs/run2_no_diff.log

# Run 3: v3 with seq_len 2048 (expected -0.019 BPB)
echo "--- Run 3: v3 + seq2048 ---"
RUN_ID=screen_seq2048 TRAIN_SEQ_LEN=2048 MAX_WALLCLOCK_SECONDS=180 $COMMON \
    torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/run3_seq2048.log
log_result 3 "v3_seq2048" logs/run3_seq2048.log

# Run 4: v3 with larger batch (786K tokens)
echo "--- Run 4: v3 + batch 786K ---"
RUN_ID=screen_bigbatch TRAIN_BATCH_TOKENS=786432 MAX_WALLCLOCK_SECONDS=180 $COMMON \
    torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/run4_bigbatch.log
log_result 4 "v3_bigbatch" logs/run4_bigbatch.log

# Run 5: v3 with lower LR (0.025 like SOTA, vs our 0.04)
echo "--- Run 5: v3 + lr=0.025 ---"
RUN_ID=screen_lowlr MATRIX_LR=0.025 SCALAR_LR=0.025 MAX_WALLCLOCK_SECONDS=180 $COMMON \
    torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/run5_lowlr.log
log_result 5 "v3_lowlr" logs/run5_lowlr.log

# Run 6: SOTA reference stack (match known best config)
echo "--- Run 6: SOTA reference (11L/3x/XSA4/RoPE16/no DiffAttn/ReLU²/lr=0.025) ---"
RUN_ID=screen_sota DIFF_ATTN_START=0 SMEAR_GATE=0 BIGRAM_HASH_BUCKETS=0 \
    MATRIX_LR=0.025 SCALAR_LR=0.025 MAX_WALLCLOCK_SECONDS=180 $COMMON \
    torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/run6_sota_ref.log
log_result 6 "sota_ref" logs/run6_sota_ref.log

echo ""
echo "========== PHASE 1 RESULTS =========="
cat "$RESULTS_FILE"
echo ""

# ============================================================
# PHASE 2: FULL RUNS (10 min each, ~28 min total)
# Use best config from Phase 1
# ============================================================
echo ""
echo "========== PHASE 2: FULL RUNS (10-min) =========="
echo ""

# Manually pick the best config from Phase 1 results
# For now, run v3 full + SOTA reference at full 10 min
BEST_CONFIG="v3_full"  # Update this based on Phase 1 results

# Run 7: Full 10-min with best config, seed 42
echo "--- Run 7: Best config, seed=42, full 10 min ---"
RUN_ID=full_seed42 SEED=42 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=1000 TRAIN_LOG_EVERY=200 \
    torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/run7_full_s42.log
log_result 7 "full_s42" logs/run7_full_s42.log

# Run 8: Full 10-min with best config, seed 1337
echo "--- Run 8: Best config, seed=1337, full 10 min ---"
RUN_ID=full_seed1337 SEED=1337 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=1000 TRAIN_LOG_EVERY=200 \
    torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/run8_full_s1337.log
log_result 8 "full_s1337" logs/run8_full_s1337.log

echo ""
echo "========== FINAL RESULTS =========="
cat "$RESULTS_FILE"
echo ""
echo "Total time: $SECONDS seconds"
echo "Done! Check $RESULTS_FILE for all results."
