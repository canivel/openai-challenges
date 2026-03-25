#!/bin/bash
# ============================================================
# H100 Quantization Experiment Plan
# ============================================================
# Goal: Measure real int6 vs int8 BPB gap on properly-trained models
# Budget: ~30 min (2 full runs + 2 quant-only tests)
# ============================================================
set -e
cd "$(dirname "$0")/../../parameter-golf"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="logs/quant_results_${TIMESTAMP}.txt"

echo "=== Quantization Experiment $TIMESTAMP ===" | tee "$RESULTS"
echo "" | tee -a "$RESULTS"

# ============================================================
# RUN 1: Full 10-min training with default int8 quantization
# ============================================================
echo "--- RUN 1: Full 10 min, int8 quant ---" | tee -a "$RESULTS"
QUANT_MODE=int8 SEED=42 MAX_WALLCLOCK_SECONDS=600 \
    python train_gpt_local.py 2>&1 | tee logs/quant_run1_int8.log
# Save the pre-quant model for re-quantizing
cp final_model.pt final_model_run1.pt

echo "" | tee -a "$RESULTS"
grep -E "val_bpb|stopping|peak|int8_zlib|Quantization mode" logs/quant_run1_int8.log >> "$RESULTS"

# ============================================================
# RUN 2: Re-quantize same model with int6 (no retraining!)
# ============================================================
echo "" | tee -a "$RESULTS"
echo "--- Re-quantize Run 1 model with int6 ---" | tee -a "$RESULTS"
QUANT_MODE=int6 python test_quant_modes.py 2>&1 | tee -a "$RESULTS"

# ============================================================
# RUN 3: Re-quantize same model with int6_gptq
# ============================================================
echo "" | tee -a "$RESULTS"
echo "--- Re-quantize Run 1 model with int6_gptq ---" | tee -a "$RESULTS"
QUANT_MODE=int6_gptq python test_quant_modes.py 2>&1 | tee -a "$RESULTS"

# ============================================================
# RUN 4: Full 10-min training with int6_gptq from the start
# (model is aware of int6 during late QAT if enabled)
# ============================================================
echo "" | tee -a "$RESULTS"
echo "--- RUN 4: Full 10 min, int6_gptq quant ---" | tee -a "$RESULTS"
QUANT_MODE=int6_gptq SEED=42 MAX_WALLCLOCK_SECONDS=600 \
    python train_gpt_local.py 2>&1 | tee logs/quant_run4_int6gptq.log

echo "" | tee -a "$RESULTS"
grep -E "val_bpb|stopping|peak|int8_zlib|Quantization mode" logs/quant_run4_int6gptq.log >> "$RESULTS"

# ============================================================
# SUMMARY
# ============================================================
echo "" | tee -a "$RESULTS"
echo "=== SUMMARY ===" | tee -a "$RESULTS"
echo "Run 1 (int8):      $(grep final_int8_zlib_roundtrip_exact logs/quant_run1_int8.log)" | tee -a "$RESULTS"
echo "Run 4 (int6_gptq): $(grep final_int8_zlib_roundtrip_exact logs/quant_run4_int6gptq.log)" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"
echo "Done! Results in $RESULTS"
