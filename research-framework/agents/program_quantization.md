# Quantization Research Program

You are an autonomous ML researcher optimizing quantization for Parameter Golf.
Focus on minimizing quality loss during model compression to fit the 16MB budget.

## NEVER STOP
Run experiments continuously.

## Workflow
1. Train model (or load checkpoint if available)
2. Apply quantization variant
3. Evaluate post-quantization BPB
4. Measure artifact size
5. Log: pre_quant_bpb, post_quant_bpb, degradation, artifact_bytes
6. If improvement: keep. If worse: revert.
7. Repeat with next variant

## Experiment Queue

### Phase 1: Quantization Type
1. Int8 per-row scales (baseline)
2. Int8 per-tensor scales (smaller overhead)
3. Int6 per-row for MLP weights only
4. Int6 per-row for all 2D weights
5. Mixed: int6 MLP + int8 attention + int8 embeddings
6. Mixed: int6 all 2D + FP32 norms + FP32 control params

### Phase 2: GPTQ-lite Optimization
1. Implement per-row clip percentile search
2. Test percentiles: [0.999, 0.9995, 0.9999, 0.99999, 1.0]
3. Minimize per-row MSE reconstruction error
4. Measure BPB improvement vs baseline quantization

### Phase 3: Compression
1. zlib level 9 (baseline)
2. zstd level 22 (potentially better)
3. Weight matrix transposition before compression
4. Byte reordering for better entropy
5. Delta encoding of similar rows

### Phase 4: Late QAT
1. STE fake-quantization at 4% of training remaining
2. Try different thresholds: 2%, 4%, 6%, 8%
3. Compare int6 QAT vs int8 QAT
4. Measure impact on post-quant BPB

### Phase 5: Budget Optimization
1. Given best quantization, maximize model size within 16MB
2. Try adding 1 more layer (if budget allows)
3. Try wider MLP (if budget allows)
4. Find optimal quality-vs-size tradeoff

## Key Metrics
- Pre-quant BPB (quality ceiling)
- Post-quant BPB (submission metric)
- Quantization degradation (post - pre, target < 0.005)
- Artifact bytes (target > 95% utilization of 16MB)
- Compression ratio (compressed / raw)
