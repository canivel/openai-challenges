# RunPod 8×H100 Run Report — 2026-04-03

## Summary

**1 complete seed run (1337), 1 partial (42 — training done, GPTQ started, pod terminated due to budget).**

## Seed 1337 Results (COMPLETE)

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | **1.0508** |
| Int6+lzma roundtrip val_bpb | 1.1549 |
| Sliding window (stride=64) val_bpb | **1.1316** |
| **SLOT (64 steps) val_bpb** | **0.6951** |
| SLOT val_loss | 1.17365611 |
| Training steps | 5,684 / 20,000 |
| Training time | 595,081 ms (595s) |
| Step avg | 104.69 ms |
| Peak VRAM | 23,639 MiB / 81,559 MiB per GPU |
| GPTQ calibration time | 411,431 ms (~6.9 min) |
| Sliding window eval time | 110,396 ms (~1.8 min) |
| SLOT eval time | 825,384 ms (**13.7 min — OVER 10 min limit**) |
| Model artifact (int6+lzma) | 15,611,244 bytes |
| Code size | 79,535 bytes |
| **Total artifact** | **15,690,779 bytes** (under 16,000,000 limit) |

## Seed 42 Results (PARTIAL — pod terminated)

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | **1.0525** |
| Training steps | 5,687 / 20,000 |
| Training time | 595,101 ms |
| Last known state | GPTQ calibrating — pod terminated |

## Bugs Found & Fixed During Run

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| 1 | `_COMPRESSOR` undefined — NameError | Would crash at quant (caught pre-run) | Added `_COMPRESSOR = "lzma"` at line 13 |
| 2 | `model(tokens, None)` in AR calib | Crash: `NoneType.reshape()` | Changed to `model.forward_logits(tokens)` |
| 3 | `torch._dynamo.config.cache_size_limit` too low | Crash: `FailOnRecompileLimitHit` after 8 recompiles | Set `cache_size_limit = 64` globally |

**Wasted ~$16 on 2 crashed runs before fixes landed.**

## Key Findings

### SLOT BPB is extraordinary
- 0.6951 BPB absolutely destroys the leaderboard SOTA (1.1147)
- Even the pending PR #1263 SOTA (0.9354) is beaten by 0.24 nats
- This is a 36% improvement over current leaderboard

### BUT: SLOT eval exceeds 10-min limit
- 64 SLOT steps × ~781K eval windows / 8 GPUs = **825 seconds (13.7 min)**
- Leaderboard rules: eval must complete in ≤ 10 min on 8×H100
- Need to reduce SLOT_STEPS to fit in 600s:
  - 64 steps → 825s (over)
  - **46 steps → ~593s** (fits with margin)
  - 44 steps → ~566s (safe)
  - 32 steps → ~412s (very safe, but worse BPB)

### Sliding window (no SLOT) is competitive but not SOTA
- 1.1316 BPB vs SOTA 1.1147 — we're 0.017 nats behind
- Quantization gap (pre-quant 1.0508 → post-quant 1.1316) = 0.081 nats

### Artifact size is fine
- 15.69 MB / 16.00 MB limit — 310 KB headroom

## Comparison to Previous Results

| Run | Pre-quant BPB | Post-quant Sliding BPB | SLOT BPB |
|-----|--------------|----------------------|----------|
| Our 1×H100 (Mar 25) | 1.3208 | — | — |
| **Our 8×H100 (today)** | **1.0508** | **1.1316** | **0.6951** |
| Leaderboard SOTA | — | 1.1147 | — |
| PR #1263 pending | — | — | 0.9354 |

## Next Steps

### Immediate (to make a valid submission)
1. **Reduce SLOT_STEPS to 46** (or 44 for safety) to fit within 600s eval budget
2. **Run 3 seeds** with the reduced steps to get statistical significance
3. **Need more compute** — minimum 3 × ~35 min × $24/hr ≈ $42

### Optimizations to try
1. **Skip intermediate evals** (roundtrip + sliding window) for submission — only compute SLOT
   - Saves ~140s of eval time → could afford SLOT_STEPS=52 instead of 46
2. **Reduce GPTQ AR generation** — 64 seqs × 2048 tokens at 4 batch = 16 forward passes through full model
   - This takes 411s — could reduce to 32 seqs (~200s) with minimal quality loss
3. **Total eval budget if optimized**: skip diagnostics (-14s), skip roundtrip (-30s), skip sliding (-110s), reduce GPTQ (-200s) → ~470s available for SLOT → could afford ~36 SLOT steps within 600s total

### For next RunPod session
- Budget needed: ~$42 minimum (3 seeds × 35 min)
- All bugs are fixed — code is ready to run
- Consider OpenAI compute grant ($1M available): https://openai.com/index/parameter-golf/#credit-form

## Files Saved Locally

```
logs/runpod_20260403/
├── seed1337.txt              # Full training log (v1+v2+v3)
├── seed1337_stdout.log       # Stdout from final v3 run
├── final_model_seed1337.int6.ptz  # Quantized model artifact (15.6 MB)
└── REPORT.md                 # This file
```
