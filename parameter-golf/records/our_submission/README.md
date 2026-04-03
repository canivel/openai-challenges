# Record: 11L LeakyReLU² + XSA-all + Full GPTQ + SLOT(64) + AR-calib + BigramHash(3072,112)

## Architecture
- 11 layers, 512 model dim, 8 heads, 4 KV heads (GQA)
- **LeakyReLU(0.5)² MLP** with 3x expansion (1536 hidden)
- **XSA (Exclusive Self Attention) on all 11 layers** (xsa_last_n=11)
- QK-Gain initialization: 4.0
- Partial RoPE: 16/64 dims
- U-Net skip connections (5 encoder + 6 decoder)
- SmearGate + **BigramHash (3072 buckets, 112d)**
- VE128 shared value embedding on layers 9-10
- LN scale: 1/sqrt(layer+1)
- Logit softcap: 30.0
- Focal loss: γ=1.0

## Training Configuration
- Parallel Muon: lr=0.025, momentum 0.92→0.99 over 1500 steps, WD=0.04
- AdamW: embed lr=0.035, scalar lr=0.025, WD=0.04
- EMA decay=0.997 + SWA every 50 steps
- No QAT (using post-training GPTQ)
- **Sqrt warmdown** over 3500 iterations
- Batch: 786,432 tokens, seq_len=2048
- Depth recurrence: **disabled** (costs 18% compute, hurts under fixed time budget)

## Quantization
- **Full Hessian GPTQ** (Cholesky error compensation + column reordering)
- **AR self-generated calibration**: model generates 64×2048 tokens at temp=0.8
  - No training data leakage; calibration distribution matches model's own distribution
- Multi-percentile clip search (5 percentiles)
- Int6 for MLP+attention, Int8 for embeddings
- **lzma preset=9** compression (replaces zstd-22; better ratio for this blob type)

## Evaluation
- Sliding window (stride=64, seq_len=2048)
- **SLOT (Score-First Test-time Learning)**: 64 AdamW steps per eval window
  - Cosine LR: 0.010 → 0.001
  - **Warmstart=0.85**: optimizer state inherited 85% from previous window
  - Model weights frozen; only logit delta + bias optimized

## Key Changes vs Previous Submission
| Component | Before | After |
|-----------|--------|-------|
| Compression | zstd-22 | **lzma preset=9** |
| BigramHash | (2048, 128) | **(3072, 112)** |
| GPTQ calibration | FineWeb samples | **AR self-generated** |
| SLOT steps | 16 | **64** |
| SLOT lr | 0.008→0.0008 | **0.010→0.001** |
| SLOT warmstart | 0.0 (disabled) | **0.85** |
| Focal loss | disabled | **γ=1.0** |
| Depth recurrence | enabled | **disabled** |
| Warmdown shape | linear | **sqrt** |

## Results

| Seed | SLOT BPB | Sliding BPB |
|------|----------|-------------|
| 1337 | TBD | TBD |
| 42   | TBD | TBD |
| 0    | TBD | TBD |
| Mean | TBD | TBD |

Target: < 1.1097 BPB (SOTA 1.1147 − 0.005 improvement threshold)

## Reproducing

```bash
# On RunPod 8×H100 SXM pod (template: y5cejece4j)
git clone https://github.com/canivel/openai-challenges /workspace/openai-challenges
cd /workspace/openai-challenges/parameter-golf
python data/cached_challenge_fineweb.py --variant sp1024

# Seed 1337
bash records/our_submission/run_runpod_8xh100.sh

# Seed 42
SEED=42 bash records/our_submission/run_runpod_8xh100.sh

# Seed 0
SEED=0 bash records/our_submission/run_runpod_8xh100.sh
```

## Statistical Validation

After 3 seed runs, run:
```bash
python records/our_submission/analyze_runs.py \
    logs/run_*_seed1337.log \
    logs/run_*_seed42.log \
    logs/run_*_seed0.log \
    --make-pr-folder
```

This verifies p < 0.01, improvement >= 0.005 nats, and generates the PR folder.
