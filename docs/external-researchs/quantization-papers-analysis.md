# Quantization Research Analysis for Parameter Golf

## Papers Reviewed (March 25, 2026)

### 1. TurboQuant (Google Research, arxiv 2504.19874)
**"TurboQuant: Redefining AI Efficiency with Extreme Compression"**

Core technique: Random orthogonal rotation → optimal Lloyd-Max scalar quantization per coordinate.

**Transferable insights:**
- **Mixed-precision outlier handling** (Section 4.3): Split channels into outlier/non-outlier, apply different bit-widths. Our implementation: embeddings stay int8, MLP/attention weights go int6.
- **Lloyd-Max non-uniform centroids**: At low bit-widths (int4-int6), non-uniform quantization levels track Gaussian weight distributions 10-20% better than uniform.
- **Entropy-aware storage**: zlib already captures most benefit, but int6 values have non-uniform distribution that compresses better.

**NOT applicable:** The KV-cache focus, QJL residual correction, random rotation overhead (512x512 = 0.5MB per matrix).

### 2. QJL (arxiv 2406.03482)
**"1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead"**

Core technique: Sign-bit quantization of JL-projected key embeddings.

**Transferable insights:**
- **Outlier-aware channel splitting** for sub-8-bit quantization
- Orthogonalized projections improve quality (relates to our Muon optimizer's Newton-Schulz)

**NOT applicable:** Designed for inference KV-cache compression, not weight storage. Sign-bit (1-bit) is far too aggressive for model weights.

### 3. PolarQuant (arxiv 2502.02617)
**"PolarQuant: Polar Coordinate Quantization"**

Core technique: Convert to polar coordinates, quantize angles with level-specific codebooks.

**Transferable insights:**
- **Per-tensor sensitivity-based bit allocation** (HIGH VALUE): Different layers tolerate different precision. Allocate int5 to tolerant layers, int7 to sensitive ones.
- **Preconditioning smooths outliers**: Random rotation before quantization distributes weight magnitudes uniformly.

**NOT applicable:** Per-row scale overhead is only 0.7% of our model — PolarQuant's scale elimination doesn't help.

## Key Implementation: Int6 + GPTQ-lite

Based on SOTA analysis + these papers, we implemented:

### Architecture
```
Int6 GPTQ-lite per-row (MLP + attention weights)
├── Try 5 clip percentiles: [0.999, 0.9995, 0.9999, 0.99999, 1.0]
├── Pick percentile minimizing per-row reconstruction MSE
├── Scale = clip_abs / 31 (int6 range: [-32, 31])
└── Store in int8 containers + fp16 per-row scales

Int8 per-row (embedding weights)
├── Dual-use (input + output) needs higher fidelity
├── Standard 99.99984 percentile clipping
└── Scale = clip_abs / 127

FP32 passthrough (control tensors)
└── attn_scale, mlp_scale, resid_mix, skip_weights, q_gain

FP16 passthrough (small tensors < 65536 elements)
└── Norms, biases, etc.
```

### Local Test Results (RTX 3080, undertrained model)

| Mode | Compressed | Total | Fits 16MB | RMSE | Time |
|------|:---------:|:-----:|:---------:|:----:|:----:|
| int8 | 8.85 MB | 8.90 MB | Yes | 0.00172 | 0.4s |
| int6 | 4.99 MB | 5.05 MB | Yes | 0.00690 | 0.0s |
| int6_gptq | 4.99 MB | 5.05 MB | Yes | 0.00690 | 1.6s |

**Note:** GPTQ-lite shows no improvement on undertrained model (weights too uniform). On H100 with 1,200+ steps, weight distributions diverge and GPTQ-lite's per-row search provides ~0.0006 BPB improvement (verified by SOTA submissions).

### Realistic Size Projections (SOTA compression ratios)

With properly-trained models (more weight entropy → less compressible):
- **11L/512D/3xMLP (26.5M)**: ~15.3 MB with int6+zlib ✓
- **12L/512D/3xMLP (28.9M)**: ~16.6 MB — OVER budget ✗
- **11L/512D/4xMLP (32.3M)**: ~18.6 MB — OVER budget ✗

The SOTA config (11L/512D/3xMLP) is already the optimal architecture for the 16MB budget. Int6 doesn't enable larger models — it reduces quantization quality loss.

## Next Steps

1. **H100 validation**: Run int8 vs int6 vs int6_gptq on properly-trained model (experiment script ready)
2. **Measure real BPB gap**: SOTA shows 0.001 BPB gap with int6_gptq vs 0.13 BPB with our current int8
3. **Late QAT with int6**: Train last 4% of steps with int6 fake-quantization
4. **zstd-22 compression**: Test zstd level 22 vs zlib level 9 (expected ~5% better)
