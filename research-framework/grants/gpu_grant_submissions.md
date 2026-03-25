# GPU Grant Submissions (Updated v4 — March 25, 2026)

## Quick Grant ($25)

### What are you going to do with it?

We've already run 4 experiments on 1xH100 achieving val_bpb 1.3208 in 10 min (1,209 steps). Built int6+GPTQ-lite quantization pipeline that cuts artifact size 43.6%. The $25 funds our first 8xH100 run to scale from ~1,200 to ~7,000+ steps, test int6 vs int8 quantization gap on properly-trained weights, and run sliding window evaluation (-0.033 BPB free). Our autonomous research framework tracks every experiment via git, so every dollar produces logged, reproducible results.

---

## Development Grant

### Brief description of your approach (max 1,500 characters)

We built an autonomous research framework (autoresearch loops + DeerFlow parallel agents). Ran 4 experiments on 1xH100: best pre-quant BPB 1.3208 in 10 min. Validated 3 model versions locally on RTX 3080. Cross-referenced two independent research analyses + 4 quantization papers (TurboQuant, QJL, PolarQuant).

Our v3 model (26.8M params) implements 13 stacked techniques:
- 11-layer transformer, 3x MLP, U-Net skips, tied embeddings
- XSA + Differential Attention + Partial RoPE 16d
- LeakyReLU², SmearGate, BigramHash 2048
- EMA 0.997, orthogonal init, muP, LN Scale, grad clip 0.3

Implemented int6+GPTQ-lite quantization (per-row optimal clip search across 5 percentiles). Locally verified: 43.6% smaller artifacts. SOTA shows 0.001 BPB quant gap vs our current 0.13.

Remaining (need 8xH100):
- Scale from 1,209→7,100 steps (8x GPU parallelism)
- Int6 quantization gap measurement on trained weights
- Sliding window eval stride=64 (-0.033 BPB free)
- FP8 training: 2x TFLOPS = +600-1000 extra steps
- Late QAT with int6 fake-quantization
- Multi-seed validation (3+ seeds, p<0.01)

Cross-validated failures to avoid: MoE (<500M), full depth recurrence, INT4, SSM/Mamba, SwiGLU, MLA.

Target: ~1.10-1.12 BPB (competitive with SOTA 1.1228).

### What have you tried so far? (max 255 characters)

4 runs on 1xH100: best 1.3208 BPB. 3 local versions on RTX 3080. Built int6+GPTQ-lite quant (43.6% smaller). Analyzed 4 quant papers. 13-technique model stack. Framework + agents ready. Need 8xH100 for full training + eval.

### Link(s) to your PR submission

https://github.com/canivel/openai-challenges

---

## Copy-Paste Ready Versions

### Quick Grant - Plain Text
```
We've already run 4 experiments on 1xH100 achieving val_bpb 1.3208 in 10 min (1,209 steps). Built int6+GPTQ-lite quantization pipeline that cuts artifact size 43.6%. The $25 funds our first 8xH100 run to scale from ~1,200 to ~7,000+ steps, test int6 vs int8 quantization gap on properly-trained weights, and run sliding window evaluation (-0.033 BPB free). Our autonomous research framework tracks every experiment via git, so every dollar produces logged, reproducible results.
```

### Development Grant - Brief Description (1,497 chars)
```
We built an autonomous research framework (autoresearch loops + DeerFlow parallel agents). Ran 4 experiments on 1xH100: best pre-quant BPB 1.3208 in 10 min. Validated 3 model versions locally on RTX 3080. Cross-referenced two independent research analyses + 4 quantization papers (TurboQuant, QJL, PolarQuant).

Our v3 model (26.8M params) implements 13 stacked techniques:
- 11-layer transformer, 3x MLP, U-Net skips, tied embeddings
- XSA + Differential Attention + Partial RoPE 16d
- LeakyReLU², SmearGate, BigramHash 2048
- EMA 0.997, orthogonal init, muP, LN Scale, grad clip 0.3

Implemented int6+GPTQ-lite quantization (per-row optimal clip search across 5 percentiles). Locally verified: 43.6% smaller artifacts. SOTA shows 0.001 BPB quant gap vs our current 0.13.

Remaining (need 8xH100):
- Scale from 1,209→7,100 steps (8x GPU parallelism)
- Int6 quantization gap measurement on trained weights
- Sliding window eval stride=64 (-0.033 BPB free)
- FP8 training: 2x TFLOPS = +600-1000 extra steps
- Late QAT with int6 fake-quantization
- Multi-seed validation (3+ seeds, p<0.01)

Cross-validated failures to avoid: MoE (<500M), full depth recurrence, INT4, SSM/Mamba, SwiGLU, MLA.

Target: ~1.10-1.12 BPB (competitive with SOTA 1.1228).
```

### What have you tried so far? (254 chars)
```
4 runs on 1xH100: best 1.3208 BPB. 3 local versions on RTX 3080. Built int6+GPTQ-lite quant (43.6% smaller). Analyzed 4 quant papers. 13-technique model stack. Framework + agents ready. Need 8xH100 for full training + eval.
```

### PR Link
```
https://github.com/canivel/openai-challenges
```
