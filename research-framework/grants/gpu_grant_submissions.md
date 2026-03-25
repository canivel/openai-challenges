# GPU Grant Submissions (Updated v3 — March 25, 2026)

## Quick Grant ($25)

### What are you going to do with it?

We validated 3 model versions locally on RTX 3080 (v1: 17M params, v2: 26.8M with 11 SOTA innovations, v3: +Differential Attention + LeakyReLU²). All train successfully. The $25 funds our first 8xH100 run to enable torch.compile (3-5x speedup), FP8 training (2x TFLOPS), and full 10-min budget — the three things we can't test locally. Our autonomous research framework tracks every experiment via git, so every dollar produces reproducible, logged results. First priority: establish true BPB with our v3 architecture.

---

## Development Grant

### Brief description of your approach (max 1,500 characters)

We built an autonomous research framework combining autoresearch (metric-driven loops) with DeerFlow (parallel agents). Validated 3 model versions on RTX 3080. Cross-referenced two independent research analyses of the competition, identifying high-value untried techniques and confirmed failures.

Our v3 model implements 13 techniques stacked from SOTA analysis:
- 11-layer transformer, 3x MLP, U-Net skips (architecture)
- XSA + Differential Attention + Partial RoPE 16d (attention)
- LeakyReLU² activation (gradient flow)
- SmearGate + BigramHash (embedding context)
- LN Scale + EMA 0.997 + orthogonal init + muP + grad clip 0.3 (training)

Planned for H100 runs (remaining 80% of expected improvement):
- FP8 training: 2x TFLOPS = +600-1000 extra steps
- Legal score-first TTT: LoRA adaptation during eval (-0.03 BPB)
- Partial weight sharing + LoRA: 8 unique → 14 effective layers
- Learned non-uniform quantization + custom codebook + Huffman
- Int6 + GPTQ-lite clip search
- Parallel Muon + parameter banking

Cross-validated confirmed failures to avoid: MoE (<500M), full depth recurrence (900x quant error), INT4, SSM/Mamba, SwiGLU, MLA, LAWA, MAML Meta-TTT.

Conservative target: ~1.07 BPB (non-TTT). Aggressive: ~1.04 BPB (with legal TTT).

### What have you tried so far? (max 255 characters)

3 model versions on RTX 3080. v3: 26.8M params, 13 SOTA techniques, LeakyReLU² + Differential Attention. Cross-validated 2 independent research analyses. Framework + agents + docs ready. Need H100 for torch.compile + FP8.

### Link(s) to your PR submission

https://github.com/canivel/openai-challenges

---

## Copy-Paste Ready Versions

### Quick Grant - Plain Text
```
We validated 3 model versions locally on RTX 3080 (v1: 17M params, v2: 26.8M with 11 SOTA innovations, v3: +Differential Attention + LeakyReLU²). All train successfully. The $25 funds our first 8xH100 run to enable torch.compile (3-5x speedup), FP8 training (2x TFLOPS), and full 10-min budget — the three things we can't test locally. Our autonomous research framework tracks every experiment via git, so every dollar produces reproducible, logged results. First priority: establish true BPB with our v3 architecture.
```

### Development Grant - Brief Description (1,496 chars)
```
We built an autonomous research framework combining autoresearch (metric-driven loops) with DeerFlow (parallel agents). Validated 3 model versions on RTX 3080. Cross-referenced two independent research analyses of the competition, identifying high-value untried techniques and confirmed failures.

Our v3 model implements 13 techniques stacked from SOTA analysis:
- 11-layer transformer, 3x MLP, U-Net skips (architecture)
- XSA + Differential Attention + Partial RoPE 16d (attention)
- LeakyReLU² activation (gradient flow)
- SmearGate + BigramHash (embedding context)
- LN Scale + EMA 0.997 + orthogonal init + muP + grad clip 0.3 (training)

Planned for H100 runs (remaining 80% of expected improvement):
- FP8 training: 2x TFLOPS = +600-1000 extra steps
- Legal score-first TTT: LoRA adaptation during eval (-0.03 BPB)
- Partial weight sharing + LoRA: 8 unique → 14 effective layers
- Learned non-uniform quantization + custom codebook + Huffman
- Int6 + GPTQ-lite clip search
- Parallel Muon + parameter banking

Cross-validated confirmed failures to avoid: MoE (<500M), full depth recurrence (900x quant error), INT4, SSM/Mamba, SwiGLU, MLA, LAWA, MAML Meta-TTT.

Conservative target: ~1.07 BPB (non-TTT). Aggressive: ~1.04 BPB (with legal TTT).
```

### What have you tried so far? (247 chars)
```
3 model versions on RTX 3080. v3: 26.8M params, 13 SOTA techniques, LeakyReLU² + Differential Attention. Cross-validated 2 independent research analyses. Framework + agents + docs ready. Need H100 for torch.compile + FP8.
```

### PR Link
```
https://github.com/canivel/openai-challenges
```
