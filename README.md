# Parameter Golf Challenge - Autonomous Research Framework

> **Goal:** Train the best small language model that fits within **16MB**, in **10 minutes** on **8x H100 GPUs**, evaluated by compression quality (bits-per-byte) on the FineWeb dataset.

## The Challenge

[OpenAI Parameter Golf](https://github.com/openai/parameter-golf) is a competition inspired by neural scaling laws. Given a fixed parameter budget, find the lowest achievable loss `L(N)` — unconstrained by compute time, data, or training steps. Participants compete to discover novel architectures, compression schemes, and training techniques.

| Constraint | Value |
|-----------|-------|
| Artifact size | <= 16,000,000 bytes (code + compressed model) |
| Training time | <= 10 minutes on 8x H100 SXM GPUs |
| Evaluation metric | BPB (bits per byte) on FineWeb validation — **lower is better** |
| Current SOTA | 1.1228 BPB (11L EMA + GPTQ-lite + warmdown3500) |
| Record threshold | Must beat SOTA by >= 0.005 nats with p < 0.01 across 3+ seeds |

## Results

### H100 GPU Runs (1x NVIDIA H100 80GB)

Our first real GPU session produced **val_bpb 1.3208** in 10 minutes on a single H100 — the model was still improving when time ran out.

| Run | Config | Time | Steps | Pre-Q BPB | Post-Q BPB | Artifact |
|-----|--------|:----:|:-----:|:---------:|:----------:|:--------:|
| 1 | v3 full stack | 5 min | 586 | 1.5492 | 2.4971 | 12.5 MB |
| 2 | v3 no DiffAttn (ablation) | 5 min | 577 | 1.5603 | 2.5356 | 12.7 MB |
| **3** | **v3 full stack** | **10 min** | **1,209** | **1.3208** | **1.4501** | **16.2 MB** |
| 4 | v3 lower LR=0.025 | 10 min | 1,119 | 1.3652 | 1.5666 | 14.6 MB |

**Key findings from H100 runs:**
- **Differential Attention confirmed helpful**: Run 1 vs 2 shows -0.011 BPB with DiffAttn enabled
- **torch.compile works**: 497 ms/step (vs 935 ms local) — 1.9x speedup even on 1 GPU
- **Quantization is the bottleneck**: Post-quant degrades by 0.13 BPB — need int6 + GPTQ-lite
- **Higher LR (0.04) beats 0.025** on 1 GPU: more steps matter when compute-limited
- **Artifact over budget** at 16.2 MB with int8 — int6 would bring it to ~12 MB

### Scaling Projection

| Hardware | Steps in 10 min | Best BPB | Status |
|----------|:--------------:|:--------:|:------:|
| RTX 3080 (local, no compile) | 127 | 3.15 | Verified |
| **1x H100 (our run)** | **1,209** | **1.32** | **Verified** |
| 8x H100 (competition target) | ~7,000-9,600 | ~1.12-1.18 | Projected |
| SOTA (8x H100, all optimized) | ~7,100 | 1.1228 | Leaderboard |

### Local Validation (RTX 3080)

Three model versions validated locally to prove code correctness:

| Version | Params | Innovations | Steps | BPB | Artifact |
|---------|:------:|:-----------:|:-----:|:---:|:--------:|
| v1 (baseline) | 17M | 0 | 118 | 2.82 | 6.35 MB |
| v2 (+11 SOTA) | 26.8M | 11 | 127 | 3.15 | 8.95 MB |
| v3 (+DiffAttn+LeakyReLU²) | 26.8M | 13 | 63 | 5.58 | 8.90 MB |

*Note: Local BPB numbers are not comparable to H100 results due to missing torch.compile, smaller batch, and shorter training time.*

## Our Approach

We built a **hybrid autonomous research framework** combining two leading agentic research systems:

- **[Karpathy's autoresearch](https://github.com/karpathy/autoresearch):** Single-agent loops, metric-driven keep/discard, git-based experiment tracking.
- **[ByteDance's DeerFlow](https://github.com/bytedance/deer-flow):** Multi-agent orchestration, parallel subagents, middleware pipelines, memory persistence.

Our framework takes autoresearch's experiment loop pattern (modify → run → measure BPB → keep or revert) and scales it with DeerFlow-inspired parallelism (multiple specialized agents in isolated git worktrees).

We also cross-referenced **two independent research analyses** of the competition (600+ PRs analyzed) to identify high-value untried techniques and confirmed failures.

## Model Architecture (v3)

Our v3 model implements **13 techniques** stacked from SOTA analysis and external research:

| Category | Technique | Source | Status |
|----------|-----------|--------|:------:|
| **Architecture** | 11 layers (was 9) | SOTA analysis | Done |
| | 3x MLP expansion (was 2x) | SOTA analysis | Done |
| | U-Net skip connections (5+6) | Baseline | Done |
| **Attention** | XSA on last 4 layers | SOTA analysis | Done |
| | Differential Attention (layers 5-10) | ICLR 2025 Oral | Done |
| | Partial RoPE (16/64 dims) | SOTA analysis | Done |
| **Activation** | LeakyReLU² (was ReLU²) | External research | Done |
| **Embeddings** | SmearGate | SOTA analysis | Done |
| | BigramHash (2048 buckets) | SOTA analysis | Done |
| **Training** | EMA (decay=0.997) | SOTA analysis | Done |
| | Gradient clipping (0.3) | SOTA analysis | Done |
| | LN Scale Factor 1/sqrt(i+1) | SOTA analysis | Done |
| | Orthogonal init + muP scaling | SOTA analysis | Done |

### Remaining High-Value Techniques (require 8x H100)

| Technique | Expected Impact | Confidence |
|-----------|:--------------:|:----------:|
| Int6 quantization + GPTQ-lite | -0.01 BPB + fits budget | High |
| Sliding window eval (stride=64) | -0.033 BPB (free) | High |
| FP8 training (2x TFLOPS) | -0.01-0.025 BPB | High |
| Legal score-first TTT | -0.03 BPB | Medium-High |
| Partial weight sharing + LoRA (8→14 layers) | -0.008-0.02 BPB | Medium |
| Learned non-uniform quantization | -0.005-0.012 BPB | Medium-High |
| Parallel Muon + parameter banking | +227 extra steps | High |

### Confirmed Failures (techniques to avoid)

From cross-validation of two independent research analyses:

| Technique | Penalty | Why |
|-----------|:-------:|-----|
| MoE (any config <500M) | -0.06-0.08 BPB | Apple ICML 2025: dense is optimal below 500M |
| Full depth recurrence | +1.14 BPB gap | 900x quantization error over 3 cycles |
| INT4 quantization | +0.065 BPB | 10x worse than int5→int6 gap |
| SSM/Mamba hybrids | +0.08 BPB gap | Underperform dense transformers at this scale |
| SwiGLU | Worse than ReLU² | Standard result at this scale |
| MLA | 2x slower | Halves throughput |
| MAML Meta-TTT | +0.085 BPB | Network too small for meta-gradients |

## Strategy: Three-Phase Optimization

### Phase 1: Architecture (BPP > 1.15) — COMPLETE
11-layer transformer, 3x MLP, XSA, Differential Attention, Partial RoPE, SmearGate, BigramHash, LeakyReLU², LN Scale. Verified on H100: **1.3208 BPB pre-quant in 10 min on 1 GPU**.

### Phase 2: Training Optimization (1.13 < BPB < 1.15) — PARTIALLY COMPLETE
EMA, gradient clipping, warmdown 3500, orthogonal init, muP scaling all implemented. Remaining: FP8 training, Parallel Muon, parameter banking (require 8x H100).

### Phase 3: Quantization & Compression (BPP < 1.13) — NEXT PRIORITY
Current int8 quantization costs 0.13 BPB degradation. Need: int6 + GPTQ-lite clip search + learned non-uniform quantization + zstd compression. This is the biggest remaining gap.

## Repository Structure

```
openai-challenges/
├── README.md                    # This file
├── CLAUDE.md                    # Project guide for Claude Code agents
├── docs/
│   ├── blog-post.md             # Narrative story of the journey
│   ├── white-paper.md           # Technical paper with charts
│   ├── fig*.png                 # Generated charts
│   └── external-researchs/     # Independent research analyses
├── parameter-golf/
│   ├── train_gpt.py             # Original baseline
│   ├── train_gpt_local.py      # Our v3 (13 innovations)
│   ├── run_local_3080.sh        # Local run script
│   └── data/                    # FineWeb dataset + tokenizer
├── research-framework/
│   ├── src/                     # Python framework (orchestrator, tracker, validator)
│   ├── agents/                  # Autonomous loop programs
│   ├── experiments/             # YAML configs + H100 run plan
│   ├── results/                 # H100 run data (TSV)
│   ├── grants/                  # GPU grant text
│   └── reviews/                 # Panel review + research synthesis
└── .claude/agents/              # 6 specialized Claude Code agents
```

## Agent System

Six specialized agents work in parallel via git worktrees:

| Agent | Role | Key Decisions |
|-------|------|--------------|
| **Orchestrator** | Strategic coordination | Which experiments to run, when to shift phases |
| **Architecture** | Model structure | Layers, attention, MLP, embeddings, skip connections |
| **Hyperparameter** | Training config | LR, schedule, optimizer, EMA, batch size |
| **Quantization** | Compression | Int6/int8, GPTQ-lite, zlib/zstd, late QAT |
| **Evaluation** | Validation | Multi-seed runs, artifact size, significance tests |
| **Literature** | Research | Papers, repos, novel techniques |

## Quick Start

### Local (RTX 3080 / consumer GPU)
```bash
cd parameter-golf
bash run_local_3080.sh
```

### H100 (single or multi-GPU)
```bash
# Single GPU
cd parameter-golf && python train_gpt_local.py

# 8x GPU (competition target)
cd parameter-golf && torchrun --nproc_per_node=8 train_gpt_local.py
```

### Research Framework
```bash
cd research-framework
python -m src.cli init my_campaign
python -m src.cli plan
python -m src.cli status
python -m src.cli best
```

## Next Steps

1. **Int6 quantization + GPTQ-lite** — close the 0.13 BPB quant gap (biggest priority)
2. **8x H100 session** — scale from 1,209 to ~7,000+ steps
3. **Sliding window eval** — free -0.033 BPB
4. **Legal TTT implementation** — potential -0.03 BPB during evaluation
5. **Multi-seed validation** — 3+ seeds for submission confidence
6. **Submit PR to openai/parameter-golf leaderboard**

## License

This project is part of the OpenAI Parameter Golf challenge. See the challenge repository for terms.
