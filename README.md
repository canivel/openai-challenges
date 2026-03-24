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

## Our Approach

We built a **hybrid autonomous research framework** that combines two leading agentic research systems:

- **[Karpathy's autoresearch](https://github.com/karpathy/autoresearch):** Single-agent loops, metric-driven keep/discard, git-based experiment tracking. Simple, elegant, autonomous.
- **[ByteDance's DeerFlow](https://github.com/bytedance/deer-flow):** Multi-agent orchestration, parallel subagents, middleware pipelines, memory persistence. Powerful and extensible.

Our framework takes autoresearch's experiment loop pattern (modify `train_gpt.py` → run → measure BPB → keep or revert) and scales it with DeerFlow-inspired parallelism (multiple specialized agents working simultaneously in isolated git worktrees).

## Strategy: Three-Phase Optimization

### Phase 1: Architecture Search (BPB > 1.15)
The biggest wins come from model architecture. Based on our analysis of all SOTA submissions:

| Technique | Expected Impact | Description |
|-----------|----------------|-------------|
| Deeper models (11-12L) | +0.05-0.10 BPB | More layers beat wider models at fixed param budget |
| U-Net skip connections | +0.05-0.10 BPB | Skip connections between encoder/decoder halves |
| Larger MLP (2.6-3.0x) | +0.03 BPB | More hidden units in feedforward layers |
| SmearGate + BigramHash | +0.020 BPB | Learned token blending with hash-based bigram embeddings |
| XSA (last 4 layers) | +0.010 BPB | Cheaper exclusive self-attention on final layers |
| Partial RoPE (16 dims) | +0.002 BPB | Apply rotary encoding to subset of head dimensions |
| LN Scale Factor | +0.002 BPB | `1/sqrt(layer_idx+1)` for depth stability |

**Why architecture first?** At 17M parameters, the model shape has outsized influence on what can be learned. A 11-layer model with U-Net skips and 2.6x MLP captures fundamentally different representations than a 9-layer baseline — no amount of hyperparameter tuning can bridge that gap.

### Phase 2: Training Optimization (1.13 < BPB < 1.15)
Once architecture is set, squeeze more from the training recipe:

| Technique | Expected Impact | Description |
|-----------|----------------|-------------|
| EMA (decay=0.997) | -0.0006 BPB | Exponential moving average of weights |
| Warmdown schedule | -0.005-0.02 BPB | 3000-3500 step linear LR decay at end |
| Muon + AdamW split | -0.01-0.05 BPB | Muon for matrices, AdamW for embeddings/scalars |
| Gradient clipping (0.3) | -0.001-0.003 BPB | Stabilize training of deep networks |
| SWA | -0.001-0.002 BPB | Stochastic weight averaging in final phase |

**Why hyperparameters second?** These are the "last mile" optimizations. Each individually contributes <0.01 BPB, but they compound. The right LR schedule + EMA + warmdown can recover 0.02-0.03 BPB total — the difference between a good submission and a record.

### Phase 3: Quantization & Compression (BPB < 1.13)
The final frontier — minimize quality loss during the 16MB squeeze:

| Technique | Expected Impact | Description |
|-----------|----------------|-------------|
| Mixed int6/int8 | -0.020 BPB | Int6 for MLP/attention, int8 for embeddings |
| GPTQ-lite | -0.0006 BPB | Per-row clip percentile optimization (free) |
| Late QAT (final 4%) | -0.0001 BPB | Quantization-aware training via STE |
| zstd level 22 | -0.010 BPB | Better compression than zlib level 9 |

**Why quantization last?** Quantization is a post-processing step — it can only preserve quality, not create it. But the difference between naive int8 and optimized mixed int6/GPTQ-lite is ~0.02 BPB, which matters at the competitive frontier.

## Local Run Results (RTX 3080)

We validated the full pipeline locally on a consumer GPU:

```
GPU:                NVIDIA GeForce RTX 3080 (10GB VRAM)
Training time:      3 minutes (118 steps)
val_bpb:            2.8187
val_loss:           4.7592
Peak VRAM:          6,688 MiB / 10,240 MiB
Artifact (int8):    6.35 MB (well under 16MB budget)
Model params:       17M (9L / 512D / 2x MLP)
Speed:              ~1.5 sec/step (no torch.compile on Windows)
```

**Key insight from local run:** The baseline 9-layer model only uses **6.35MB** of the 16MB budget. This means we can fit a significantly larger model (11-12 layers, 2.6x MLP) which is exactly what SOTA submissions do. The remaining ~9.5MB of budget is free parameter capacity waiting to be utilized.

## Why We Need H100 GPU Compute

Our local RTX 3080 validated the pipeline works, but **cannot produce competitive results** for several critical reasons:

### 1. torch.compile is Essential (3-5x speedup)
The 3080 runs on Windows without Triton, so `torch.compile` is disabled. On H100s with Triton, compiled kernels fuse operations and reduce memory bandwidth — this isn't just faster, it enables **3-5x more training steps** in the same 10-minute window. More steps = lower loss = better BPB.

### 2. 8x GPU Parallelism (8x throughput)
The challenge is designed for 8x H100 SXM with NVLink. Our local 1x 3080 processes 131K tokens/step; the target setup processes 524K tokens/step with 8-way data parallelism. Larger effective batch size improves optimization landscape and final convergence.

### 3. Hopper Architecture Advantages
H100s have hardware support for:
- **Flash Attention 3** (Hopper-optimized): 2x faster attention than Ampere
- **FP8 compute**: Native 8-bit training support
- **Higher memory bandwidth**: 3.35 TB/s vs 760 GB/s (RTX 3080)
- **80GB HBM3**: vs 10GB GDDR6X — enables full batch without micro-stepping

### 4. Full Dataset Access
We downloaded only 10/80 training shards locally (2GB vs 16GB). The full dataset provides more diverse training signal, which is critical for generalization on the validation set.

### 5. Experiment Velocity
At 1.5 sec/step locally (no compile) vs ~0.2 sec/step on 8xH100 (with compile), the H100 setup enables **~7.5x more experiments per hour**. Our research framework runs automated experiment loops — more iterations means more discoveries.

## What We Will Achieve With the Grant

### With $25 Quick Grant (~2-3 hours of 8xH100)
1. **Establish H100 baseline**: Run the unmodified `train_gpt.py` on target hardware to get reference BPB
2. **Depth scaling test**: Compare 9 vs 11 vs 12 layer models to confirm depth advantage
3. **Validate framework**: Confirm our worktree-based parallel experiment system works on the cluster
4. **First architecture improvements**: Test U-Net skip connections and larger MLP (2.6x)

### With Development Grant (~50+ hours of 8xH100)
1. **Full architecture search** (Phase 1): Systematically test all SOTA techniques — U-Net, XSA, SmearGate, BigramHash, partial RoPE. Each experiment = 10 min, so 50 hours = ~300 experiments.
2. **Hyperparameter sweep** (Phase 2): Optimize Muon LR, warmdown, EMA, SWA, gradient clipping across best architecture.
3. **Quantization pipeline** (Phase 3): Implement and benchmark mixed int6/int8, GPTQ-lite, zstd compression.
4. **Multi-seed validation**: 3+ seed runs on top configs for statistical significance.
5. **Novel technique exploration**: Test ideas from literature agent — MoE at small scale, differential attention, learned quantization scales.
6. **Record submission PR**: Prepare and submit a PR to `records/track_10min_16mb/` beating SOTA.

## Repository Structure

```
openai-challenges/
├── README.md                    # This file
├── CLAUDE.md                    # Project guide for Claude Code agents
├── parameter-golf/              # Challenge repo (cloned from OpenAI)
│   ├── train_gpt.py             # Original training script
│   ├── train_gpt_local.py       # Windows/3080 adapted version
│   ├── run_local_3080.sh        # One-command local run script
│   ├── data/                    # FineWeb dataset + tokenizer
│   └── logs/                    # Training logs
├── research-framework/          # Our autonomous research framework
│   ├── src/                     # Python framework code
│   │   ├── orchestrator.py      # Central research loop coordinator
│   │   ├── worktree_manager.py  # Git worktree management for parallelism
│   │   ├── experiment_tracker.py # Results tracking (TSV + JSON)
│   │   ├── artifact_validator.py # 16MB budget validation + size estimation
│   │   ├── config.py            # YAML experiment configuration
│   │   └── cli.py               # Command-line interface
│   ├── agents/                  # Autonomous loop programs per agent type
│   ├── experiments/             # YAML experiment configs (baseline + SOTA target)
│   ├── grants/                  # GPU grant submission text
│   └── results/                 # Experiment results database
├── autoresearch/                # Karpathy's autoresearch (reference implementation)
├── deer-flow/                   # ByteDance's DeerFlow (reference implementation)
└── .claude/agents/              # Claude Code agent definitions
    ├── research-orchestrator.md # Coordinates all agents
    ├── architecture-agent.md    # Model architecture exploration
    ├── hyperparam-agent.md      # Training optimization
    ├── quantization-agent.md    # Compression/quantization
    ├── eval-agent.md            # Validation + submission prep
    └── literature-agent.md      # Research discovery
```

## Agent System

Our research framework uses **6 specialized agents** that can work in parallel via git worktrees:

| Agent | Role | Key Decisions |
|-------|------|--------------|
| **Orchestrator** | Strategic coordination | Which experiments to run, when to shift phases |
| **Architecture** | Model structure | Layers, attention, MLP, embeddings, skip connections |
| **Hyperparameter** | Training config | LR, schedule, optimizer, EMA, batch size |
| **Quantization** | Compression | Int6/int8, GPTQ-lite, zlib/zstd, late QAT |
| **Evaluation** | Validation | Multi-seed runs, artifact size, significance tests |
| **Literature** | Research | Papers, repos, novel techniques |

Each agent follows an **autoresearch-style loop**: propose change → commit → run → measure BPB → keep if improved, revert if not. The orchestrator coordinates which agents run and shifts strategy based on cumulative results.

## Quick Start

### Local (RTX 3080 / consumer GPU)
```bash
cd parameter-golf
bash run_local_3080.sh
```

### H100 Cluster (target environment)
```bash
cd parameter-golf
torchrun --nproc_per_node=8 train_gpt.py
```

### Research Framework
```bash
cd research-framework
python -m src.cli init my_campaign    # Initialize research campaign
python -m src.cli plan                # Plan next experiments
python -m src.cli status              # View current results
python -m src.cli search-arch         # Search optimal architectures
python -m src.cli best                # View top experiments
```

## License

This project is part of the OpenAI Parameter Golf challenge. See the challenge repository for terms.
