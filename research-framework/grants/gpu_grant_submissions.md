# GPU Grant Submissions

## Quick Grant ($25)

### What are you going to do with it?

We've already validated our pipeline locally on an RTX 3080 (val_bpb: 2.82, 118 steps in 3 min, 6.35MB artifact). The $25 will fund our first 8xH100 run to establish the true baseline BPB with torch.compile + full batch + 80 data shards. We'll then run architecture scaling tests (9 vs 11 vs 12 layers) and U-Net skip connections — the two highest-impact changes identified from SOTA analysis (each worth +0.05-0.10 BPB). Our autonomous research framework automates experiment loops with git-based tracking, so every dollar of compute produces logged, reproducible results.

---

## Development Grant

### Brief description of your approach (max 1,500 characters)

We built an autonomous research framework combining Karpathy's autoresearch (metric-driven experiment loops, git-based tracking) with ByteDance's DeerFlow (parallel agent orchestration). We validated locally on RTX 3080: 17M-param baseline trains to val_bpb 2.82 in 3 min, artifact 6.35MB — proving the pipeline works end-to-end.

Our system uses 5 specialized agents in parallel git worktrees:

1. Architecture Agent: Depth scaling (11-12L), U-Net skips, XSA attention, SmearGate, BigramHash embeddings — guided by artifact size estimation to maximize parameters within 16MB.

2. Hyperparameter Agent: Sweeps Muon+AdamW optimizer (matrix_lr, scalar_lr), LR warmdown (3000-3500 steps), EMA (decay 0.997), SWA, gradient clipping — one variable at a time, then interaction testing.

3. Quantization Agent: Mixed int6/int8 quantization, GPTQ-lite per-row clip search, zstd compression, late QAT — minimizing quality loss during 16MB compression.

4. Evaluation Agent: Multi-seed validation (3+ seeds, p<0.01), artifact size verification, submission preparation.

5. Literature Agent: Novel techniques — MoE at small scale, differential attention, learned quantization.

Three-phase strategy: architecture first (biggest BPB gains), then training optimization, then quantization polish. Local run shows 6.35MB artifact vs 16MB budget — room for ~2.5x more parameters via deeper/wider architecture. With H100 compute: torch.compile (3-5x speedup), 8-GPU parallelism, Flash Attention 3, and full 80-shard dataset access.

### What have you tried so far? (max 255 characters)

Built research framework, analyzed SOTA (1.1228 BPB). Ran local on RTX 3080: 17M params, val_bpb 2.82, 6.35MB artifact in 3 min. Pipeline works end-to-end. Need H100s for torch.compile + full batch + competitive training.

### Link(s) to your PR submission

https://github.com/canivel/openai-challenges

---

## Copy-Paste Ready Versions

### Quick Grant - Plain Text
```
We've already validated our pipeline locally on an RTX 3080 (val_bpb: 2.82, 118 steps in 3 min, 6.35MB artifact). The $25 will fund our first 8xH100 run to establish the true baseline BPB with torch.compile + full batch + 80 data shards. We'll then run architecture scaling tests (9 vs 11 vs 12 layers) and U-Net skip connections — the two highest-impact changes identified from SOTA analysis (each worth +0.05-0.10 BPB). Our autonomous research framework automates experiment loops with git-based tracking, so every dollar of compute produces logged, reproducible results.
```

### Development Grant - Brief Description (1,489 chars)
```
We built an autonomous research framework combining Karpathy's autoresearch (metric-driven experiment loops, git-based tracking) with ByteDance's DeerFlow (parallel agent orchestration). We validated locally on RTX 3080: 17M-param baseline trains to val_bpb 2.82 in 3 min, artifact 6.35MB — proving the pipeline works end-to-end.

Our system uses 5 specialized agents in parallel git worktrees:

1. Architecture Agent: Depth scaling (11-12L), U-Net skips, XSA attention, SmearGate, BigramHash embeddings — guided by artifact size estimation to maximize parameters within 16MB.

2. Hyperparameter Agent: Sweeps Muon+AdamW optimizer (matrix_lr, scalar_lr), LR warmdown (3000-3500 steps), EMA (decay 0.997), SWA, gradient clipping — one variable at a time, then interaction testing.

3. Quantization Agent: Mixed int6/int8 quantization, GPTQ-lite per-row clip search, zstd compression, late QAT — minimizing quality loss during 16MB compression.

4. Evaluation Agent: Multi-seed validation (3+ seeds, p<0.01), artifact size verification, submission preparation.

5. Literature Agent: Novel techniques — MoE at small scale, differential attention, learned quantization.

Three-phase strategy: architecture first (biggest BPB gains), then training optimization, then quantization polish. Local run shows 6.35MB artifact vs 16MB budget — room for ~2.5x more parameters via deeper/wider architecture. With H100 compute: torch.compile (3-5x speedup), 8-GPU parallelism, Flash Attention 3, and full 80-shard dataset access.
```

### What have you tried so far? (253 chars)
```
Built research framework, analyzed SOTA (1.1228 BPB). Ran local on RTX 3080: 17M params, val_bpb 2.82, 6.35MB artifact in 3 min. Pipeline works end-to-end. Need H100s for torch.compile + full batch + competitive training.
```

### PR Link
```
https://github.com/canivel/openai-challenges
```
