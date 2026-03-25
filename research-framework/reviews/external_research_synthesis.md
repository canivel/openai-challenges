# External Research Synthesis & Cross-Validation

**Sources analyzed:**
- Claude Researchers: "Five Novel Techniques to Push Below 1.10 BPB"
- Google Researchers: "Optimization under Extreme Constraints" (exhaustive academic analysis)

**Updated SOTA (as of March 25, 2026):**
- Merged SOTA: **1.1228 BPB** (PR #414)
- Pending non-TTT: **1.1175 BPB**
- Pending legal TTT: **1.1164 BPB**
- Disputed multi-pass TTT: **1.0523 BPB** (PR #573, legality challenged)
- GEPA evolutionary search: **1.0672 BPB** (independently verified)

---

## I. TECHNIQUE CROSS-VALIDATION MATRIX

Both research teams independently analyzed the competition. Here's where they **agree**, **disagree**, and what's **unique** to each.

### STRONG AGREEMENT (Both teams recommend, high confidence):

| Technique | Claude Est. | Google Coverage | Our Status | Priority |
|-----------|:-----------:|:--------------:|:----------:|:--------:|
| FP8 Training | -0.01-0.025 BPB | Confirmed: 2x TFLOPS on H100 | NOT IMPL | **P0** (H100) |
| Differential Attention | -0.005-0.015 BPB | Extensive: GDA section, ICLR Oral | NOT IMPL | **P0** (local) |
| Learned Non-Uniform Quant | -0.005-0.012 BPB | Confirmed: SqueezeLLM, CALDERA | NOT IMPL | **P1** (local) |
| LeakyReLU² over ReLU² | (implicit in stack) | Explicit: prevents dead neurons | NOT IMPL | **P0** (local) |
| EMA >> SWA | Confirmed | Confirmed: 0.003 BPB advantage | IMPL | Done |
| Warmdown 3500 | Part of SOTA stack | Confirmed | IMPL | Done |
| Grad clipping 0.3 | Part of SOTA stack | Confirmed | IMPL | Done |

### UNIQUE TO CLAUDE RESEARCH:

| Technique | Estimate | Risk | Our Assessment |
|-----------|:--------:|:----:|:---------------|
| Partial Weight Sharing + LoRA | -0.008-0.02 BPB | Medium | **HIGH VALUE**: 8 unique → 14 effective layers. Avoids full recurrence failure. Uses Cycle(Rev) ordering. |
| BitNet 1.58-bit QAT | -0.01-0.03 BPB | High | **EXPERIMENTAL**: ~45M params in 16MB. Needs from-scratch ternary training. High risk, high reward. |
| Custom Codebook + Huffman | -21% vs int6+zstd | Low | **HIGH VALUE**: Frees ~2MB for more params. Low risk. |

### UNIQUE TO GOOGLE RESEARCH:

| Technique | Description | Our Assessment |
|-----------|:----------:|:---------------|
| TTT (Test-Time Training) | LoRA adaptation during eval | **HIGH VALUE**: Legal score-first TTT pushes 1.1164 BPB. Must implement for competitive entry. |
| GEPA (Genetic-Pareto Evolution) | AI-driven architecture search | **MEDIUM VALUE**: Achieved 1.0672 BPB. Integrates with our agent framework. |
| Parallel Muon + Param Banking | Fuse 66 weights → 4 banks | **HIGH VALUE** (H100): -3.1% step time = +227 steps. |
| Hypernetworks | 9.34x compression | **EXPERIMENTAL**: 2.8M → 26.5M params. 2.09MB artifact. Radical. |
| Neural Cellular Automata | Pre-pre-training | **LOW PRIORITY**: 8.6% PPL reduction interesting but requires extra pipeline. |
| MatMul-Free Ternary | P-RCSM architecture | **LOW PRIORITY**: 4.1M params, CPU-focused, not competitive on BPB. |

---

## II. CONFIRMED FAILURES (Avoid These)

Both teams converge on techniques that **definitively do not work**:

| Technique | Penalty | Source | Reason |
|-----------|:-------:|:------:|--------|
| **MoE (any config <500M)** | -0.06 to -0.08 BPB | Both | Apple ICML 2025: dense optimal below 500M |
| **INT4 quantization** | +0.065 BPB | Claude | 10x worse than int5→int6 gap |
| **Full depth recurrence** | +1.14 BPB gap | Both | 900x quantization error over 3 cycles |
| **MLA (Multi-Head Latent)** | 2x slower | Google | 83ms vs 43ms baseline, halves throughput |
| **LAWA weight averaging** | +0.023 BPB | Claude | Worse than SWA; EMA+XSA strictly superior |
| **SSM/Mamba hybrids** | +0.08 BPB gap | Claude | Hymba 1.1828, underperforms dense at this scale |
| **SwiGLU** | Worse than ReLU² | Both | Standard result at this scale |
| **Data selection/DSIR** | N/A | Claude | Training dataset fixed for all participants |
| **MAML Meta-TTT** | +0.085 BPB | Google | Network too small for meta-gradients |

---

## III. UPDATED TECHNIQUE PRIORITY (Post-Synthesis)

### Tier 0: Implement NOW (local, before GPU grant)

**1. LeakyReLU² (replace ReLU²)**
- Swap: `relu(x).square()` → `leaky_relu(x, 0.5).square()`
- Impact: Prevents dead neurons, smoother gradient flow
- Risk: Zero. Direct activation swap.
- Both teams confirm superiority at this scale.

**2. Differential Attention (deep layers)**
- Replace standard attention in layers 5-10 with Diff Attention
- Partition Q,K into two groups, use difference of softmax maps
- Halve head count (8→4 diff heads, each with 2 sub-heads)
- Synergizes with XSA (different failure modes: self-bias vs noise)
- Claude: -0.005-0.015 BPB. Google: "ICLR 2025 Oral, 35% efficiency gain"

**3. Learned Non-Uniform Quantization**
- 64-level lookup table per layer via k-means on weight distribution
- Storage overhead: ~2.75 KB total (negligible)
- Claude: -0.005-0.012 BPB. Compatible with GPTQ pipeline.

### Tier 1: Implement on GPU (first H100 runs)

**4. FP8 Training**
- Convert 2D weight matmuls to `torch.float8_e4m3fn`
- Keep master weights in BF16, embeddings/norms untouched
- Expected: +600-1000 extra training steps (20-40% more)
- Both teams: highest confidence, highest impact among untried

**5. Legal Score-First TTT**
- LoRA rank-8 adaptation during evaluation
- Score tokens FIRST under inference_mode, THEN backprop on evaluated tokens
- Reset LoRA at document boundaries (BOS tokens)
- Cosine LR schedule with per-layer grouping (3x multiplier vs flat)
- Google: pushed to 1.1164 BPB (pending, legal)

**6. Partial Weight Sharing + LoRA Deltas**
- 8 unique transformer blocks → 14 effective layers
- Per-layer LoRA rank 4-8 (~1-2% overhead per shared invocation)
- Cycle(Rev) ordering: {1,2,3,4,5,6,7,7,6,5,4,3,2,1}
- Claude: -0.008-0.02 BPB. Avoids full recurrence catastrophe.

**7. Parallel Muon + Parameter Banking**
- Fuse 66 separate weights into 4 contiguous 3D parameter banks
- Async reduce-scatter for gradient aggregation
- Claude/Google: -3.1% step time, +227 additional steps

### Tier 2: Polish and Optimize

**8. Custom Codebook + Huffman Encoding**
- Saves 21% vs int6+zstd
- Frees ~2MB for additional parameters

**9. GEPA-style Evolutionary Search**
- Use our agent framework as GEPA analog
- Architecture agent = mutation operator
- Orchestrator = Pareto selection
- Already partially implemented in our framework

### Tier 3: Experimental (separate branch)

**10. BitNet 1.58-bit QAT**
- ~45M params in 16MB (doubles capacity)
- Requires from-scratch ternary training
- High risk: may not converge in 10 min

---

## IV. REVISED PROJECTED BPB

### Conservative Path (no TTT):
```
Current SOTA (non-TTT):                          1.1175
- FP8 Training (+600 steps):                     -0.015  →  1.1025
- Differential Attention (deep layers):           -0.010  →  1.0925
- Partial Weight Sharing (8→14 layers):           -0.012  →  1.0805
- Learned Non-Uniform Quantization:               -0.008  →  1.0725
- Custom Codebook + Huffman:                      -0.005  →  1.0675
──────────────────────────────────────────────────────────
Conservative non-TTT target:                      ~1.07 BPB
```

### Aggressive Path (with legal TTT):
```
Conservative non-TTT:                             1.0675
- Legal Score-First TTT:                          -0.030  →  1.0375
──────────────────────────────────────────────────────────
Aggressive target with TTT:                       ~1.04 BPB
```

### Moonshot Path (BitNet):
```
BitNet 1.58-bit (45M params, 16-layer):           ~1.03-1.06 BPB
+ Legal TTT:                                      ~1.00-1.03 BPB
```

---

## V. IMPACT ON OUR FRAMEWORK

### Changes to Agent Roles:

| Agent | Previous Focus | Updated Focus |
|-------|---------------|---------------|
| Architecture | 11L + XSA + SmearGate | + Differential Attention + Partial Weight Sharing + LeakyReLU² |
| Hyperparameter | LR + warmdown + EMA | + FP8 training + Parallel Muon + Parameter Banking |
| Quantization | Int6 + GPTQ-lite | + Learned Non-Uniform Quant + Custom Codebook + Huffman |
| Evaluation | Multi-seed + artifact size | + Legal Score-First TTT + sliding window |
| Literature | General search | **DEPRIORITIZE** - research phase complete, execution phase begins |
| Orchestrator | Phase-adaptive | + GEPA-style evolutionary selection |

### New Agent Needed: TTT Agent
- Implements legal score-first TTT during evaluation
- Manages LoRA adapter lifecycle (init → adapt → reset at doc boundary)
- Tunes TTT hyperparams: chunk size, LoRA rank, LR schedule

---

## VI. VALIDATED vs INVALIDATED ASSUMPTIONS

### Our Previous Assumptions — VALIDATED:
- ✅ Architecture matters most (confirmed by both teams)
- ✅ EMA > SWA (quantified: +0.003 BPB)
- ✅ 11 layers + 3x MLP is correct depth/width
- ✅ XSA + Partial RoPE stack (confirmed)
- ✅ SmearGate + BigramHash are net positive
- ✅ Int6 >> Int8 for this budget

### Our Previous Assumptions — NEEDS UPDATE:
- ⚠️ ReLU² is optimal → **LeakyReLU² is better** (prevents dead neurons)
- ⚠️ Standard attention is fine → **Differential Attention is 35% more efficient**
- ⚠️ 11 fixed layers is the limit → **Weight sharing can give 14 effective layers**
- ⚠️ Uniform quantization is fine → **Non-uniform saves 0.005-0.012 BPB**
- ⚠️ We don't need TTT → **Legal TTT gives -0.03 BPB, mandatory for competition**
- ⚠️ Standard Muon is fine → **Parallel Muon + Banking gives +227 steps**

### Our Previous Assumptions — INVALIDATED:
- ❌ MoE could help → Apple proved dense is optimal below 500M
- ❌ SSM/Mamba could work → Hymba at 1.1828, 0.08 gap from frontier
- ❌ Weight sharing is safe → Full recurrence fails (900x quant error)
  - BUT: Partial sharing + LoRA DOES work (Claude research)
