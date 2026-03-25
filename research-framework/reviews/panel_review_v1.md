# Expert Panel Review — Parameter Golf Submission v1

**Date:** 2026-03-24
**Submission:** Baseline train_gpt_local.py (9L/512D/2x MLP, int8 quantization)
**Result:** val_bpb 2.82 (118 steps, 3 min, RTX 3080) | Artifact: 6.35 MB
**SOTA Target:** 1.1228 BPB

---

## Panel Members

1. **Dr. Sarah Chen** — Scaling Laws & Efficiency (Neural Scaling Laws group)
2. **Dr. Marcus Thompson** — Systems Optimization (Training infrastructure)
3. **Dr. Priya Patel** — Quantization & Compression (Post-training optimization)
4. **Dr. James Park** — Architecture Design (Transformer variants)
5. **Dr. Elena Rodriguez** — Competition Strategy (ML competition veteran)

---

## Dr. Sarah Chen — Scaling Laws & Efficiency

### Assessment: CRITICAL ISSUES

**1. Massive parameter budget waste (severity: HIGH)**
Your artifact is **6.35MB out of 16MB** — you're using only 40% of your budget. This is the single biggest issue. The challenge is literally called "Parameter Golf" — every unused byte is wasted capacity. The SOTA uses 15.5-15.9MB. You're leaving ~9.5MB of model capacity on the table.

**Recommendation:** Increase model size immediately. An 11-layer model with 3x MLP expansion uses ~26.8M params and compresses to ~15.5MB with int6. This is a **~58% parameter increase** over your 17M params.

**2. Compute-optimal scaling is wrong for this challenge (severity: MEDIUM)**
Classical Chinchilla scaling suggests balanced compute between model size and data. But this challenge has a **fixed parameter budget** and **unlimited data** (80 shards = 16B tokens). The optimal strategy is: **maximize parameters within the 16MB budget, then train as long as possible.** You should be using every byte.

**3. Sequence length is suboptimal (severity: MEDIUM)**
You train with seq_len=512. SOTA uses 2048. Longer sequences provide more context per forward pass, which is especially important for language modeling where long-range dependencies matter. On 8xH100 with compile, seq_len=2048 is feasible and gives ~0.019 BPB improvement.

**Action items:**
- [ ] Increase to 11 layers
- [ ] Increase MLP mult to 2.6-3.0x
- [ ] Increase seq_len to 1024 (local) or 2048 (H100)
- [ ] Target artifact size > 15MB

---

## Dr. Marcus Thompson — Systems Optimization

### Assessment: TRAINING PIPELINE IS 10x SLOWER THAN NEEDED

**1. No torch.compile is a showstopper for benchmarking (severity: HIGH)**
Without torch.compile, you're measuring a fundamentally different system. On H100 with compile, you'd get ~85ms/step. You're getting 1525ms/step. That's **18x slower**. You completed 118 steps; SOTA completes ~7,100 steps. The model barely started learning.

**This means your val_bpb of 2.82 is meaningless as a quality signal.** The baseline should reach ~1.22 BPB with full training. You're comparing apples to oranges.

**Recommendation:** For local development, focus on code correctness, not BPB numbers. Use short runs (50-100 steps) to verify no crashes, then deploy to H100 for real benchmarking.

**2. Warmup steps are wasteful locally (severity: LOW)**
You set WARMUP_STEPS=2, which is fine locally. But the original 20-step warmup is designed to prime torch.compile code paths. Without compile, warmup just wastes time.

**3. Validation is the bottleneck (severity: HIGH)**
Your validation took ~10 minutes for 62M tokens without compile. SOTA uses sliding window eval with compiled `forward_logits` (75-88 seconds). For local testing, reduce validation dramatically:
- Use only first 1M tokens of validation
- Or skip validation entirely and just check train_loss trends

**4. Grad accumulation overhead (severity: MEDIUM)**
With world_size=1, you have grad_accum_steps=8. Each micro-batch is only 16K tokens (131072/8). The overhead of 8 forward/backward passes per optimizer step is significant without compile. Consider: for local testing, manually set grad_accum to 1-2 with a smaller total batch.

**Action items:**
- [ ] Add a `--local-dev` flag that: skips most validation, reduces grad_accum, shortens warmup
- [ ] Don't benchmark BPB locally — just verify code correctness
- [ ] Implement sliding window eval for H100 runs

---

## Dr. Priya Patel — Quantization & Compression

### Assessment: USING BASELINE QUANTIZATION, MISSING 3 MAJOR WINS

**1. Still using int8 when SOTA uses int6 (severity: HIGH)**
Your code uses the baseline `quantize_state_dict_int8` with per-row int8 quantization. SOTA uses **int6 for MLP and attention weights** (per-row, [-31,31] range). This saves ~25% on weight storage, allowing a larger model within 16MB.

The int6 quantization gap is small (~0.001-0.002 BPB) with proper GPTQ-lite optimization, but the **size savings are huge**: an int6 model at 15.5MB holds ~26M params, while int8 at 15.5MB holds only ~20M params.

**2. No GPTQ-lite clip percentile search (severity: MEDIUM)**
The baseline uses a fixed 99.99984 percentile clip. SOTA searches across [0.999, 0.9995, 0.9999, 0.99999, 1.0] per row and picks the one minimizing reconstruction MSE. This is **free** (zero training cost, post-training only) and gives -0.0006 BPB.

**3. No late QAT (severity: LOW)**
Quantization-aware training in the final 4% of training helps the model adapt to quantization noise. Uses STE (straight-through estimator) for gradient flow. Small improvement (~0.0001 BPB) but cumulative.

**4. Using zlib, not zstd (severity: LOW)**
zstd level 22 gives ~5% better compression than zlib level 9. On a 15MB payload, that's ~750KB savings — enough to fit more parameters.

**Recommendation:** Implement int6 quantization with GPTQ-lite search. This is the highest-ROI change for artifact size.

**Action items:**
- [ ] Implement int6 per-row quantization ([-31, 31] range, 6-bit values)
- [ ] Add GPTQ-lite per-row clip percentile search
- [ ] Add late QAT with STE in final 4% of training
- [ ] Switch to zstd compression if available

---

## Dr. James Park — Architecture Design

### Assessment: BASELINE ARCHITECTURE IS 6 INNOVATIONS BEHIND SOTA

Your model is the unmodified 9-layer baseline. Every single SOTA innovation is missing. Ordered by impact:

**1. U-Net skip connections — ALREADY PRESENT but only 9 layers (severity: HIGH)**
Your code already has U-Net skips (encoder/decoder split with skip_weights). Good. But with 9 layers (4 encoder + 5 decoder), the skip connections are limited. With 11 layers (5+6), you get more skip connections and more representation depth.

**2. No Exclusive Self Attention / XSA (severity: HIGH, -0.002 BPB)**
XSA on the last 3-4 layers forces deeper layers to attend to context rather than self-reference. Implementation:
```python
# In attention forward, after computing y:
if self.xsa:
    y_grouped = y.reshape(B, T, Hkv, group_size, D)
    vn = F.normalize(v, dim=-1).unsqueeze(-2)
    y = (y_grouped - (y_grouped * vn).sum(-1, keepdim=True) * vn).reshape(B, T, H, D)
```
~2ms overhead per step. Worth it.

**3. No Partial RoPE (severity: MEDIUM, -0.002 BPB)**
Apply rotary embeddings to only 16 of 64 head dimensions. The remaining 48 dims learn position-invariant patterns. Zero parameters, zero compute overhead.

**4. No SmearGate + BigramHash (severity: MEDIUM, -0.003 BPB)**
SmearGate blends each token with its predecessor (~512 params). BigramHash captures token-pair features via a hash table (~524K params). Together they inject local context at the embedding layer.

**5. No LN Scale Factor (severity: LOW, -0.002 BPB)**
`1/sqrt(layer_idx + 1)` dampens deeper layers. Stabilizes training, especially in 11+ layer models. Zero parameters.

**6. No Value Embedding (severity: LOW, -0.001 BPB)**
Shared token identity injected into attention values at specific deep layers.

**7. MLP expansion is 2x, SOTA uses 3x (severity: HIGH)**
With int6 quantization freeing space, SOTA uses 3x MLP expansion (hidden=1536). This is the simplest high-impact change after adding layers.

**Action items (priority order):**
- [ ] Increase to 11 layers with 3x MLP (biggest single improvement)
- [ ] Implement XSA on last 4 layers
- [ ] Implement Partial RoPE (16 dims)
- [ ] Implement SmearGate + BigramHash
- [ ] Add LN Scale Factor
- [ ] Add Value Embedding at layers 9,10

---

## Dr. Elena Rodriguez — Competition Strategy

### Assessment: GOOD FRAMEWORK, WRONG EXECUTION ORDER

**1. You're optimizing in the wrong order (severity: HIGH)**
Your local run tested the unmodified baseline. That's fine for pipeline validation, but you should not iterate on hyperparameters or quantization until the architecture is competitive. The architecture gap accounts for **~0.08 BPB** of your deficit to SOTA. Hyperparameters account for ~0.01. Quantization accounts for ~0.02.

**Do architecture first. Everything else follows.**

**2. Your 6.35MB artifact is actually great news (severity: POSITIVE)**
Most beginners try to fit too much. You have headroom. This means you can:
- Add 11 layers + 3x MLP and still fit
- Use int8 initially (simpler) while developing architecture
- Switch to int6 later for the final squeeze

**3. Sliding window eval is the single biggest "free" improvement (severity: HIGH)**
Your baseline uses standard chunked eval. SOTA uses sliding window with stride=64, which gives **-0.033 BPB improvement** just from better evaluation — no model changes needed. This is the lowest-hanging fruit.

**4. Your local testing strategy should be different (severity: MEDIUM)**
Don't chase BPB locally. Instead:
- Verify code compiles and runs without errors
- Check parameter counts and artifact sizes match expectations
- Test each architectural innovation in isolation (does it crash? does loss decrease?)
- Save the actual BPB benchmarking for H100 runs

**5. The grant submission is strong but needs a PR to link (severity: HIGH)**
The grant asks for "Link(s) to your PR submission." You don't have a PR to the OpenAI repo yet. You should:
1. Fork openai/parameter-golf
2. Create a branch with your modified train_gpt.py
3. Open a WIP PR (even with non-competitive results)
4. Link that PR in the grant application

**6. Competition timeline awareness (severity: MEDIUM)**
Deadline is April 30, 2026. That's 37 days away. With GPU compute, you can run ~300 experiments. Prioritize:
- Week 1: Architecture (11L + XSA + SmearGate + BigramHash)
- Week 2: Training optimization (EMA + warmdown + LR tuning)
- Week 3: Quantization (int6 + GPTQ-lite + zstd)
- Week 4: Multi-seed validation + submission polish
- Week 5: Buffer for novel ideas and final improvements

**Action items:**
- [ ] Fork openai/parameter-golf and create WIP PR immediately
- [ ] Implement sliding window eval (free -0.033 BPB)
- [ ] Focus all local development on architecture changes
- [ ] Don't benchmark BPB locally — only verify correctness
- [ ] Plan the 37-day timeline against GPU budget

---

## CONSENSUS PRIORITY RANKING

All panelists agree on this priority order:

### P0 — Do Immediately (before GPU grant)
1. **Increase to 11 layers + 3x MLP** — biggest architectural improvement
2. **Implement sliding window eval** — free -0.033 BPB, no model change
3. **Fork openai/parameter-golf and create WIP PR** — needed for grant link

### P1 — Do First on GPU
4. **Implement int6 quantization + GPTQ-lite** — enables larger model in 16MB
5. **Add XSA on last 4 layers** — proven -0.002 BPB
6. **Add Partial RoPE (16 dims)** — zero cost, proven -0.002 BPB
7. **Increase seq_len to 2048** — proven -0.019 BPB

### P2 — Do After P1 Results
8. **Add SmearGate + BigramHash** — proven -0.003 BPB
9. **Add LN Scale Factor** — zero cost stability improvement
10. **Tune EMA decay + warmdown** — proven -0.001 to -0.006 BPB
11. **Add late QAT** — small but cumulative

### P3 — Final Polish
12. **Multi-seed validation (3+ seeds)**
13. **Artifact size optimization (fill 16MB budget)**
14. **Switch to zstd compression**
15. **Prepare formal submission**

---

## EXPECTED IMPROVEMENT TRAJECTORY

| Stage | Changes | Expected BPB | Delta |
|-------|---------|-------------|-------|
| Baseline (full train) | 9L/2x/int8 | ~1.2243 | — |
| + Sliding window eval | stride=64 | ~1.1908 | -0.0335 |
| + 11L/3x MLP | Architecture | ~1.1502 | -0.0406 |
| + XSA + Partial RoPE | Attention | ~1.1457 | -0.0045 |
| + SmearGate + BigramHash | Embeddings | ~1.1407 | -0.0050 |
| + Int6 + GPTQ-lite | Quantization | ~1.1300 | -0.0107 |
| + EMA + warmdown | Training | ~1.1228 | -0.0072 |
| **Target** | **All combined** | **< 1.12** | **-0.10+** |
