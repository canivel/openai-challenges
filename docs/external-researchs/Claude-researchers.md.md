# Five novel techniques to push Parameter Golf below 1.10 BPB

**The most promising untried approaches for the OpenAI Parameter Golf competition combine throughput optimizations (FP8 training, fused kernels) that yield more training steps with architectural innovations (Differential Attention, partial weight sharing) that extract more quality per parameter.** Each technique targets a different bottleneck — compute throughput, parameter efficiency, quantization quality, or effective depth — and they stack multiplicatively. Based on deep analysis of the competition's 600+ PRs, the live AI commentary tracker, confirmed failures, and recent ML literature, these five techniques have the highest expected value among approaches not yet implemented on the leaderboard.

The current merged SOTA stands at **1.1228 BPB** (PR #414), with pending frontrunners at 1.1175 (non-TTT) and 1.1164 (legal TTT). The disputed multi-pass TTT entry sits at 1.0523 but faces legality challenges. The frontier technique stack is well-established: 11-layer transformer, XSA-4, EMA(0.997), VRL, LeakyReLU², BigramHash/TrigramHash, GPTQ, late QAT, and int5/int6 mixed precision with zstd-22 compression. Everything below targets gaps *beyond* this stack.

---

## 1. FP8 training doubles effective compute within the time budget

**Estimated improvement: 0.01–0.025 BPB | Confidence: High | Complexity: Medium**

H100 SXM GPUs deliver **2× TFLOPS in FP8 versus BF16** (1,979 vs 989 TFLOPS for dense operations). Switching matrix multiplications to FP8 during training would yield **20–40% more training steps** in the 10-minute window — equivalent to seeing hundreds of millions of additional tokens. The competition's own AI commentary (Issue #140) lists FP8 training as a Tier 1 untried technique with high expected value.

The primary risk is activation outliers corrupting gradient flow. Two recent papers solve this cleanly. **TWEO Loss** (arXiv:2511.23225, ICLR 2026) prevents outlier amplification during FP8 training with minimal overhead. **Rank-Aware Spectral Scaling** (arXiv:2602.18851) provides an alternative stabilization approach. The NanoGPT speedrun already uses FP8 matmul for the LM head with asymmetric rescaling and softcap logits, proving viability in this exact ecosystem.

Implementation requires converting linear layers to use `torch.float8_e4m3fn` for forward/backward passes while keeping master weights in BF16. The critical detail: only apply FP8 to the **2D weight matrices** (attention projections, MLP layers), not embeddings or layer norms. Combined with Muon's existing Newton-Schulz orthogonalization (which operates on the BF16 master weights), this should be a clean integration. One competition entry (PR #389) already achieved **125ms/step with custom Triton kernels** — FP8 would push this further, potentially to ~90–100ms/step, adding **~600–1,000 extra training steps**.

---

## 2. Differential Attention cancels noise for 35% better parameter efficiency

**Estimated improvement: 0.005–0.015 BPB | Confidence: Medium-High | Complexity: Medium-High**

The **Differential Transformer** (Ye et al., Microsoft/Tsinghua, arXiv:2410.05258) received an **ICLR 2025 Oral** — the highest-impact attention mechanism paper of the past two years. It partitions Q and K into two groups, computes two separate softmax attention maps, and uses their *difference* as the final attention score. This noise-cancellation mechanism eliminates attention to irrelevant tokens, analogous to differential signaling in electronics.

The headline result: **Diff Transformer requires only ~65% of model size or training tokens to match standard Transformer performance.** At the 25M-parameter scale where every parameter matters, this 35% efficiency gain translates directly to BPB improvement. The competition's live AI commentary lists it with an estimated **0.005–0.015 BPB improvement** but notes it as "High complexity" — no participant has implemented it yet.

The implementation adds negligible parameters: just a learnable scalar λ per head. To match FLOPs, halve the number of heads (e.g., 8→4 differential heads, each containing two sub-heads). The mechanism is FlashAttention-compatible. The key architectural decision is whether to apply Differential Attention to all 11 layers or only the deeper layers (where attention noise is most problematic). Starting with the **last 6–8 layers** while keeping standard attention for early layers is the safer approach — this mirrors the XSA strategy of partial application that proved optimal in the competition.

A critical synergy exists with the existing XSA technique. XSA removes self-value bias via orthogonal projection; Differential Attention removes irrelevant-token noise via attention map subtraction. These target **different failure modes** and should stack. The combination of Diff Attention + XSA on the last 4–6 layers could yield the full 0.015 BPB improvement.

---

## 3. Partial weight sharing with LoRA deltas enables 14 effective layers

**Estimated improvement: 0.008–0.02 BPB | Confidence: Medium | Complexity: Medium**

The competition definitively proved that **full depth recurrence fails** — quantization error amplifies ~900× over 3 cycles (PR #363), rendering shared-weight loops unusable. But a more nuanced approach avoids this catastrophe entirely: **partial weight sharing with per-layer low-rank deltas**.

The idea, listed explicitly in the competition's AI commentary as an untried Tier 2 technique: store 7–8 unique transformer blocks, then create 14 effective layers by sharing pairs of adjacent blocks, each differentiated by a small **LoRA adapter** (rank 4–8, adding ~1–2% parameter overhead per shared invocation). This is grounded in the **Relaxed Recursive Transformers** paper (Bae et al., Google DeepMind, 2024), which showed that recursive Gemma-1B with per-iteration LoRA modules outperformed vanilla Gemma-1B trained from scratch by up to **13.5% absolute accuracy**.

The key insight avoiding the depth-recurrence disaster: each "virtual" layer has its own unique LoRA parameters, so quantization of the shared base creates identical error in paired layers, but the LoRA deltas provide unique corrections. The error doesn't compound because the LoRA adapters are quantized independently. Research on parameter sharing patterns (Takase & Kiyono, 2023) shows that **Cycle(Rev) ordering** — palindromic sharing like {1,2,3,4,5,6,7,7,6,5,4,3,2,1} — consistently outperforms sequential or simple cyclic patterns for autoregressive language models.

The math works well for the 16MB budget: 8 unique layers at current parameter density ≈ ~18M params for layers + ~2M for LoRA adapters across 14 layers + embeddings. At int5 quantization with zstd-22, this fits comfortably. The extra 3 effective layers (14 vs 11) should deliver meaningful BPB improvement — the jump from 10 to 11 layers was one of the largest single gains in competition history.

---

## 4. Learned non-uniform quantization extracts more from every bit

**Estimated improvement: 0.005–0.012 BPB | Confidence: Medium-High | Complexity: Low-Medium**

Current competition entries use **uniform quantization** (int5/int6 with evenly spaced levels). But weight distributions in neural networks are highly non-uniform — approximately Gaussian with heavy tails. **Learned non-uniform quantization** assigns more quantization levels to high-density regions of the weight distribution, dramatically reducing reconstruction error at the same bit width.

SqueezeLLM (Kim et al., 2024) demonstrated the magnitude of this gap: at 3-bit precision on LLaMA-7B, non-uniform quantization achieved **PPL 7.75** versus uniform quantization's **PPL 28.26** on C4 — a 3.6× improvement from the same number of bits. While this gap narrows at higher bit widths, even at 5–6 bits the improvement is meaningful.

The implementation requires a **64-level lookup table per layer** (for int6) that maps quantization indices to optimal reconstruction values, learned via k-means clustering on each layer's weight distribution. Storage overhead is minimal: 64 values × 2 bytes × ~22 weight matrices = **~2.75 KB total** — negligible against the 16MB budget. This is fully compatible with late QAT: during the QAT phase, use the non-uniform codebook as the quantization grid instead of uniform steps, and train with straight-through estimator gradients through the nearest-codebook-entry operation.

The competition's AI commentary lists this as an untried technique and notes it's compatible with the existing GPTQ pipeline. A more aggressive variant uses **CALDERA-style decomposition** (NeurIPS 2024): decompose each weight matrix as W ≈ Q + LR, where Q is low-bit quantized and L,R are low-rank factors capturing the residual that quantization misses. This consistently outperforms pure quantization at the same total storage. For the existing int5 MLP / int6 attention setup, adding rank-4 residual factors per layer would cost ~0.5MB but recover most of the quantization-induced BPB loss.

An even more promising direction is **custom codebook + Huffman encoding**. One competition entry already demonstrated this saves **21% versus int6+zstd** (compressing to 14.12MB versus ~18MB). The freed space enables fitting ~3–4M additional parameters — enough for an extra transformer layer or wider MLPs.

---

## 5. BitNet 1.58-bit QAT enables a radically larger model

**Estimated improvement: 0.01–0.03 BPB | Confidence: Medium (high risk, high reward) | Complexity: High**

The most radical untested approach: **train with ternary (1.58-bit) quantization-aware training from scratch**, enabling a ~40–50M parameter model to fit within 16MB. This roughly doubles the model's effective capacity compared to the current ~25M parameter int5/int6 approach.

Microsoft's **BitNet b1.58** restricts weights to {-1, 0, 1} during training, achieving competitive performance with full-precision models at ≥3B parameters. More critically for this competition, the **"BitNet b1.58 Reloaded"** paper (Nielsen et al., 2024) specifically studied small models from **12M to 48M parameters** and found that ternary QAT acts as a beneficial regularizer at small scale, with models achieving "close to state-of-the-art performance" and even exceeding SOTA on some tasks due to reduced overfitting. The **AbsMedian** variant improves robustness for small models specifically.

The storage arithmetic is compelling:

- **Current approach**: ~25M params × 5–6 bits = 15.6–18.75MB raw → ~14–16MB compressed
- **BitNet approach**: ~45M params × 1.58 bits = ~8.9MB raw → ~7–8MB compressed (ternary weights with ~42% zeros compress extremely well with zstd)
- **Headroom**: ~8–9MB freed for embeddings (keep at 8-bit), code, and metadata

The **Sparse-BitNet** finding (2025) adds another lever: 1.58-bit models naturally produce ~42% zero weights, enabling additional N:M semi-structured sparsity with less degradation than FP16 models. Ternary quantization and sparsity are synergistic.

The risks are real. BitNet requires native QAT from the first training step — you cannot post-train quantize to ternary. The 10-minute training budget must be sufficient for the larger model to converge, though 8×H100 should handle 45M ternary parameters faster than 25M BF16 parameters (ternary matmuls can be accelerated). The BPB-per-parameter may be worse at ternary precision, but the question is whether **doubling parameter count** more than compensates. At 45M parameters, you could run a 16-layer transformer with 640-dimensional hidden states — a substantial capacity increase that scaling laws suggest should reduce BPB by **0.03–0.05** before accounting for quantization degradation.

---

## How these techniques compose into a winning stack

The five techniques target orthogonal bottlenecks and should combine cleanly. A practical implementation roadmap, ordered by risk and dependency:

**Phase 1 (low risk, implement first):** FP8 training + learned non-uniform quantization + custom Huffman coding. These are incremental improvements to existing infrastructure. Combined estimated gain: **0.015–0.035 BPB**.

**Phase 2 (medium risk, requires architecture changes):** Differential Attention on layers 5–11 + partial weight sharing (8 unique → 14 effective layers with LoRA). These require more engineering but have strong theoretical backing. Combined estimated gain: **0.013–0.035 BPB**.

**Phase 3 (high risk, alternative path):** BitNet 1.58-bit QAT as a separate experimental branch. If successful, it could obsolete Phase 1's quantization work but potentially deliver the largest single improvement. Estimated gain: **0.01–0.03 BPB**.

The conservative path (Phase 1 + Phase 2) projects a total improvement of **0.025–0.065 BPB** beyond current SOTA, potentially pushing below **1.06 BPB** without TTT. With legal score-first TTT adding its typical **0.02–0.04 BPB** bonus, the combined stack could reach **1.02–1.04 BPB** — well beyond the current frontier.

## What definitively does not work

The competition has produced equally valuable negative results that save implementation time:

- **MoE** at any configuration below 500M params (PR #480: −0.06 to −0.08 BPB, Apple ICML 2025 Best Paper confirms dense is optimal below 500M)
- **INT4 quantization** (+0.065 BPB, 10× worse than int5→int6 gap)
- **Full depth recurrence** (900× quantization error amplification over 3 cycles)
- **MLA** (Multi-Head Latent Attention — 83ms/step versus 43ms baseline, halving throughput)
- **LAWA** weight averaging (PR #201: 0.023 BPB worse than SWA; EMA with XSA is strictly superior)
- **SSM/Mamba hybrids** (Hymba at 1.1828 BPB, ~0.08 gap from frontier — SSMs underperform dense transformers at this tiny scale)
- **SwiGLU** on standard architecture (worse than ReLU²)
- **Data selection/DSIR** (training dataset is fixed for all participants)

## Conclusion

The competition's low-hanging fruit has been picked. The remaining BPB gains require either **more effective compute** (FP8 training, fused kernels) or **more effective parameters** (Differential Attention, weight sharing, better quantization). The single highest expected-value move is combining FP8 training with the GEPA architecture base, legal TTT, and VRL — this stack alone could reach ~1.08 BPB. Adding Differential Attention and partial weight sharing to this base pushes into uncharted territory below 1.06. The BitNet path represents the most radical departure: if ternary QAT works at this scale, the ability to fit nearly double the parameters changes the entire optimization landscape. The competition runs through April 30, 2026, leaving five weeks for these innovations to mature — plenty of time for any of these techniques to redefine the frontier.