# How We Built an Autonomous AI Research Lab to Win OpenAI's Parameter Golf

*A story about cramming intelligence into 16 megabytes*

---

## The Challenge That Hooked Us

Imagine you're told to build the smartest possible brain, but it has to fit inside a matchbox. That's essentially what OpenAI's Parameter Golf challenge asks: train the best language model you can, but the entire thing — code and model weights combined — must fit in **16 megabytes**. Oh, and you only get **10 minutes** to train it on 8 of the world's most powerful GPUs.

When we first read the challenge description, the constraints seemed almost contradictory. Modern language models like GPT-4 are hundreds of gigabytes. We had to build something a thousand times smaller that could still *understand language* well enough to compress text efficiently. The metric is called **BPB** — bits per byte — and it measures how well your model predicts the next character in a stream of web text. Lower is better. The current record stands at **1.1228 BPB**, meaning the model compresses each byte of text down to about 1.12 bits. For context, raw English text is about 8 bits per byte, so the winning model is achieving nearly 7:1 compression through understanding alone.

We decided to approach this differently. Instead of manually tweaking hyperparameters one at a time, we would build a **team of AI research agents** that could explore the solution space in parallel — and we'd pressure-test our approach by running it through a simulated panel of expert reviewers before spending any GPU credits.

---

## Chapter 1: Understanding the Battlefield

### What Exactly Is BPB?

Before diving into our approach, let's understand the metric we're optimizing. **Bits per byte (BPB)** measures compression quality. When a language model reads text, it assigns probabilities to what comes next. If the model predicts the next token with high confidence, it takes fewer bits to encode that prediction. If the model is surprised, it takes more bits.

The formula works like this: the model computes a **cross-entropy loss** (how surprised it is, measured in "nats" — natural log units), converts that to bits (divide by ln(2)), then normalizes by the actual byte count of the text. Because the challenge uses a 1,024-token vocabulary built with **SentencePiece** (a tokenizer that breaks text into subword pieces), different tokens represent different numbers of bytes. A token might be a single letter (1 byte) or a common word like "the" (3 bytes). BPB accounts for this, making comparisons fair regardless of vocabulary choice.

### The 16MB Budget

The artifact constraint is precise: **16,000,000 bytes** (decimal, not 16 mebibytes). This includes:

- **Your code** (`train_gpt.py`): typically 50-70 KB
- **Your compressed model weights**: the remaining ~15.9 MB

The model weights are compressed through a pipeline:
1. **Quantization**: Convert 32-bit floating point weights to 8-bit (or 6-bit) integers
2. **Compression**: Apply zlib or zstd to squeeze out remaining redundancy

This means a model with 17 million parameters (at 1 byte each after int8 quantization) compresses to roughly 6.3 MB — leaving over 9 MB of budget unused. The winning submissions use 26-27 million parameters with int6 quantization to fill nearly the entire 16 MB budget.

### The 10-Minute Clock

Training happens on **8x NVIDIA H100 SXM GPUs** — each with 80GB of HBM3 memory and 3.35 TB/s bandwidth. These are $30,000 GPUs designed for AI training. With `torch.compile` (which fuses operations into optimized GPU kernels), the winning submissions complete about 7,100 training steps in 600 seconds — roughly 85ms per step. Each step processes about 786,000 tokens of web text.

Without `torch.compile` (like on our consumer RTX 3080), the same step takes 950ms — over 10x slower. This is why the challenge requires specific hardware: the competition is about what you can learn in 10 minutes at maximum throughput.

---

## Chapter 2: Learning from the Masters

We began by deeply studying two existing frameworks for autonomous ML research:

### Karpathy's autoresearch

Andrej Karpathy (former Tesla AI director, OpenAI founding member) created [autoresearch](https://github.com/karpathy/autoresearch) — an elegant system where an AI agent runs overnight, autonomously conducting experiments. The design is beautifully simple:

1. The agent reads a `program.md` file containing research instructions
2. It modifies `train.py` with one change
3. It runs the training (5-minute budget)
4. It checks the result (val_bpb)
5. If improved: **keep** (git commit). If worse: **revert** (git reset)
6. Repeat forever

The genius is in the constraints: single-file mutation, metric-driven decisions, git-based history. Every experiment is a commit, every failure is visible. The agent runs ~12 experiments per hour, ~100 per overnight session.

**Strength**: Simplicity. One agent, one GPU, one metric, clear feedback loop.
**Weakness**: Sequential. Only one experiment at a time. No parallelism.

### ByteDance's DeerFlow

[DeerFlow](https://github.com/bytedance/deer-flow) takes the opposite approach — a full "agent operating system" with:

- A **lead agent** that decomposes complex tasks
- **Subagents** that execute in parallel (up to 3 concurrent)
- **Middleware chain** (12 ordered stages) handling context, memory, guardrails
- **Sandboxed execution** per thread (isolated filesystems)
- **Skills** (pluggable workflow templates)
- **Memory persistence** across sessions

It's built on LangGraph and runs as a web service with FastAPI.

**Strength**: Scale. Parallel agents, structured decomposition, production-ready.
**Weakness**: Complexity. 6,000 lines of Python. Not specialized for ML training research.

### Our Hybrid: Taking the Best of Both

We built a framework that combines autoresearch's experiment loop with DeerFlow's parallelism:

- **From autoresearch**: The modify-train-measure-keep/revert loop. Metric-driven decisions. Git-based tracking. "Never stop" autonomous operation.
- **From DeerFlow**: Multiple specialized agents working simultaneously. Structured role decomposition. Memory persistence across sessions.
- **Our innovation**: **Git worktrees** for parallel experiments. Each agent gets its own isolated copy of the repository on its own branch. The architecture agent can test 12-layer models while the hyperparameter agent tunes the current 11-layer config — simultaneously, with no conflicts.

---

## Chapter 3: The Agent Team

We created six specialized agents, each with a focused domain of expertise:

### The Orchestrator
The strategic coordinator. It decides *what* to try based on where we are in the optimization journey. Early on (BPB > 1.15), it prioritizes architecture changes — the biggest gains. As quality improves (BPB 1.13-1.15), it shifts focus to training optimization. Near the frontier (BPB < 1.13), it focuses on quantization and compression to squeeze the last drops.

### The Architecture Agent
The model designer. It explores structural changes: how many layers, how wide, what kind of attention, what embeddings. This agent reads the current `train_gpt.py`, proposes one focused change, runs the experiment, and keeps or reverts based on BPB.

### The Hyperparameter Agent
The training optimizer. It systematically sweeps learning rates, batch sizes, learning rate schedules, EMA decay rates, and gradient clipping thresholds. One variable at a time, then interaction effects.

### The Quantization Agent
The compression specialist. It optimizes how we squeeze 27 million floating-point parameters into 16 MB: int6 vs int8 quantization, GPTQ-lite clip percentile search, zstd vs zlib compression, late quantization-aware training.

### The Evaluation Agent
The quality assurance team. It runs multi-seed experiments (at least 3 different random seeds), computes statistical significance, validates artifact sizes, and prepares submission materials.

### The Literature Agent
The research scout. It searches for novel techniques from papers and repositories that could give us an edge: mixture of experts at small scale, differential attention, learned quantization scales.

---

## Chapter 4: The Expert Panel

Before spending any GPU credits, we did something unusual: we created a **simulated panel of five expert reviewers** — modeled after the kind of researchers who would evaluate submissions at a top ML conference — and ran our approach through their critique.

### Dr. Sarah Chen (Scaling Laws)
Her verdict was blunt: *"You're using 40% of your budget. This is literally called Parameter Golf — every unused byte is wasted capacity."*

She was right. Our baseline used a 9-layer model that compressed to just 6.35 MB — leaving 9.65 MB of the 16 MB budget empty. That's like entering a weight class boxing match at half the allowed weight. The SOTA submissions fill 97% of the budget.

### Dr. Marcus Thompson (Systems)
He pointed out that our local RTX 3080 results were *"meaningless as a quality signal."* Without `torch.compile`, we were 18x slower than the target hardware. Our 118 training steps were a rounding error compared to the 7,100 steps SOTA achieves on H100s.

His advice: *"For local development, focus on code correctness, not BPB numbers."*

### Dr. Priya Patel (Quantization)
She identified three missing wins: int6 quantization (saves 25% on weight storage), GPTQ-lite clip percentile search (free 0.0006 BPB improvement), and late QAT (quantization-aware training in the final 4%).

The key insight: *"Int6 isn't just about quality — it's about fitting more parameters in the budget. An int6 model at 15.5 MB holds ~26M params; int8 at the same size holds only ~20M."*

### Dr. James Park (Architecture)
He catalogued six missing innovations from SOTA, ordered by impact:

1. **Deeper models (11-12 layers)**: More layers capture richer representations
2. **XSA (Exclusive Self Attention)**: Forces deep layers to attend to context, not self
3. **Partial RoPE**: Apply rotary position encodings to only 16 of 64 dimensions, letting the remaining 48 learn position-invariant patterns
4. **SmearGate + BigramHash**: Inject local context at the embedding layer
5. **LN Scale Factor**: `1/sqrt(layer_idx + 1)` dampens deeper layers for stability
6. **3x MLP expansion**: Larger feedforward hidden dimensions

### Dr. Elena Rodriguez (Competition Strategy)
She approved our framework but challenged our execution order: *"You're optimizing in the wrong order. Architecture accounts for 0.08 BPB of your deficit. Hyperparameters account for 0.01. Do architecture first."*

She also identified the single biggest "free" improvement: **sliding window evaluation** with stride=64 gives -0.033 BPB with zero model changes — just a better evaluation strategy.

---

## Chapter 5: Building the Improved Model

Armed with the panel's feedback, we implemented all recommended changes in priority order.

### The Architecture Upgrade

We changed from the 9-layer baseline to an **11-layer transformer with 3x MLP expansion**. Here's what that means in plain English:

A **transformer** processes text through a series of layers. Each layer has two main components:

- **Self-attention**: Each word looks at every other word to understand context. "The cat sat on the mat" — when processing "sat," the attention mechanism figures out that "cat" is the subject and "mat" is the destination. We use **GQA (Grouped Query Attention)** where 8 query heads share 4 key/value heads, reducing memory while maintaining quality.

- **MLP (Multi-Layer Perceptron)**: A feedforward network that transforms each token's representation independently. The "expansion factor" determines how wide this hidden layer is — 2x means it's twice the model dimension (512 → 1024 hidden units), 3x means three times (512 → 1536). Larger MLPs can represent more complex transformations.

Going from 9 to 11 layers with 3x MLP increased our parameter count from **17 million to 26.8 million** — a 58% increase. This fills more of the 16 MB budget and gives the model more capacity to learn.

### U-Net Skip Connections

Our model already had these, but they become more powerful with 11 layers. The architecture splits into an "encoder" (first 5 layers) and "decoder" (last 6 layers). **Skip connections** pass information directly from encoder layers to corresponding decoder layers — the same idea that made U-Net revolutionary in image segmentation. This gives decoder layers direct access to earlier representations without them getting diluted through many intervening layers.

Each skip connection has a learned weight (a vector of 512 values) that controls how much of the encoder signal to mix in. During training, the model learns the optimal blend.

### Exclusive Self Attention (XSA)

Applied to the **last 4 layers**, XSA modifies standard attention by subtracting each token's "self-value" from the attention output. In mathematical terms:

```
y = attention_output - dot(attention_output, normalize(v)) * normalize(v)
```

This forces the deeper layers to focus on information from *other* tokens in the sequence, not from the token itself. The intuition: shallow layers should build rich token representations (where self-information is useful), but deep layers should focus on contextual relationships (where self-information is redundant).

We implement this efficiently using GQA-aware reshaping instead of expensive `repeat_interleave` operations — a trick from the SOTA submissions that adds only ~2ms overhead per step.

### Partial RoPE (16 of 64 Dimensions)

**RoPE (Rotary Position Embeddings)** is how the model knows word order. It rotates query and key vectors based on their position in the sequence, so the model can distinguish "the cat ate the fish" from "the fish ate the cat."

Standard RoPE applies rotation to all head dimensions. **Partial RoPE** applies it to only the first 16 out of 64 dimensions. The remaining 48 dimensions are free to learn **position-invariant patterns** — things that are true regardless of where they appear in the sequence, like "a verb often follows a subject" regardless of position.

This costs zero extra parameters, zero extra compute, and improves BPB by ~0.002. The insight: not all attention heads need positional information, and forcing it on all of them limits what they can learn.

### SmearGate

A tiny module (~512 parameters) that blends each token's embedding with its predecessor:

```
output = sigmoid(gate) * current_token + (1 - sigmoid(gate)) * previous_token
```

The gate is initialized at ~0.95 (sigmoid(3.0)), meaning the model starts almost as an identity function and gradually learns how much bigram context to inject. This gives the model local context information right at the embedding layer, before any attention computation.

### BigramHash Embedding

A hash table with 2,048 buckets that captures **token pair features**. Given the current and previous token, we compute:

```
bucket = (previous_token * 92821 + current_token) % 2048
```

Each bucket has a 128-dimensional learned embedding, projected to the model dimension (512) through a linear layer. This adds ~524K parameters but provides the model with explicit bigram statistics that would otherwise take many attention layers to learn implicitly.

Together, SmearGate + BigramHash contribute ~0.003 BPB improvement by injecting local context at the cheapest possible layer.

### LN Scale Factor

A simple but effective stabilization trick: multiply each layer's contribution by `1/sqrt(layer_index + 1)`. Layer 0 contributes at full strength. Layer 10 contributes at 1/3.3 strength.

This prevents deeper layers from dominating the residual stream and creates a natural "warmup" where the model builds representations gradually. Zero extra parameters, just a multiplication constant per layer.

### EMA (Exponential Moving Average)

Instead of using the final training weights directly, we maintain a **running average** that updates every step:

```
ema_weights = 0.997 * ema_weights + 0.003 * current_weights
```

This smooths out the noisy trajectory of SGD training. The decay of 0.997 means the EMA has a half-life of ~231 steps — it "remembers" the last few hundred steps of training, smoothing out oscillations while still tracking the overall learning trajectory. SOTA submissions show this consistently saves 0.0006 BPB.

### Orthogonal Initialization + muP Scaling

We replaced the default random initialization with **orthogonal initialization** for all large weight matrices. Orthogonal matrices have the property that they preserve vector norms — gradients neither explode nor vanish at initialization, leading to faster, more stable convergence.

For output projection layers, we apply **muP (Maximal Update Parameterization) scaling**: multiply by `1/sqrt(2 * num_layers)`. This comes from the muP theory of hyperparameter transfer — it ensures that the magnitude of updates is consistent regardless of model depth, letting hyperparameters transfer across different model sizes.

---

## Chapter 6: The Local Validation

We ran two experiments on our consumer RTX 3080 (10GB VRAM, no torch.compile):

### Baseline (v1): The Starting Point

```
Model:          9 layers, 512-dim, 2x MLP, 17M parameters
Batch:          131K tokens, seq_len=512
Training:       3 minutes, 118 steps
Result:         val_bpb = 2.8187
Artifact:       6.35 MB (int8+zlib)
VRAM:           6,688 MiB
```

The loss curve showed rapid early learning (6.93 → 4.76 in 118 steps) but the model was far from convergence. More importantly, only 40% of the 16MB budget was utilized.

### Improved (v2): After Panel Review

```
Model:          11 layers, 512-dim, 3x MLP, 26.8M parameters
                + XSA + Partial RoPE + SmearGate + BigramHash
                + LN Scale + EMA + Gradient Clipping
Batch:          65K tokens, seq_len=256
Training:       2 minutes, 127 steps
Result:         val_bpb = 3.1543 (pre-quant) / 3.4667 (post-quant int8)
Artifact:       8.95 MB (int8+zlib)
VRAM:           3,996 MiB
```

Wait — the BPB is *worse*? Yes, but this is expected and the panel predicted it:

1. **Smaller batch and shorter sequences**: We used seq_len=256 (vs 512) and half the batch to fit the larger model. Shorter context means the model has less information per prediction.

2. **Fewer effective tokens**: Only 127 steps × 65K tokens = 8.3M tokens seen, vs 118 × 131K = 15.4M for the baseline. The larger model needs more data.

3. **Local numbers don't matter**: As Dr. Thompson said, without torch.compile on H100s, these numbers are meaningless for competition. What matters is that **all innovations compile, train without errors, and the loss curve is healthy**.

The critical wins from v2:
- **26.8M params** instead of 17M — filling the budget properly
- **Only 4GB VRAM** — massive headroom for larger batches on H100
- **EMA working**: "Using EMA weights for serialization" confirmed in logs
- **All 8 innovations functional**: No crashes, no NaNs, smooth loss curves

### The Quantization Gap Problem

Post-quantization BPB jumped from 3.15 to 3.47 — a 0.31 degradation. This is unacceptably high. The reason: our baseline int8 quantization clips at a fixed 99.99984 percentile and applies per-row scaling, which is too coarse for a 26.8M parameter model with diverse weight distributions.

SOTA solves this three ways:
1. **Int6 quantization** with per-row GPTQ-lite search across 5 clip percentiles
2. **Late QAT**: Training the last 4% of steps with fake quantization so the model adapts
3. **Mixed precision**: Int6 for large matrices, int8 for embeddings, FP32 for control parameters

With these techniques, SOTA achieves a quantization gap of only 0.0001-0.0015 BPB. This is a P1 priority for our H100 runs.

---

## Chapter 7: What We Learned

### The Architecture Tax

Every byte matters. Going from 9 to 11 layers with 3x MLP nearly doubled our parameter count (17M → 26.8M) but also nearly doubled our artifact size (6.35 → 8.95 MB). With int6 quantization (which we haven't implemented yet), the same model would compress to ~14-15 MB — fitting nicely in the 16 MB budget with room for code.

The lesson: **architecture and quantization are inseparable**. You can't design the model without knowing how it will be compressed, and you can't optimize compression without knowing the model structure.

### The Compound Effect

The SOTA improvement from baseline (1.2243) to record (1.1228) is 0.1015 BPB — about 8.3%. This comes from **15+ individual improvements**, each contributing 0.0001-0.0335 BPB. No single change is transformative. The competition is won by systematically stacking small gains:

| Improvement | BPB Gain |
|-------------|----------|
| Sliding window eval | -0.0335 |
| Deeper model (10-11L) | -0.0050 |
| Int6 + larger MLP (3x) | -0.0100 |
| SmearGate + BigramHash | -0.0050 |
| XSA + Partial RoPE + LN Scale | -0.0048 |
| EMA | -0.0006 |
| GPTQ-lite + warmdown tuning | -0.0006 |
| Longer sequences (2048) | -0.0186 |
| FP16 embeddings | -0.0020 |

### The Expert Panel Was Worth It

Running our approach through simulated reviewers before training saved us from several mistakes:
- We would have wasted GPU credits training a 17M model when we should have been training a 26.8M model
- We would have focused on hyperparameter tuning when architecture was the bottleneck
- We would have skipped sliding window eval, missing the single biggest free improvement

---

## What Comes Next

With GPU compute (via OpenAI's grant program), we plan to:

1. **Run the full pipeline on 8xH100**: torch.compile + full batch + 80 data shards + 10 minutes
2. **Implement int6 quantization with GPTQ-lite**: Close the quantization gap
3. **Add sliding window evaluation**: Free -0.033 BPB
4. **Systematic ablation**: Test each innovation in isolation to measure true contribution
5. **Multi-seed validation**: 3+ runs for statistical significance
6. **Submit a PR to the leaderboard**: Our target is BPB < 1.12

The framework is built. The agents are ready. The architecture is validated. Now we need the compute to unleash them.

---

*This post documents our work on the OpenAI Parameter Golf challenge. Code and framework available at [github.com/canivel/openai-challenges](https://github.com/canivel/openai-challenges).*
