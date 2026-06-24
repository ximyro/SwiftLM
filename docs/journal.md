# 📖 Journal: Fixing Gemma-4 Inference in Aegis-AI (SwiftLM)

**Date**: April 3, 2026
**Goal**: Resolve startup crashes for `mlx-community/gemma-4-26b-a4b-it-4bit` on Apple Silicon.

## 2026-06-24 — Fix Gemma 4 tool prompt deadlock

**Milestone:** SwiftLM Gemma 4 compatibility
**Status:** done

- Fixed Gemma 4 tool schema normalization so nested OpenAI/opencode-style tool schemas no longer trigger `upper filter requires string` in the model chat template.
- Added guarded chat slot cleanup so pre-generation failures after `semaphore.wait()` do not leave `parallel=1` servers stuck with `requests_active=1`.
- Verified with targeted Swift tests and manual `mlx-community/gemma-4-12B-it-6bit` non-streaming, tool-bearing, and streaming requests.

## 🏔️ The Journey

### 1. Identifying the Root Cause
The first major obstacle was tracking down why the server abruptly crashed with `Mismatched parameter model.layers.0.mlp.down_proj.weight`. 
* **The Investigation**: We reviewed MLX Swift logs and dived into the implementation of `Gemma4Model`. What was expected to be a straight 4-bit dimension packing was inexplicably arriving with 8-bit dimensions for specific sub-layers.
* **The Culprit**: **Mixed-Precision Quantization**. While the top-level `config.json` indicated 4-bit packaging overall, the creators left critical projections like `mlp.down_proj` in 8-bit to preserve model perplexity score capability. Our monolithic `sanitize` algorithm ignored those differences.

### 2. 🧱 The Blockers
The journey had a lot of detours due to syntax errors and library protections:
- **`internal` Protection Masking**: When updating the `Gemma4` wrapper to process `SwitchLinear` bits exactly like `Linear`, we hit a wall where Swift's visibility protocols blocked us. `SwitchLinear` held its `weight` property under internal protection, forcing us to detour into `SwitchLayers.swift` and export its dimension protocols publicly.
- **Syntax Quirks in MLX Arrays**: A heavy blocker hit us near the end. `Shapes (1,21) and (1,0) cannot be broadcast.` Why? Because Swift handles array slices completely uniquely compared to Python. Passing standard `[0..., kth...]` mapped physically missing sequence axes, yielding `(1,0)`. Tracing MLX's index protocol and discovering the true explicit form of `[0..., 0..., kth...]` for 3D tensors was critical.
- **Redeclaration Conflicts**: While iterating the `Gemma4RouterProj`, structural redeclarations crashed the compile chain. 

### 3. 🛠️ The Architecture Refactor
Rather than forcing constraints, we engineered the logic to be universally adaptable:
We implemented the `determineBits()` math (`32 * original / packed`) locally on every single linear operation initialization within MLX's weight mapping tree. Each individual layer dynamically detects its own native quantization state natively extracted directly from its Safetensor footprint checkpoint, completely bypassing unreliable config generalizations.

### 4. 🚀 The End Result
After several rounds of debugging and recompilation via `swift build -c release`, we finally produced a robust `SwiftLM` backend. 
When we bound it to `127.0.0.1:5430` natively loading all variables entirely on the Metal GPU in 13 tokens/second execution speeds without zero shape errors, we achieved the final breakthrough!

A smooth copy of our `SwiftLM` binary into the Aegis `b21/macos-arm64/` environment officially crowned the feature fully shipped and complete.

---

## 🔮 Phase 2: Making It Actually _Think_ (April 3-4, 2026)

### 5. TurboKV Crash on 512-Dim Global Heads
The server now loaded the model... then immediately crashed with `turbo_encode_k requires 128 or 256 but got 512`. 
* **The Fix**: Added a strict whitelist guard in `KVCache.swift` — only 128 and 256 dimensions pass through to the Metal kernel; everything else gracefully falls back to fp16.

### 6. The Token Collapse — Output Was All Dashes
After the TurboKV fix, the model loaded and ran at 627 t/s prefill speed. But every generated token was `236772` — a dash character. *Infinite dashes*.

We audited the entire forward pass against the [Python mlx-vlm reference](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/language.py) and found **7 critical differences**:

| # | Bug | Python Reference | Swift Had |
|---|-----|-----------------|-----------|
| 1 | MLP activation | `gelu_approx` | `silu` (gate * sigmoid(gate)) |
| 2 | Attention scale | `1.0` (norms handle it) | `1/sqrt(queryPreAttnScalar)` ≈ 0.0625 |
| 3 | Global RoPE | `ProportionalRoPE` (custom class) | Standard `RoPE` on wrong dims |
| 4 | Router topK | `argpartition(-scores, kth=topK-1)[:topK]` | `argpartition(probs, kth=N-topK)[kth:]` |
| 5 | Softcapping | `tanh(logits/30)*30` | Disabled (believed to saturate) |
| 6 | Embedding scale | `h * sqrt(hidden_size)` ≈ 53x | Missing entirely |

### 7. The Breakthrough: **Missing Embedding Scale**
After fixing bugs 1-5, the output changed from dashes to... *all spaces*. The logit distribution looked numerically reasonable (max~35, min~-46) but always peaked at the same whitespace token.

This was the **smoking gun**: the logits had the right *shape* but the wrong *magnitude*. The Gemma architecture family (since Gemma 1) scales embedding outputs by `sqrt(hidden_size)`. For Gemma 4 with `hidden_size=2816`, that's a **53x multiplier**. Without it, every activation in the entire 32-layer transformer was 53x too small, causing the model to "think in whispers" and default to whitespace.

**One line of code:**
```swift
h = h * MLXArray(Float(config.hiddenSize).squareRoot())
```

### 8. 🎉 First Words
```
"What is 2+2?" → "2 + 2 equals 4."
"Write a haiku about the ocean." → "Blue waves kiss the shore,
                                     Endless tides rise and fall low,
                                     Deep salt mystery."
```

The model speaks. Coherently. Creatively. With proper EOS stopping.

### 9. Key Lesson: ProportionalRoPE
The most complex fix was implementing `Gemma4ProportionalRoPE` — a custom positional encoding class that:
- Computes frequencies relative to the **full** head_dim (512)
- But only rotates 25% of the dimensions (`partial_rotary_factor=0.25` → 128 dims)
- Uses the HuggingFace `rotate_half` convention: split head into left/right halves, take rotated_dims//2 from each half
- The standard `RoPE` class couldn't handle this — it either rotates ALL dims or rotates the FIRST N dims. The Python reference has an entirely separate `ProportionalRoPE` class.

### 📋 Files Changed
- `Gemma4.swift` — All forward pass fixes, ProportionalRoPE, embed_scale
- `KVCache.swift` — TurboKV head_dim guard
- `Evaluate.swift` — Debug print cleanup

### 🚀 Deployment
Binary deployed to `~/.aegis-ai/mlx_binaries/b21/macos-arm64/` as both `SwiftLM` and `mlx-server`.

---

## 🌟 Appendix: Optimization and The Future of SSD MoE Streaming

### The Hacker News Discussion
**vessenes**
> I like this idea on expert streaming. I've been poking around fairly thoroughly at the same idea - can we fix a set of experts? when can we fix them? How long is the top-k selection "good" for in terms of number of forward passes?
> One thing I've turned up in smaller models and I'm sort of winding my way toward verifying in larger ones is that if you train the MoE model from scratch with this kind of knockout / subset of experts baked in, then you get significantly better loss outcomes. In small models, it's actually better than training an MOE without conditioning on a reduced set of experts per pass.
> Anyway, pretty cool. There's some Pareto-optimal curve based on memory bandwidth, amount of GPU / unified RAM and inference compute times for streaming stuff in.

**aegis_camera** (reply)
This is an incredible insight, and what you are seeing with the "expert knockout" training outcome aligns perfectly with some of the most cutting-edge research happening right now around efficient MoE architectures and memory-constrained inference.

If we look at the entire pipeline—from how we design the training objective to how we execute the binary on macOS with SSD streaming—there is a very clear path to optimizing this.

Here is my end-to-end thought process on how this entire pipeline fits together, and why your observation about training and temporal locality is the key to unlocking the Pareto frontier for consumer hardware.

#### 1. The Training Implication (Expert Knockout & Regularization) 
Your observation that training an MoE from scratch with a reduced/fixed set of experts per pass yields better loss is profound. Standard token-level routing often suffers from "expert collapse" (where a few experts do all the work) or requires heavy auxiliary loss penalties just to keep the routing balanced.
By aggressively enforcing "expert knockout" or fixing the subset of experts over a sequence/chunk during training:
- **You are forcing generalization:** It acts like a macro-level Dropout or DropConnect. The model can't over-rely on a specific "super-expert" because it knows that expert might not be available in the current pass. 
- **Redundant Knowledge Distribution:** The network learns to distribute critical semantic representations across multiple experts, making the model far more robust. 

#### 2. Temporal Locality (How long is Top-K "good" for?) 
In standard auto-regressive generation, a model does not wildly shift its semantic domain or syntactic structure from token $N$ to token $N+1$. There is massive temporal locality in expert activation.
If you train the model with block-level routing (forcing the routing decision to be fixed for $M$ tokens), the answer to your question—how long is the top-k selection good for?—changes from "1 token" to "10 to 50 tokens."
This completely changes the math for SSD streaming. Instead of paying the NVMe latency tax on every single forward pass, you amortize the SSD read across an entire semantic chunk.

#### 3. The Pareto-Optimal Curve (Bandwidth vs. RAM vs. Compute) 
On Apple Silicon, the variables are very stark:
- **GPU Compute:** Effectively instant for these matrix sizes. 
- **Unified Memory Bandwidth:** ~400 GB/s to 800 GB/s (plenty fast). 
- **SSD Bandwidth:** ~5 GB/s to 10 GB/s (the massive bottleneck). 

The Pareto frontier comes down to Hit Rate vs. Fetch Cost. If you can keep a small LRU (Least Recently Used) cache of experts in Unified RAM, and the model has high temporal locality, your SSD fetch rate drops to near zero for long stretches of generation. You only hit the SSD when the semantic context shifts (e.g., moving from writing Python code to explaining it in English).

#### 4. The Ideal "Full Pipeline" Architecture 
If we were to build the ultimate MoE pipeline optimizing for SSD streaming on consumer hardware, here is how the whole thing looks:

**Phase 1: Pre-training (Temporal Block MoE)**
Train the model to route at the chunk level (e.g., every 16 or 32 tokens) rather than the token level. Apply your "expert knockout" during training to ensure the model maintains performance even if its preferred expert is forcibly swapped out. 

**Phase 2: Lookahead Routing (The Pre-fetcher)**
During inference, because the MoE layer sits deep within the transformer block (after attention), you can compute the routing logits early. Better yet, train a tiny, ultra-fast auxiliary MLP (a "Routing Predictor") that runs on the CPU. It looks at the current context and predicts which experts will be needed 3-4 tokens in the future. 

**Phase 3: Asynchronous Zero-Copy DMA (The MLX/Metal Layer)**
While the GPU is crunching the Attention layers for the current token... The CPU triggers an async `pread()` directly pointing to the unified memory command buffer. The NVMe controller DMA's the upcoming MoE weights straight from the SSD into RAM. Crucially: Because of Apple's Unified Memory architecture, you bypass the CPU RAM -> VRAM copy entirely. The GPU just reads the pointer once the DMA completes. 

**Phase 4: LRU Eviction & Quantization**
You maintain a strict budget of RAM (e.g., 2 GB for active experts). The experts themselves are aggressively quantized (e.g., 4-bit or even lower using something like TurboQuant). When the context shifts and a new expert is swapped in, the oldest expert is simply discarded (since it's read-only, there's no write-back penalty). 

#### Summary 
What you are poking at is exactly the future of local LLMs. Models are getting too big for VRAM, but SSDs are getting fast enough to bridge the gap if the model architecture cooperates. By changing the training objective to favor temporal blocks and expert knockout, you are effectively "hardware-aware" training the model to be friendly to the SSD PCIe lane.
It completely shifts the bottleneck from the hardware (bus speed) to the algorithm (routing predictability).

---

## 🔬 Phase 3: Extreme Context Profiling & The Prompt Cache Discovery (April 5, 2026)

### 10. Building the Profiling Framework

With Gemma 4-26B stable and generating, we needed to answer the real deployment question: *How does this model behave at extreme context lengths across different memory configurations?* We built `scripts/profiling/profile_runner.py` — an automated profiling framework that:
- Iterates through 4 configurations: Dense/Vanilla, SSD Stream, TurboQuant, SSD+TurboQuant
- Tests across 3 context depths: 512, 40K, and 100K tokens
- Captures both **Active RAM** (OS physical footprint via `mach_task_basic_info`) and **GPU Memory Allocated** (Apple GPU driver allocation via `ioreg AGXAccelerator`)

### 11. The `ioreg` Breakthrough

The initial profiling used only `phys_footprint` — the OS physical memory metric. But at 100K context, both Dense/Vanilla (49.3 GB) and TurboQuant (49.3 GB) showed identical numbers. This made no sense — TurboQuant was clearly compressing the KV cache.

The problem: `phys_footprint` is **capped by available physical RAM**. On a 64 GB machine, it tops out at ~49 GB regardless of actual demand. We needed to see the *total GPU allocation* including memory swapped to SSD.

Following the same pattern used by the Aegis-AI `HardwareDetector`, we queried Apple's `AGXAccelerator` GPU driver via `ioreg` for the `"Alloc system memory"` counter. This metric **CAN exceed physical RAM** — revealing the true memory demand:

| Configuration | Active RAM (capped) | GPU Alloc (true demand) |
|---|---|---|
| Dense/Vanilla @ 40K | 49.4 GB | **52.6 GB** |
| TurboQuant @ 40K | 32.4 GB | **35.0 GB** |

The 52.6 GB vs 35.0 GB difference was invisible in the OS metric but clearly visible via `ioreg`.

### 12. 🐛 The Prompt Cache Bug

At 100K context, even with the `ioreg` metric, TurboQuant (52.5 GB) and Dense/Vanilla (52.1 GB) were nearly identical. Tracing through the code revealed the root cause:

1. During prefill, the full 100K fp16 KV cache is built (~37 GB)
2. After the first generation token, TurboQuant compresses it to ~3 GB of polar buffers ✅
3. But then `onPrefillDone` fires → the prompt cache calls `cache.state`
4. The `state` getter **decodes ALL compressed polar buffers back to full fp16** to create a restorable snapshot
5. The `eval()` call materializes this decoded copy — a fresh **~37 GB allocation**
6. Net result: compression savings completely negated

The key insight: for Dense/Vanilla, `cache.state` returns **views** (zero-copy references) of existing buffers. For TurboQuant, it creates **new arrays** via `turboDecodeK/V` — an O(N) memory allocation the size of the entire context.

### 13. 🔧 The Fix

One targeted change: skip prompt cache save when TurboQuant has actively compressed data.

**Results at 100K context (SSD + TurboQuant):**

| Metric | Before Fix | After Fix |
|---|---|---|
| GPU Memory Allocated | 52.2 GB | **33.3 GB** (-36%) |
| Active RAM | 49.1 GB | **29.6 GB** (-40%) |

**29.6 GB Active RAM for a 26B model at 100K tokens.** This fits in a 32 GB Mac Studio — previously required 64 GB.

### 📋 Files Changed
- `Sources/SwiftLM/MemoryUtils.swift` — Added GPU active memory and total demand metrics
- `Sources/SwiftLM/Server.swift` — OS_RAM + MEM_DEMAND + GPU_MEM logging at prefill and post-generation; prompt cache TurboQuant guard
- `scripts/profiling/profile_runner.py` — Full profiling framework with `ioreg` GPU allocation tracking

### 🎯 Key Lesson: Measure What Matters
The OS `phys_footprint` metric is what Activity Monitor shows — but it lies by omission. It's capped by physical RAM and doesn't reveal how much memory the GPU driver has actually allocated (including SSD-swapped pages). For memory-constrained deployment, the `ioreg AGXAccelerator "Alloc system memory"` counter is the ground truth.
