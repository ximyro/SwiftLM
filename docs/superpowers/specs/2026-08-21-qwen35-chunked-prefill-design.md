# Qwen3.5 Chunked Prefill Design

## Goal

Make the existing OpenCode workflow complete a 14,945-token follow-up request without the SwiftLM process being killed, while keeping the current command and its 45,000-token context size unchanged.

The affected command uses the Qwen3.5 VLM implementation in text-only mode with `--prefill-size 2048` and `--pin-system-prompt`.

## Problem

The first 574-token request completes and pins its 522-token system prefix. The next 14,945-token request starts prefilling and the operating system kills SwiftLM under memory pressure.

The Qwen3.5 VLM implementation accepts `windowSize` in `prepare`, but ignores it. It passes the entire prompt to the language model as one lazy MLX graph and returns logits. Qwen3.5 includes recurrent Gated DeltaNet layers that build work across the full sequence, so a long prompt creates a much larger transient graph than the configured 2,048-token prefill size suggests.

The regular `LLMModel.prepare` path already avoids this problem. It evaluates prompt chunks, materializes the cache after each chunk, and returns the final chunk to `TokenIterator`. The Qwen3.5 VLM path does not use that behavior.

## Scope

This change will add chunked prefill to the text-only branch of `Qwen35.prepare`.

It will not:

- add `--hybrid-cache-entries`;
- change the pinned system-prefix cache or the rolling prompt cache;
- change context-size reservation or memory-limit policy;
- split image or video prompts;
- add a new cache abstraction.

Multi-entry hybrid caching can be considered after this fix is measured. It does not address the current unbounded prefill graph and would retain more memory in a process that is already under pressure.

## Design

### Text-only prefill

When an input has no image or video data, `Qwen35.prepare` will honor `windowSize`:

1. Reset the language model's position state, as the current text-only path does.
2. Treat a batch-one token tensor such as `[1, N]` as an `N`-token sequence for chunking. Slice the token dimension, never the batch dimension.
3. While more than one prefill chunk remains, run at most `prefillStepSize` tokens through the language model with the existing cache.
4. Call `eval(cache)` after each processed chunk. This materializes the hybrid KV and recurrent cache state and breaks the lazy graph before the next chunk is built.
5. Return the unprocessed final chunk as `.tokens`. `TokenIterator` will evaluate that chunk and sample from its final logits through the existing path.

Token order and cache offsets must be identical to evaluating the same prompt in one call. Each token is consumed exactly once.

### Prefill size

Use `windowSize` when it is greater than zero. If it is absent or non-positive, use the existing 512-token default. This prevents a non-progressing loop without introducing a new command-line policy.

### Multimodal inputs

Image and video inputs will keep the current single-call `.logits` path. Chunking those inputs also requires splitting embeddings, masks, and multimodal position IDs at matching boundaries. That is separate work and is not needed for the failing OpenCode request, which contains text only.

### Existing prompt caches

Pinned-prefix lookup and the single rolling cache remain unchanged. The server's exact-boundary path already supports a model returning `.tokens`: it evaluates the returned remainder before saving the prefix snapshot. Normal generation uses the same `TokenIterator` contract.

## Edge Cases and Invariants

- Prompts no longer than the prefill size go directly to `TokenIterator` without an eager chunk evaluation.
- A prompt exactly one token longer than the prefill size evaluates one chunk and returns one token.
- Batch-one `[1, N]` and sequence `[N]` token tensors preserve the same sequence order.
- The final chunk is never empty.
- Text masks, when present, remain aligned with the corresponding token slice.
- Cache evaluation happens only after a non-empty processed chunk.
- Image and video behavior is unchanged.

## Verification

Add a focused test around the chunk-boundary calculation using a 14,945-token, batch-one input. It must show that:

- every eager chunk contains at most 2,048 tokens;
- concatenating the eager chunks and final chunk reproduces the original sequence;
- no token is skipped or processed twice;
- absent and non-positive sizes fall back to 512 tokens.

Then run:

- the existing `PromptCacheTests` suite;
- a release build;
- the exact `run_model.sh` command with `--ctx-size 45000`, `--prefill-size 2048`, and `--pin-system-prompt`;
- the exact OpenCode command that produces the 14,945-token follow-up request.

The runtime acceptance criterion is that the follow-up request completes without `SIGKILL`. Record peak memory from the server telemetry so the result can be compared with the previous 32.0 GB memory-demand reading.

## Risk

The main correctness risk is slicing the wrong tensor axis or advancing the hybrid cache twice. The boundary test covers sequence partitioning, while the existing prompt-cache tests exercise cache reuse. The end-to-end OpenCode run is required because unit tests cannot reproduce the operating system's memory-pressure decision.
