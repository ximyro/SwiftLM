# Qwen3.5 Zero-Copy Prompt Cache

## Goal

Reduce cached time to first token for the Qwen3.5 text workload from 165–170 seconds to no more than 5 seconds. The benchmark uses a reusable prefix of about 5,500 tokens followed by at most 100 new tokens. Memory demand must remain below 28 GB on the 32 GB test machine.

This work addresses prompt-cache save and restore costs. It does not address Qwen3.5 prefill kernel performance.

## Observed problem

The affected command runs a Qwen3.5 VLM checkpoint with `--parallel 1`, `--pin-system-prompt`, `--hybrid-cache-entries 2`, and a 45,000-token context.

SwiftLM correctly restores the cached prefix and sends only the uncached suffix to the model. The delay occurs while preserving prompt-cache snapshots:

- retained arrays are detached with `array + 0`;
- `eval` materializes every detached array;
- a cached request can copy state during both restore and the next message-boundary save;
- memory demand reaches 31.5–31.8 GB, leaving the process vulnerable to unified-memory paging.

The symptom is independent of suffix length. Requests with 71 and 29 uncached tokens take approximately the same 165–170 seconds. This points to snapshot handling rather than suffix prefill.

## Scope

The fast path applies only when all of these conditions hold:

- the loaded architecture is Qwen3.5;
- the server has one inference slot (`--parallel 1`);
- prompt caching is otherwise eligible;
- every cache layer is `MambaCache`, `RotatingKVCache`, or `KVCacheSimple`;
- every `RotatingKVCache` layer is below `maxSize` at the snapshot boundary;
- the cached state and metadata pass the existing restore checks.

All other requests keep the current defensive-copy behavior.

This change will not:

- add a generic copy-on-write cache framework;
- change cache matching, eviction, pinning, or hybrid-entry ordering;
- change cache behavior for other model families;
- add KV quantization;
- change VLM routing;
- add or replace GatedDeltaNet kernels;
- add concurrent snapshot borrowing.

## Safety invariant

A retained zero-copy snapshot must remain unchanged after the live request advances its cache.

Qwen3.5 satisfies this invariant for the allowed cache types:

- `MambaCache` updates replace its array slots with newly computed arrays.
- `KVCacheSimple` snapshots expose arrays trimmed to the logical offset. The first append cannot fit in that exact-length state, so `update` allocates an expanded backing array before writing.
- `RotatingKVCache` snapshots also expose arrays trimmed to the logical offset. A multi-token append concatenates into new arrays. A one-token append first expands an exact-length buffer while it remains below `maxSize`, then writes into the expanded buffer.

The strict-prefix rule for non-trimmable state remains in place, so the live cache always receives at least one new token before decoding. A rotating cache at `maxSize` is ineligible because a one-token suffix can write directly into its ring. If a future cache implementation does not satisfy these mechanics, the eligibility gate rejects it and uses defensive copying.

## Design

### Eligibility

Resolve architecture eligibility when the model is loaded and carry it in `ServerConfig`. Qwen3.5 eligibility is disabled when more than one inference slot is configured.

Before each zero-copy save or restore, validate the actual cache array. Every layer must be one of the allowed cache types, and rotating layers must still have room to expand. Validation happens before any live cache object is modified. A failed check selects the existing defensive path.

### Snapshot policy

Prompt-cache save, pinned save, and restore receive an internal zero-copy policy. The default is disabled.

For an eligible Qwen3.5 cache:

1. Read state and metadata at the exact prompt or message boundary.
2. Keep exact-length array views without applying `array + 0`.
3. Evaluate the state arrays so the snapshot contains materialized values rather than a lazy graph tied to later work.
4. Retain the snapshot under the existing pinned or hybrid-cache rules.

The live cache and retained snapshot may initially refer to the same immutable arrays. The first model update replaces or expands the live state according to the safety invariant above.

On restore, assign the retained state to fresh cache objects without detaching or evaluating a physical copy. Token slicing and suffix generation continue through the existing path.

### Defensive fallback

The defensive path remains the source of truth for unsupported configurations. Zero-copy handling must fall back before restoration when:

- architecture or parallelism is ineligible;
- any cache layer has an unexpected type;
- state shape, metadata, or offset validation fails;
- the existing strict-prefix or trimming checks reject the candidate.

Fallback is silent from the API caller's perspective. The request continues with the existing defensive-copy or cache-miss behavior.

### Diagnostics

Cache-hit logs will include:

- source (`pinned`, `hybrid`, or `rolling`);
- restore mode (`zero-copy` or `defensive-copy`);
- matched and remaining token counts;
- restore duration.

Existing request telemetry supplies total time to first token and memory demand. No separate metrics subsystem is needed.

## Verification

### Automated tests

Add focused prompt-cache tests for:

1. Existing defensive-copy behavior, including direct mutation of a restored state.
2. Qwen3.5 zero-copy `MambaCache` restoration followed by model-style slot replacement, then a second restore that returns the original snapshot.
3. Qwen3.5 zero-copy `RotatingKVCache` restoration followed by one-token extension, then a second restore that returns unchanged keys, values, offset, and rotation metadata.
4. The same rotating-cache isolation check with a multi-token extension.
5. `KVCacheSimple` isolation after extension from an exact-length snapshot.
6. Defensive fallback for a `RotatingKVCache` whose offset has reached `maxSize`.
7. Rejection of an unexpected cache type before live state is modified.
8. Repeated pinned and hybrid zero-copy restores producing identical state.
9. Eligibility disabled when parallelism is greater than one.

Run the existing prompt-cache test suite and a release build after the focused tests pass.

### Runtime benchmark

Use the same Qwen3.5 checkpoint and server arguments that produced the regression. Send a deterministic sequence consisting of:

1. An initial request that pins a system prefix of about 5,500 tokens.
2. A follow-up request with no more than 100 uncached tokens.
3. A second follow-up that restores a rolling hybrid snapshot.

Record the cache source, restore mode, matched tokens, restore duration, time to first token, and memory demand for each request.

The change passes when:

- both cached requests use `zero-copy` mode;
- each cached request reaches its first generated token within 5 seconds;
- memory demand stays below 28 GB;
- deterministic output matches the defensive-copy path;
- no existing prompt-cache test regresses;
- the release build succeeds.

## Risks

The primary risk is a cache implementation changing from replacement or expansion to in-place mutation while remaining on the allowlist. The mutation-isolation tests lock down the update behavior relied upon by this design.

The fast path may still miss the 5-second target if materializing exact-length views or processing the uncached suffix dominates after copies are removed. In that case, retain this change only if it measurably reduces cache restore time and memory demand, then profile the remaining work before expanding scope.
