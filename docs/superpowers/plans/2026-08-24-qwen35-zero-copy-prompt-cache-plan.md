# Qwen3.5 Zero-Copy Prompt Cache Implementation Plan

## Objective

Make a cached Qwen3.5 request with a reusable prefix of about 5,500 tokens and no more than 100 uncached tokens reach its first generated token within 5 seconds. Memory demand must stay below 28 GB on the 32 GB test machine.

The implementation follows the approved [design](../specs/2026-08-24-qwen35-zero-copy-prompt-cache-design.md). It changes only SwiftLM's prompt-cache policy, tests, and cache-hit logging. It does not modify the vendored Qwen3.5 kernels, add KV quantization, or change VLM routing.

## Files

- Modify `Sources/SwiftLM/Server.swift`.
- Modify `tests/SwiftLMTests/PromptCacheTests.swift`.
- Do not add dependencies or new production files.

## Task 1: Add the Qwen3.5 and single-slot eligibility gate

### 1. Write failing eligibility tests

Add table-driven coverage to `PromptCacheTests` for an internal helper named `qwen35ZeroCopyPromptCacheEnabled(modelType:parallel:)`:

- `qwen3_5` with one slot returns `true`.
- `qwen3.5` with one slot returns `true` after normalization.
- `qwen3_5` with two slots returns `false`.
- `qwen3_5_moe`, another model type, and `nil` return `false`.

Run:

```sh
swift test --filter PromptCacheTests/testQwen35ZeroCopyPromptCacheEligibility
```

Expected red result: the helper does not exist.

### 2. Implement the gate and carry it to `PromptCache`

In `Server.swift`:

1. Add `qwen35ZeroCopyPromptCacheEnabled(modelType:parallel:)`. Normalize case and replace `.` with `_`, then require an exact `qwen3_5` match and `parallel == 1`.
2. Move `parallelSlots` before `ServerConfig` construction.
3. Add `qwen35ZeroCopyPromptCache: Bool` to `ServerConfig`, calculated from `architecture.modelType` and `parallelSlots`.
4. Extend `PromptCache.init` with `qwen35ZeroCopy: Bool = false` and pass the config value at the server's single `PromptCache` construction site.

The default remains defensive copying, so existing unit tests and non-Qwen3.5 callers keep their current behavior.

### 3. Verify the gate

Run:

```sh
swift test --filter PromptCacheTests/testQwen35ZeroCopyPromptCacheEligibility
```

Expected green result: all model-type and parallelism cases pass.

## Task 2: Lock down snapshot isolation before removing copies

### 1. Add focused cache helpers

In `PromptCacheTests.swift`, add small helpers that build:

- a populated `MambaCache` with both slots set;
- a populated `RotatingKVCache` with configurable `maxSize`, token count, and value;
- one-token and multi-token key/value arrays for cache updates.

Use tiny float arrays. Do not load a model or add fixtures.

### 2. Add failing mode and isolation tests

Add an internal `PromptCacheRestoreMode` assertion point through `PromptCache.stats()`. The tests must cover:

1. The default `PromptCache()` reports `defensive-copy` after a retained hybrid restore.
2. An eligible Qwen3.5 cache containing `MambaCache`, `RotatingKVCache`, and `KVCacheSimple` reports `zero-copy`.
3. Replacing restored `MambaCache` slots and restoring again returns the original snapshot values.
4. Extending a restored `RotatingKVCache` by one token below `maxSize` leaves the retained snapshot's arrays, offset, and rotation metadata unchanged.
5. Extending a restored `RotatingKVCache` by multiple tokens leaves the retained snapshot unchanged.
6. Extending a restored exact-length `KVCacheSimple` leaves the retained snapshot unchanged.
7. A rotating layer already at `maxSize` uses `defensive-copy`; mutating its live ring cannot change the retained snapshot.
8. An `ArraysCache` or `CacheList` encountered under the Qwen3.5 policy uses `defensive-copy` before any destination layer is modified.
9. Repeated pinned and hybrid restores return the same retained values.

Run:

```sh
swift test --filter PromptCacheTests
```

Expected red result: the zero-copy mode and mode reporting do not exist.

## Task 3: Implement zero-copy save and restore

### 1. Represent the selected mode

In `PromptCache`:

1. Add an internal `PromptCacheRestoreMode: String, Sendable` with `zero-copy` and `defensive-copy` values.
2. Store the snapshot mode in `CachedState`.
3. Track the last successful restore mode in the existing `stats()` result so tests can verify policy selection without parsing stdout.

### 2. Validate the complete cache before choosing zero-copy

Add one private cache-eligibility function used by both save and restore. It returns `true` only when:

- the actor was constructed with `qwen35ZeroCopy == true`;
- the cache is non-empty;
- every layer is `MambaCache`, `RotatingKVCache`, or `KVCacheSimple`;
- each `RotatingKVCache` has a non-nil `maxSize` and `offset < maxSize`.

Before restoration, separately verify that the state, metadata, and offset arrays have the same layer count as the destination. Run both checks before assigning state to any live layer. A failed eligibility check selects `defensive-copy`; invalid saved-state counts follow the existing cache-miss behavior.

### 3. Remove physical copies on eligible saves

Update `snapshot(tokens:cache:)`:

1. Keep the existing exact-token slicing for `KVCacheSimple`.
2. For an eligible Qwen3.5 cache, retain the exact state arrays returned by each layer without calling `detached`.
3. For an ineligible cache, retain the current `array + 0` defensive copy for non-simple layers.
4. Keep `eval` over all stored arrays in both modes so no lazy graph remains attached to later model work.
5. Record the selected mode in `CachedState`.

This path is shared by rolling, hybrid, and pinned saves. Do not create separate implementations for them.

### 4. Remove physical copies on eligible restores

Update `restore(newTokens:into:allowPinned:)`:

1. Keep candidate selection, strict-prefix checks, trimming, and LRU refresh unchanged.
2. Select `zero-copy` only when the stored snapshot was zero-copy and the full destination cache passes the same eligibility checks.
3. Assign retained arrays directly for an eligible restore.
4. Use the current `detached` plus `eval` path for pinned or hybrid snapshots in defensive mode.
5. Keep the rolling dense behavior unchanged.
6. Record the selected restore mode only after all layer state has been restored successfully.

Do not change cache class implementations in `mlx-swift-lm`. The safety contract is enforced by the allowlist, the rotating-capacity check, and the mutation-isolation tests.

### 5. Run the focused suite

```sh
swift test --filter PromptCacheTests
```

Expected green result: all old prompt-cache tests and the new policy and isolation tests pass.

## Task 4: Add cache-hit diagnostics

In `PromptCache.restore`, start a wall-clock timer at function entry. Replace the existing hit line with one that includes:

- source: `pinned`, `hybrid`, or `rolling`;
- mode: `zero-copy` or `defensive-copy`;
- matched token count;
- remaining token count;
- restore duration in seconds with three decimal places.

Keep this as one log line. Do not add a metrics type or logging dependency.

Example shape:

```text
[SwiftLM] Prompt cache HIT [hybrid]: mode=zero-copy matched=5624 remaining=29 restore=0.004s
```

Run:

```sh
swift test --filter PromptCacheTests
```

Expected result: all prompt-cache tests still pass.

## Task 5: Verify the full change

### 1. Automated verification

Run from the SwiftLM repository root:

```sh
swift test --filter PromptCacheTests
swift test
swift build -c release
git diff --check
```

All commands must pass. Review the final diff and confirm it contains only the design amendment, `Server.swift`, `PromptCacheTests.swift`, and this plan.

### 2. Runtime benchmark

Start the same model and flags used in the reported regression:

```sh
../../run_model.sh --without-reasoning prism-ml/Ternary-Bonsai-27B-mlx-2bit:swiftlm
```

Use the same client to send:

1. an initial request that pins about 5,500 system-prefix tokens;
2. a follow-up with no more than 100 uncached tokens;
3. another follow-up that hits the rolling hybrid entry.

For each cached request, record the hit source, mode, matched tokens, remaining tokens, restore duration, time to first token, and `MEM_DEMAND` from the server log.

The change passes when both cached requests:

- report `mode=zero-copy`;
- reach the first generated token within 5 seconds;
- keep memory demand below 28 GB;
- produce the same deterministic answer as the defensive path.

If unit tests pass but the runtime target fails, stop and profile the remaining suffix prefill or view materialization. Do not add GDN kernels or KV quantization under this change.

## Commit

After automated and runtime verification pass, commit the implementation and tests together using the repository convention:

```sh
git add Sources/SwiftLM/Server.swift tests/SwiftLMTests/PromptCacheTests.swift docs/superpowers/specs/2026-08-24-qwen35-zero-copy-prompt-cache-design.md docs/superpowers/plans/2026-08-24-qwen35-zero-copy-prompt-cache-plan.md
git commit -m "fix: reduce Qwen3.5 cache-hit latency"
```
