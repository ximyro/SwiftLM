# Qwen3.5 Cached TTFT Implementation Plan

## Objective

Default exact Qwen3.5 text requests to the text model, retain explicit vision support, and verify a cached ~5,500-token prefix reaches first token within 5 seconds while memory demand stays below 28 GB.

## Task 1: Remove the failed experiment

Revert the uncommitted Qwen3.5 zero-copy prompt-cache policy, its tests, and the `MLX.copied` addition. Preserve unrelated existing work and retain only cache-hit diagnostics that help measure the benchmark without changing cache semantics.

Verify the existing prompt-cache tests pass.

## Task 2: Share the existing GatedDelta implementation

In the `mlx-swift-lm` submodule:

1. Add a focused numerical test comparing the shared GatedDelta path with the ops reference on deterministic Qwen3.5-compatible tensors and cached state.
2. Make the existing `MLXLLM` GatedDelta update function public.
3. Import `MLXLLM` from `MLXVLM/Models/Qwen35.swift`.
4. Delete the VLM's duplicate GatedDelta helper and ops functions so its existing call uses the shared implementation.

Do not add a new kernel, dependency, abstraction, or CLI flag.

## Task 3: Select the text implementation by default

In `Server.swift`, detect the exact `qwen3_5` architecture and suppress automatic VLM selection unless `--vision` is present. Add a focused test for normalization, the explicit override, and the `qwen3_5_moe` exclusion.

Do not change routing for other architectures.

## Task 4: Automated verification

Run:

```sh
swift test --filter GatedDelta
swift test --filter Qwen35
swift test --filter PromptCacheTests
swift test --filter SwiftLMTests
swift build -c release
git diff --check
```

The repository's known unrelated platform-test failures may be reported separately, but all focused tests and the release build must pass.

## Task 5: Runtime benchmark

Start:

```sh
../../run_model.sh --without-reasoning prism-ml/Ternary-Bonsai-27B-mlx-2bit:swiftlm
```

Send a deterministic initial request that pins approximately 5,500 tokens, followed by a request with no more than 100 new tokens. Record matched and remaining tokens, cache restore duration, cached TTFT, deterministic output, and memory demand.

Pass criteria:

- cached TTFT no more than 5 seconds;
- memory demand below 28 GB;
- output matches the reference response.

Measured result: 1.44-second cached TTFT with 5,490 reused tokens and 15.5 GB memory demand.

## Task 6: Commit

Commit the `mlx-swift-lm` fix using that repository's existing subject style, then commit the parent submodule pointer, diagnostics, tests, and revised docs with:

```text
fix: default Qwen3.5 to text-only serving
```
