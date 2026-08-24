# Qwen3.5 Cached TTFT

## Goal

Make a cached Qwen3.5 request with a reusable prefix of about 5,500 tokens and no more than 100 uncached tokens reach its first generated token within 5 seconds. Memory demand must stay below 28 GB on the 32 GB test machine.

## Diagnosis

The reported SwiftLM and Rapid-MLX runs use the same checkpoint but different model implementations:

- SwiftLM auto-detects `vision_config` and loads `MLXVLM.Qwen35`.
- Rapid-MLX is started with `--no-mllm` and loads the text model.

SwiftLM restores the cached prefix correctly and spends approximately zero time in cache lookup. Sharing `MLXLLM`'s fused GatedDelta kernel with the VLM improves cold prefill, but the VLM path still misses the cached-TTFT target. Loading the text implementation, as Rapid-MLX does, reduces cached TTFT to 1.44 seconds and memory demand to 15.5 GB.

The zero-copy prompt-cache experiment did not improve TTFT or memory demand and is not part of the final change.

## Design

For the exact `qwen3_5` architecture, default text-only server requests to `MLXLLM`. Keep `--vision` as the explicit way to load `MLXVLM`, so image requests remain available. Do not apply this routing rule to `qwen3_5_moe` or any other architecture.

Also make the existing `MLXLLM` GatedDelta update function available to `MLXVLM`. Delete the duplicate helpers from `MLXVLM/Models/Qwen35.swift`, import `MLXLLM`, and let the Qwen3.5 VLM use the shared implementation when vision is requested.

Do not port Rapid-MLX's separate blocked-sequential kernel or add another CLI flag. The existing `--vision` flag already expresses the non-default mode.

Revert the unproven zero-copy cache policy and its `mlx-swift` copy primitive. Keep the existing prompt-cache behavior unchanged.

## Safety

The routing gate normalizes case and dots but requires an exact `qwen3_5` match. Explicit `--vision` always wins. The shared function retains its existing GPU/CPU dispatch and masked fallback behavior; unsupported small head dimensions use the operations fallback.

Before accepting the change, compare the shared kernel with the old ops recurrence on small deterministic inputs, including a cached state. Existing Qwen3.5 VLM tests and SwiftLM prompt-cache tests must continue to pass.

## Verification

Run focused numerical and Qwen3.5 tests, the SwiftLM test suite, a release build, and `git diff --check`.

The deterministic two-request benchmark passed with:

- 5,490 cached tokens from a 5,511-token prompt;
- cached TTFT of 1.44 seconds;
- memory demand of 15.5 GB;
- the expected deterministic response.
