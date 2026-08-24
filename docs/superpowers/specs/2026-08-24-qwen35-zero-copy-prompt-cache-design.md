# Qwen3.5 Cached TTFT

## Goal

Make a cached Qwen3.5 request with a reusable prefix of about 5,500 tokens and no more than 100 uncached tokens reach its first generated token within 5 seconds. Memory demand must stay below 28 GB on the 32 GB test machine.

## Diagnosis

The reported SwiftLM and Rapid-MLX runs use the same checkpoint but different model implementations:

- SwiftLM auto-detects `vision_config` and loads `MLXVLM.Qwen35`.
- Rapid-MLX is started with `--no-mllm` and loads the text model.

SwiftLM restores the cached prefix correctly and spends approximately zero time in cache lookup. The remaining suffix is slow because `MLXVLM.Qwen35` contains a private, token-by-token GatedDelta implementation built from ordinary MLX operations. `MLXLLM.Qwen35TextModel` already routes the same recurrence through the project's Metal GatedDelta kernel. A 21-token cached suffix therefore creates a large lazy graph in the VLM path instead of using the existing fused kernel.

The zero-copy prompt-cache experiment did not improve TTFT or memory demand and is not part of the final change.

## Design

Make the existing `MLXLLM` GatedDelta update function available to the dependent `MLXVLM` target. Delete the duplicate helpers from `MLXVLM/Models/Qwen35.swift`, import `MLXLLM`, and let the existing Qwen3.5 VLM call resolve to the shared implementation.

This is an inherently Qwen3.5-only change: only `MLXVLM.Qwen35.GatedDeltaNet` changes its call target. Other VLMs, text models, cache types, model routing, and CLI behavior remain unchanged. Text-only requests still load the VLM when the checkpoint advertises vision support, so image capability is preserved.

Do not port Rapid-MLX's separate blocked-sequential kernel. The existing Swift Metal kernel is the smallest first fix and already serves the text implementation. Do not add a `--no-mllm` flag as the fix because that would avoid rather than repair the vision-capable path.

Revert the unproven zero-copy cache policy and its `mlx-swift` copy primitive. Keep only cache-hit timing diagnostics if they remain useful and do not affect behavior.

## Safety

The shared function retains its existing GPU/CPU dispatch and masked fallback behavior. The VLM passes the same Qwen3.5 tensor layout and cache state used by the text model. No new Metal source or shape contract is introduced.

Before accepting the change, compare the shared kernel with the old ops recurrence on small deterministic inputs, including a cached state. Existing Qwen3.5 VLM tests and SwiftLM prompt-cache tests must continue to pass.

## Verification

Run focused numerical and Qwen3.5 tests, the SwiftLM test suite, a release build, and `git diff --check`.

Then run the original server command and deterministic two-request benchmark. The change passes when the cached request:

- reuses about 5,500 prompt tokens;
- has no more than 100 uncached tokens;
- reaches the first generated token within 5 seconds;
- keeps memory demand below 28 GB;
- produces the expected deterministic response.

If the existing kernel does not meet the target, profile that shared kernel before considering Rapid-MLX's blocked implementation as a separate change.
