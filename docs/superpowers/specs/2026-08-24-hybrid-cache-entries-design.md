# Hybrid Prompt Cache Entries

## Goal

Add `--hybrid-cache-entries N` so models with non-trimmable hybrid or sliding-window caches can retain several reusable prompt prefixes. This is intended for repeated workloads whose prompts share different large prefixes, while keeping memory use explicitly bounded by an entry count.

## CLI contract

| Value | Behaviour |
| --- | --- |
| `0` | Disable rolling hybrid-cache snapshots. |
| `1` | Retain one rolling hybrid snapshot. This is the default and preserves current SwiftLM behaviour. |
| `N > 1` | Retain up to `N` rolling hybrid snapshots using least-recently-used eviction. |
| `N < 0` | Reject the configuration at startup. |

The pinned system-prompt snapshot is separate from this limit. Pure `KVCacheSimple` models continue to use their existing single rolling snapshot regardless of this option.

## Scope

The change extends the existing in-memory `PromptCache` actor. It does not port Rapid-MLX's wider cache subsystem, byte-based budgets, persistence, radix indexing, response caching, or automatic model-specific defaults.

Existing cache exclusions remain unchanged, including multimodal input, quantized KV cache, speculative decoding, MTP, and TurboQuant paths.

## Architecture

### Cache ownership

`PromptCache` receives a hybrid entry capacity when the server starts. It continues to own:

- one rolling snapshot for fully trimmable `KVCacheSimple` state;
- up to `N` rolling snapshots for state containing any non-`KVCacheSimple` layer;
- one optional pinned system-prompt snapshot outside both rolling stores.

The hybrid snapshots are held in least-recently-used order. Keeping this state inside the actor preserves the existing concurrency boundary.

### Saving snapshots

Snapshots are saved only at the exact message boundaries already used by SwiftLM. A fully trimmable snapshot replaces the existing single rolling snapshot.

For a non-trimmable snapshot:

1. Capacity `0` skips the save.
2. An existing snapshot with identical tokens is replaced and becomes most recently used.
3. A new snapshot is appended as most recently used.
4. Entries are evicted from the least-recently-used end until the configured capacity is met.

Evictions are logged. The server configuration log includes the selected capacity.

### Restoring snapshots

A fully trimmable request compares the existing rolling snapshot and the pinned snapshot. A non-trimmable request compares all hybrid snapshots and the pinned snapshot, then selects the longest safe prefix.

Non-trimmable state is reusable only for strict prefix extension: the cached tokens must match from the start, no cached token may need trimming, and the new request must contain at least one uncached token. An exact full match remains ineligible because generation still needs the final prompt token to be replayed.

A hybrid hit refreshes that entry's least-recently-used position. Equal-length ties prefer a rolling entry over the pinned entry and prefer the most recently used hybrid entry. Restored state is copied before use so later generation cannot mutate a retained snapshot.

## Memory and safety

The capacity is an entry-count bound, not a byte bound. Hybrid snapshots can be large, so users selecting values above `1` accept a model-dependent memory trade-off. The default does not increase SwiftLM's current retained snapshot count.

The pinned system prompt remains write-once and cannot be evicted by rolling hybrid entries. Existing restore checks for cache shape, metadata, offsets, and layer compatibility remain in force.

## Verification

Automated coverage will verify:

- CLI default, zero, positive, and negative values;
- capacity zero skipping hybrid saves;
- longest-prefix selection across hybrid entries;
- hit-based least-recently-used refresh and eviction;
- replacement of identical-token entries without growth;
- the pinned snapshot surviving hybrid eviction;
- unchanged single-snapshot behaviour for `KVCacheSimple`;
- immutable repeated restores;
- the existing prompt-cache test suite and a release build.

After implementation, runtime validation will use the lower-memory model supplied by the user. It will confirm reuse in server logs and compare retained memory at capacities `0`, `1`, and a small value above `1`.
