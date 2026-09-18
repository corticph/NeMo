# Trie Serialization Round-Trip vs. Rebuild

## Overview

This benchmark answers a key architecture question: **can we pass a compiled boosting trie back and forth between the service layer and the GPU inference endpoint, instead of rebuilding it from phrases on each request?**

### Context

We cannot rely on sticky sessions between `ml-service-transcription` and the Triton server. Each request may be routed to a different Triton replica via round-robin load balancing. This means:

- A compiled trie cannot live permanently on one Triton endpoint.
- The trie must either travel with every request, or be rebuilt on the endpoint that receives the request.
- Rebuilding from phrases takes ~1.36s for 10k terms (per [SR-2752](https://linear.app/corti/issue/SR-2752/benchmark-how-long-it-takes-to-build-a-boosting-trie)), which would add unacceptable latency to every request.

This benchmark measures the cost of serializing a compiled trie, transferring it, and restoring it on the GPU side, and compares that to rebuilding from scratch.

## What Was Benchmarked

Three serialization formats were tested:

1. **state_dict + torch.save** (lightweight bytes, closest to a gRPC payload)
2. **.nemo file** (tarball with state_dict + YAML config, the existing on-disk format)
3. **direct multi-model** (serialize tensors + metadata, deserialize directly into `GPUBiasingMultiModel`'s concatenated GPU buffers — no intermediate `GPUBoostingTreeModel` created)

For formats 1-2, the following stages were timed separately:

| Stage | Description | Code path |
|-------|-------------|-----------|
| Build | Compile trie from phrases (baseline) | `GPUBoostingTreeModel.from_config()` |
| Serialize | Convert compiled trie to bytes/file | `model.state_dict()` + `torch.save()` or `model.save_to()` |
| Deserialize | Reconstruct `GPUBoostingTreeModel` from bytes/file | `load_state_dict()` or `from_nemo()` |
| Register | Add deserialized trie to `GPUBiasingMultiModel` on GPU | `biasing_multi_model.add_model()` |
| Round-trip | Serialize + Deserialize + Register | — |

For format 3 (direct), the deserialize and register steps are combined into a single operation that copies tensors directly from bytes into the multi-model's concatenated GPU buffers, bypassing the intermediate `GPUBoostingTreeModel` object entirely:

| Stage | Description | Code path |
|-------|-------------|-----------|
| Serialize (direct) | Extract 9 tensor slices + metadata from compiled trie | `torch.save(payload, buf)` |
| Deserialize (direct) | Copy from bytes → multi-model's GPU buffers + set metadata | `torch.load()` → `all_*[offset:offset+n].copy_()` |

Correctness was verified by calling `advance()` on both the original and deserialized trie with identical random states and asserting scores and next_states match. For the direct path, correctness was verified through the `GPUBiasingMultiModel.advance()` method (with alpha scaling), confirming the multi-model produces identical results to the standalone model.

## Configuration

| Parameter | Value |
|---|---|
| Model | `nvidia/parakeet-tdt-0.6b-v2` |
| Device | `cuda:0` |
| Iterations | 10 (3 warmup) |
| Seed | 42 (reproducible phrase generation) |
| Term counts | 100, 1000, 5000, 10000 |
| Alpha | 2.0 |

Phrases were synthetic medical-style multi-word terms (1-4 words from a pool of ~100 medical terms), generated with a fixed seed for reproducibility. All phrases within each set are unique.

## Results

### Per-stage timing

#### 100 terms (trie: 873 states, 1,854 arcs)

| Stage | Median | Size | vs. Build |
|---|---|---|---|
| build (from_config) | 13ms | -- | 1.000x |
| serialize (state_dict) | 0ms | 68.7KB | 0.029x |
| serialize (.nemo) | 5ms | 80.0KB | 0.392x |
| serialize (direct) | 0ms | 52.8KB | 0.031x |
| deserialize (state_dict) | 4ms | -- | 0.333x |
| deserialize (.nemo) | 8ms | -- | 0.586x |
| deserialize (direct) | 118ms | -- | 8.862x |
| register to multi-model | 130ms | -- | 9.774x |

| Round-trip | Total | vs. Build |
|---|---|---|
| state_dict | 134ms | 10.136x |
| .nemo | 143ms | 10.752x |
| direct | 118ms | 8.894x |

#### 1,000 terms (trie: 8,273 states, 9,239 arcs)

| Stage | Median | Size | vs. Build |
|---|---|---|---|
| build (from_config) | 99ms | -- | 1.000x |
| serialize (state_dict) | 1ms | 357.8KB | 0.006x |
| serialize (.nemo) | 6ms | 370.0KB | 0.066x |
| serialize (direct) | 1ms | 341.8KB | 0.006x |
| deserialize (state_dict) | 5ms | -- | 0.055x |
| deserialize (.nemo) | 10ms | -- | 0.101x |
| deserialize (direct) | 86ms | -- | 0.871x |
| register to multi-model | 90ms | -- | 0.913x |

| Round-trip | Total | vs. Build |
|---|---|---|
| state_dict | 96ms | 0.974x |
| .nemo | 107ms | 1.080x |
| direct | 87ms | 0.878x |

#### 5,000 terms (trie: 39,411 states, 40,377 arcs)

| Stage | Median | Size | vs. Build |
|---|---|---|---|
| build (from_config) | 752ms | -- | 1.000x |
| serialize (state_dict) | 1ms | 1.54MB | 0.001x |
| serialize (.nemo) | 13ms | 1.55MB | 0.017x |
| serialize (direct) | 2ms | 1.52MB | 0.002x |
| deserialize (state_dict) | 7ms | -- | 0.009x |
| deserialize (.nemo) | 13ms | -- | 0.017x |
| deserialize (direct) | 83ms | -- | 0.110x |
| register to multi-model | 77ms | -- | 0.102x |

| Round-trip | Total | vs. Build |
|---|---|---|
| state_dict | 84ms | 0.112x |
| .nemo | 102ms | 0.136x |
| direct | 84ms | 0.112x |

#### 10,000 terms (trie: 73,586 states, 74,552 arcs)

| Stage | Median | Size | vs. Build |
|---|---|---|---|
| build (from_config) | 1.47s | -- | 1.000x |
| serialize (state_dict) | 1ms | 2.84MB | 0.001x |
| serialize (.nemo) | 24ms | 2.85MB | 0.016x |
| serialize (direct) | 2ms | 2.83MB | 0.001x |
| deserialize (state_dict) | 8ms | -- | 0.006x |
| deserialize (.nemo) | 15ms | -- | 0.010x |
| deserialize (direct) | 76ms | -- | 0.052x |
| register to multi-model | 76ms | -- | 0.052x |

| Round-trip | Total | vs. Build |
|---|---|---|
| state_dict | 85ms | 0.058x |
| .nemo | 115ms | 0.078x |
| direct | 78ms | 0.053x |

### Summary: round-trip / build ratio

| Terms | Build | RT (state_dict) | RT (.nemo) | RT (direct) | SD ratio | .nemo ratio | Direct ratio |
|---|---|---|---|---|---|---|---|
| 100 | 13ms | 134ms | 143ms | 118ms | 10.136x | 10.752x | 8.894x |
| 1,000 | 99ms | 96ms | 107ms | 87ms | 0.974x | 1.080x | 0.878x |
| 5,000 | 752ms | 84ms | 102ms | 84ms | 0.112x | 0.136x | 0.112x |
| 10,000 | 1.47s | 85ms | 115ms | 78ms | 0.058x | 0.078x | 0.053x |

### Correctness

All sizes and all three formats passed correctness verification:

- **state_dict**: PASS at 100, 1000, 5000, 10000 terms
- **.nemo**: PASS at 100, 1000, 5000, 10000 terms
- **direct**: PASS at 100, 1000, 5000, 10000 terms (verified through `GPUBiasingMultiModel.advance()` with alpha scaling)

### Direct path: beam search and batching compatibility

The direct path was verified to be fully compatible with batched beam search:

- `GPUBiasingMultiModel` does **not** store references to individual `GPUBoostingTreeModel` objects (no `self.models` list).
- `add_model()` performs a one-time copy of tensor data into concatenated `all_*` GPU buffers + metadata bookkeeping. After this, the original model object is not needed.
- The Triton kernel is launched **once per batch** with `batch_size` programs; each program independently routes to its model via `model_ids` → offset lookup into the concatenated buffers.
- `advance()`, `get_alphas()`, and `remove_model()` operate exclusively on the multi-model's own buffers and the `model_id` integer.
- The direct path copies the **exact same data** into the **exact same buffer locations** as `add_model()` — the kernel cannot distinguish between the two paths.

## Architecture Implications

### The no-sticky-session constraint

We cannot guarantee that consecutive requests from the same session are routed to the same Triton replica. The boosting trie must therefore be available at whichever endpoint receives each request. Three approaches were considered:

#### 1. Compile on Triton, return to service, pass with subsequent requests (recommended)

On the first request of a session, the service sends the keyterm list to Triton. Triton compiles the trie, decodes the audio, and returns the transcript **plus the serialized trie bytes**. The service caches the trie bytes in Redis session state. On subsequent requests, the service sends the cached trie bytes instead of the keyterm list. Any Triton endpoint can deserialize the trie in ~78ms (direct path) and proceed with decoding.

**Per-request cost breakdown** (10k terms, direct format):

| Step | Where | Time |
|------|-------|------|
| Serialize trie to bytes (first request only) | Triton side | ~2ms |
| Transfer 2.83 MB over network | Network | ~25ms (1 Gbit link) |
| Deserialize directly into multi-model GPU buffers | Triton side | ~76ms |
| **Total per-request overhead** (subsequent requests) | | **~101ms** |
| vs. rebuild from phrases | Triton side | 1,470ms |

This is **14.5x faster** than rebuilding and requires no state management on the Triton side. The service only needs to store ~2.83 MB of bytes in Redis per session.

#### 2. Rebuild on Triton side from phrases

Each Triton endpoint rebuilds the trie from the raw phrase list on every request. This adds ~1.47s latency per request for 10k terms, which is unacceptable for real-time transcription of ~8s audio chunks.

#### 3. Shared cache (Redis/NATS)

Build the trie once, store the serialized bytes in a shared cache (Redis or NATS). Each Triton endpoint fetches and caches locally on first hit for a given session/keyterm set. Subsequent requests to the same endpoint hit the local cache.

This avoids sending 2.83 MB on every request but adds:
- Cache management complexity (TTL, invalidation, eviction)
- A cache-miss path that falls back to rebuild or remote fetch
- Potential for stale caches if keyterms change mid-session

This may be worth pursuing as an optimization if the 2.83 MB per-request transfer becomes a bottleneck, but the pass-with-request approach is simpler and sufficient for the initial production rollout.

### Format recommendation

**Direct multi-model** is the recommended format for passing tries over the network:
- 78ms round-trip vs 85ms for state_dict vs 115ms for .nemo at 10k terms
- No intermediate `GPUBoostingTreeModel` object created — bytes go directly into GPU buffers
- Smallest payload (2.83 MB vs 2.84 MB for state_dict — saves the `vocab_size` padding slots)
- Eliminates redundant `_resolve_final()` call (final weights are pre-resolved before serialization)
- Fully compatible with batched beam search and per-stream biasing

The state_dict format is a good fallback if the direct path is not yet implemented on the Triton side — it works with existing NeMo APIs (`from_nemo` / `load_state_dict`) and only requires adding `add_model()` after deserialization.

### Scaling characteristics

- **Build time** scales linearly with term count (O(n_phrases) for tokenization + graph construction)
- **Serialize time** is negligible (~1-2ms) for all formats and sizes
- **Deserialize time** for state_dict/.nemo scales with trie size but stays under ~15ms even at 10k terms
- **Deserialize+register (direct)** is dominated by GPUBiasingMultiModel construction + GPU copy — ~76-83ms at scale, largely fixed overhead
- **Serialized size** scales linearly: ~0.05 MB/100 terms → ~2.83 MB/10k terms

The crossover point where serialization beats rebuilding is around **1000 terms** — below that, rebuilding is fast enough (~99ms) that serialization overhead (dominated by GPUBiasingMultiModel construction ~80ms) doesn't pay off. Above 1000 terms, serialization is increasingly dominant.

## Key Implementation Notes

- The compiled trie (`GPUBoostingTreeModel`) contains 9 tensors (3 `nn.Parameter` + 6 persistent `nn.Buffer`), all captured by `state_dict()`. A small metadata dict (`num_states`, `num_arcs`, `max_order`, `vocab_size`) is needed to instantiate the empty model before `load_state_dict()`.
- After `load_state_dict()`, `_resolve_final()` must be called to resolve backoff chains in `final_weights`. This is a CPU-only operation taking <1ms. In the direct path, `final_weights` are pre-resolved before serialization, so this step is eliminated.
- The `GPUBiasingMultiModel.add_model()` method copies trie data from the `GPUBoostingTreeModel` into concatenated GPU buffers. This is pure tensor copying + metadata bookkeeping — no computation, no remapping.
- The `GPUBiasingMultiModel` does not store references to individual `GPUBoostingTreeModel` objects. After `add_model()` returns a `model_id`, every subsequent operation (`advance`, `get_alphas`, `remove_model`) works purely from the `model_id` integer and the concatenated buffers.
- The direct serialization format stores only `num_arcs` elements of arc data (not the `vocab_size` padding), making it slightly smaller than state_dict.
- An existing in-memory cache (`_BIASING_MODEL_CACHE` in `biasing_multi_model.py`) allows caching compiled `GPUBoostingTreeModel` instances by a string key. This could be used on the Triton side to avoid re-deserializing the same trie if the same session hits the same endpoint twice.

## References

- [SR-2752: Benchmark how long it takes to build a boosting trie](https://linear.app/corti/issue/SR-2752/benchmark-how-long-it-takes-to-build-a-boosting-trie)
- [SR-3053: POC: Trie serialization round-trip benchmark](https://linear.app/corti/issue/SR-3053/poc-trie-serialization-round-trip-benchmark-passing-compiled-trie-per)
- [Project: Enable ASR decoder-based keyterm biasing with GPU-PB](https://linear.app/corti/project/enable-asr-decoder-based-keyterm-biasing-with-gpu-pb-59120dd4f777)
- GPU-PB paper: https://arxiv.org/abs/2508.07014
- NeMo docs: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/asr_customization/word_boosting.html
- POC repo (Corti fork, `key-term-bias-demo` branch): https://github.com/corticph/NeMo/tree/key-term-bias-demo

## Usage

```bash
# Full benchmark (all default term counts)
export CUDA_VISIBLE_DEVICES=0
python scripts/benchmark_trie_serialization.py

# Custom term counts and iterations
python scripts/benchmark_trie_serialization.py --term-counts 1000,10000 --iterations 20

# Quick smoke test
python scripts/benchmark_trie_serialization.py --term-counts 10 --iterations 2 --warmup 1
```
