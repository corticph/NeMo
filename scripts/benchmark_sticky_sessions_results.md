# Sticky Session Capacity Benchmark

## Overview

**Question:** How many concurrent sticky sessions (each with its own boosting trie) can a single Triton endpoint maintain on a 10 GB MIG partition, and what is the memory cost?

- **Production target:** H100 split into 7 MIGs (~10 GB each)
- **Local POC:** A100 GPU 7 with 7x `1g.10gb` MIG partitions (9.5 GB each)
- **Model:** `nvidia/parakeet-tdt-0.6b-v2` (0.6B params, 2.41 GB GPU memory)
- **Audio:** LibriSpeech test.clean — tested with natural segments (max 23.3s) and longest available (30–35s) to simulate production `/transcripts` traffic
- **Production batch limit:** `max_batch_size: 8` (see `triton-model-repository/src/packages/nemo_tdt/config.pbtxt`)

## Results

### Registered Session Capacity (10k terms per trie, ~74k arcs each)

Memory as concurrent sessions accumulate. Each trie is compiled once and registered into a single `GPUBiasingMultiModel`, which uses power-of-2 buffer doubling.

| Sessions | Trie Mem | Persistent Total | Full-Pipeline Peak (23s) | Full-Pipeline Peak (35s) | % of 9.5 GB (35s) |
|----------|----------|------------------|--------------------------|--------------------------|-------------------|
| 0 | 0 MB | 2.41 GB | — | — | 25% |
| 200 | 974 MB | 3.37 GB | — | — | — |
| 400 | 2,009 MB | 4.38 GB | — | — | — |
| 600 | 4,084 MB | 6.40 GB | — | — | — |
| 800 | 4,084 MB | 6.40 GB | 7.57 GB | 8.14 GB | 86% |
| ~850 | OOM | — | — | — | buffer doubling needs ~2 GB |

Full-pipeline peak = model + tries + encoder workspace + decoder workspace, measured during a complete encode+decode of batch=8. The encoder workspace scales with audio duration (1.1 GB at 23s, 1.7 GB at 35s); the decoder workspace is only ~0.04 GB.

**800 sessions + batch=8 full pipeline peaks at 7.57 GB (23s) or 8.14 GB (35s)** — 80–86% of the MIG. The breaking point is ~850 sessions, caused by buffer doubling requiring ~2 GB to grow, not by trie data.

Memory is flat between doublings: 600 and 800 sessions use the same 4,084 MB because the buffer last doubled at ~441 sessions.

### Decode Performance (LibriSpeech, batch=8, 10k terms per trie)

Full pipeline (encode + decode) timed and measured. The 0-session peak includes one-time CUDA init; all rows have all 800 tries registered in memory.

| Registered Sessions | 23s Decode Time | 23s RTFx | 23s Peak | 35s Decode Time | 35s RTFx | 35s Peak |
|---------------------|-----------------|----------|----------|-----------------|----------|----------|
| 0 | 1907ms | 42x | 7.57 GB* | 3054ms | 85x | 8.14 GB* |
| 200 | 1901ms | 42x | 7.57 GB | 3047ms | 85x | 8.14 GB |
| 400 | 1898ms | 42x | 7.57 GB | 3058ms | 85x | 8.14 GB |
| 600 | 1896ms | 42x | 7.57 GB | 3120ms | 83x | 8.14 GB |
| 800 | 1900ms | 42x | 7.57 GB | 3040ms | 85x | 8.14 GB |

\*0-session peak includes one-time CUDA kernel compilation. All subsequent decodes peak the same since all 800 tries remain registered.

**Zero decode performance degradation from 0 to 800 concurrent tries.** The Triton kernel's `model_id` → offset lookup is O(1). RTFx is flat — far above realtime.

### Memory Breakdown

The full-pipeline peak is the maximum of two phases:

| Phase | What it does | Memory | Duration-dependent? |
|-------|-------------|--------|---------------------|
| Encoder | Convolutions + subsampling on raw audio | Persistent + 1.1 GB (23s) / 1.7 GB (35s) | Yes — scales with timesteps |
| Decoder | Beam search on encoded output | Persistent + ~0.04 GB | No — small, constant |

**The encoder workspace dominates the variable memory cost**, not the decoder. This is not unique to biasing — the encoder workspace is the same with or without tries. The encoder allocates large intermediate tensors for convolutional features that scale with input duration.

At 800 tries:
- 23s: encoder peak = 6.40 + 1.13 = 7.53 GB (vs decoder peak = 6.40 + 0.04 = 6.44 GB) → encoder dominates
- 35s: encoder peak = 6.40 + 1.70 = 8.10 GB (vs decoder peak = 6.40 + 0.04 = 6.44 GB) → encoder dominates

### Variable Trie Sizes (0–50k terms, realistic production mix)

Production sessions will have different keyterm counts. This test draws random term counts from [0, 50,000] per trie (mean ~25,000). Memory only — decode performance not measured separately since the uniform test already showed O(1) scaling.

| Sessions | Variable Trie Mem | Variable Total GPU | Uniform 10k Trie Mem | Variable avg arcs/trie |
|----------|-------------------|---------------------|--------------------|-----------------------|
| 50 | 457 MB | 2.85 GB | 196 MB | 126,738 |
| 100 | 974 MB | 3.36 GB | 457 MB | 152,697 |
| 150 | 2,009 MB | 4.37 GB | 974 MB | 154,814 |

Memory is proportional to total arcs, not session count or uniformity. The variable tries average ~150k arcs/trie (2x the uniform 74k) and use 2x the memory. Production capacity estimates should use **average terms per session**, not worst-case.

## Capacity Summary

At 800 registered sessions (10k terms each), full pipeline with batch=8:

| Component | 23s audio | 35s audio | Notes |
|-----------|-----------|-----------|-------|
| Model | 2.41 GB | 2.41 GB | Fixed |
| 800 tries | 4.08 GB | 4.08 GB | Persistent, buffer-doubled |
| Encoder workspace | 1.13 GB | 1.70 GB | Scales with audio duration |
| Decoder workspace | ~0.04 GB | ~0.04 GB | Constant |
| **Total peak** | **7.57 GB** | **8.14 GB** | Of 9.5 GB available |
| **Headroom** | **1.93 GB** | **1.36 GB** | |

Per H100 (7 MIGs): **5,600 concurrent sessions** with 10k terms each.

**Audio duration matters:** 35s inputs use 0.57 GB more than 23s due to encoder workspace. At 800 tries, this reduces headroom from 1.93 GB to 1.36 GB — tight but still fits. At higher session counts or longer audio, this could OOM.

## Buffer Doubling and Deployment Memory Management

### How trie storage works

All registered tries are concatenated into a single set of GPU buffers (`all_arcs_weights`, `all_from_states`, `all_to_states`, `all_ilabels`, and state-level buffers for backoff/final weights). Each trie gets an offset into these shared buffers. There is one `GPUBiasingMultiModel` per endpoint, holding all sessions' tries.

### The buffer doubling problem

The multi-model starts with 1M-entry buffers (`INIT_NUM_ARCS = 1_000_000`). When a new trie is added via `add_model()` and the total exceeds the reserved capacity, `_maybe_extend_arcs_and_states()` doubles the buffer using `torch.cat(old_buffer, new_zeros)`. This requires **both the old and new buffer to exist simultaneously in GPU memory** during the copy. For example, growing a 2 GB buffer to 4 GB needs ~6 GB for a few hundred milliseconds.

This is why the benchmark OOMs at ~850 sessions: at that point the buffer holds ~63M arcs (~4 GB reserved), and the next doubling tries to grow to 128M entries. The `torch.cat` call needs the old 4 GB buffer + new 8 GB buffer = 12 GB simultaneously. With the 2.41 GB model already loaded, the spike exceeds the 10 GB MIG.

### Session lifecycle and why buffers never shrink

In production, sessions arrive and leave continuously:

- **Session start**: first request carries the keyterm list. The endpoint compiles the trie (~1.5s for 10k terms), calls `add_model()`, gets a `model_id`. All subsequent requests in the session use this `model_id` via `multi_biasing_ids`.
- **Session end**: `remove_model()` is called. The trie's arcs are cleared and the buffer is compacted (subsequent entries shifted left to fill the gap). The `model_id` returns to a free pool for reuse by new sessions.
- **But buffer capacity never shrinks**: `num_arcs_extended_reserved` only grows. After the first peak load, the buffer stays at peak size even if sessions leave and the actual arc count drops.

This means the buffer doubles to whatever size the peak concurrent session count requires, and stays there for the endpoint's lifetime. Memory is effectively allocated at peak, not at current load.

### Pre-allocation as the solution

Instead of starting at 1M entries and doubling at runtime, initialize the buffers at startup for the expected concurrent session count:

```python
biasing_multi_model = GPUBiasingMultiModel(vocab_size=1024, use_triton=True)
biasing_multi_model.pre_allocate(
    max_arcs=64_000_000,    # 800 sessions × ~74k arcs + safety margin
    max_states=64_000_000,
    max_models=1024,
)
```

This allocates the full buffers once at startup (~4 GB), before any sessions arrive. No runtime doubling, no `torch.cat` spikes, no OOM risk. The endpoint can handle up to the pre-allocated session count for its entire lifetime.

### Tradeoffs

| | Current (doubling) | Pre-allocated |
|---|---|---|
| Startup memory (0 sessions) | 2.41 GB (model only) | 6.41 GB (model + buffers) |
| Peak memory (800 sessions, 35s) | 8.14 GB | 8.14 GB |
| Runtime OOM risk | Yes — `torch.cat` spike can exceed MIG | No — allocated once at startup |
| Capacity limit | ~850 sessions (buffer doubling OOMs) | Pre-allocated limit (e.g., 800) |
| Memory after sessions leave | Stays at peak (buffer never shrinks) | Stays at pre-allocated (same) |

Pre-allocation is strictly better for long-running endpoints expected to reach high session counts. The memory cost is the same as the doubling approach would reach anyway, but without the OOM risk. The only scenario where doubling wins is lightly-loaded endpoints that never exceed the 1M initial buffer (~14 sessions with 10k terms).

### Handling variable trie sizes in production

With variable keyterm counts (e.g., 0–50k terms, avg ~150k arcs per trie), 800 sessions would need ~120M arcs. Pre-allocating 128M entries would use ~8 GB, leaving only ~1 GB for the model + decode — doesn't fit on a 10 GB MIG.

The practical approach: pre-allocate for **expected average concurrent sessions × average arcs per trie**, with a safety margin. When the buffer is full, reject new sessions and route them to another MIG rather than attempting runtime growth. For example:

- 800 sessions × 74k arcs (10k terms) → pre-allocate 64M arcs → fits
- 400 sessions × 150k arcs (25k terms avg) → pre-allocate 64M arcs → fits
- 200 sessions × 300k arcs (50k terms) → pre-allocate 64M arcs → fits

The MIG has room for one of these profiles, not all at once.

## Key Findings

1. **800 registered sessions + batch=8 full pipeline peaks at 7.57 GB (23s) / 8.14 GB (35s)** — 80–86% of the 10 GB MIG. Per H100: 5,600 sessions.

2. **Zero decode performance impact** from 0 to 800 concurrent tries — O(1) model_id lookup, RTFx flat.

3. **The encoder workspace dominates the variable memory cost**, not the decoder. Encoder workspace is 1.1 GB at 23s and 1.7 GB at 35s; decoder workspace is only ~0.04 GB. This is not unique to biasing — the encoder cost is the same with or without tries.

4. **Audio duration matters.** 35s inputs use 0.57 GB more than 23s due to encoder workspace. At 800 tries, headroom drops from 1.93 GB (23s) to 1.36 GB (35s).

5. **Breaking point (~850 sessions) is caused by buffer doubling**, not trie data. See [Buffer Doubling and Deployment Memory Management](#buffer-doubling-and-deployment-memory-management) for details.

6. **Variable trie sizes work seamlessly.** Memory scales with total arcs regardless of per-trie uniformity.

## Architecture Implications

- **Trie compiled once at session start** (~1.5s for 10k terms), reused for entire session
- **Failure recovery**: serialized trie restores in 78ms from cache (vs 1.5s recompile)
- **Pre-allocation is the key optimization**: sizing multi-model buffers for expected session count at startup eliminates runtime doubling and its OOM risk
- **Session lifecycle**: register trie at session start, deregister at session end

## Conclusion and Recommendations

The benchmark confirms that sticky-session keyterm biasing is viable at production scale on 10 GB MIG partitions. 800 concurrent sessions with 10k terms each + a full batch=8 encode+decode peaks at 7.57–8.14 GB (80–86% of the MIG), with zero decode performance impact. Biasing was verified to take effect.

The capacity bottleneck is buffer doubling in `GPUBiasingMultiModel`, not trie data size. See [Buffer Doubling and Deployment Memory Management](#buffer-doubling-and-deployment-memory-management) for details and the pre-allocation solution.

### Recommended Next Steps

1. **Add trie support to `triton-model-repository`**: implement per-session trie registration/deregistration in the NeMo TDT Triton backend. The `GPUBiasingMultiModel` and `BeamBatchedTDTInfer` APIs already support per-stream biasing via `multi_biasing_ids` — wire this through the Triton model to accept a session ID per request.

2. **Integrate sticky session routing in `ml-service-transcription`**: route requests from the same session to the same Triton endpoint so the trie stays local. Cache serialized trie bytes (78ms restore) in Redis for failure recovery — if the endpoint dies, the trie can be restored on a new endpoint without recompilation (1.5s).

3. **Pre-allocate multi-model buffers**: size the `GPUBiasingMultiModel` buffers for the target session count at endpoint startup instead of power-of-2 runtime doubling. This eliminates the ~850 session breaking point and reduces memory waste from over-allocation.

4. **Session lifecycle management**: register trie at session start (compile from keyterms, ~1.5s for 10k terms), deregister at session end to free the slot. Track session→endpoint mapping in the load balancer.

5. **Diarizer state migration**: move `FIFO_CACHE`/`SPKCACHE` from per-request passing to endpoint-local state, same as the trie. This is a natural follow-up since sticky sessions already require endpoint-local state.

## Reproduction

```bash
# Full benchmark: biasing verification + memory capacity + decode performance (23s)
CUDA_VISIBLE_DEVICES=MIG-b9e23c79-55f7-5638-823d-56a0a9b84b09 \
    python scripts/benchmark_sticky_sessions.py \
    --max-sessions 800 --term-count 10000 --batch-size 8 \
    --decode-session-counts 0,200,400,600,800 --session-steps 200,400,600,800

# Same but with 35s audio (simulates production /transcripts traffic)
CUDA_VISIBLE_DEVICES=MIG-b9e23c79-55f7-5638-823d-56a0a9b84b09 \
    python scripts/benchmark_sticky_sessions.py \
    --max-sessions 800 --term-count 10000 --batch-size 8 --pad-to-duration 30 \
    --decode-session-counts 0,200,400,600,800 --session-steps 200,400,600,800

# Variable trie sizes (0-50k terms, memory only)
CUDA_VISIBLE_DEVICES=MIG-b9e23c79-55f7-5638-823d-56a0a9b84b09 \
    python scripts/benchmark_sticky_sessions.py \
    --max-sessions 150 --variable-trie-sizes --no-decode-test --session-steps 50,100,150
```
