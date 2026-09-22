# Keyterm Biasing Benchmark: MedRad (arXiv:2605.16545v2 reproduction)

## Summary

GPU phrase-boosting with a 2507-term medical vocabulary reproduces the Symphony keyterm-biasing result on MedRad almost exactly: at biasing strength α=0.5 with beam search (width 4), the medical-term miss rate halves (rFNR_med **50.2%** vs the paper's 50.9%) while precision barely moves (P_med 97.7% → 96.6% vs the paper's 97.0% → 96.7%), and WER drops 23.4% → 19.3%. Biasing alone (greedy decoding) adds ~+10 pts of term recall over the unbiased baseline; switching to beam search adds ~+9 more on top.

- **Model:** `nvidia/parakeet-tdt-0.6b-v2` (Parakeet TDT 0.6B) — the paper's open-source Parakeet baseline
- **Dataset:** MedRad — `radai_en` (config `en`, split `test_spoken`; 4855 utterances, ~30 h; 2507 unique medical terms from the `medical_terms` column)
- **Decoding:** batched greedy decoding (`greedy_batch`) and `malsd_batch` beam search with GPU phrase-boosting (GPU-PB) shallow fusion; `context_score=1.0`, `depth_scaling=2.0`, `bpe_mode=case_insensitive`
- **Script:** `scripts/asr_context_biasing/eval_keyterm_biasing_radai.py` — unit tests: `tests/collections/asr/test_eval_keyterm_biasing_radai.py`
- **Reference point (paper Table 7, Symphony on MedRad):** rFNR_med = 50.9%, P_med 97.0% → 96.7%

**Legend:** bold marks the best value per column within each table — lower is better for WER, higher for recall, precision and rFNR. **Pilot** tables use a random 500-utterance subset (seeded shuffle, `seed=1234`) for configuration guidance only; all other tables use the full 4855-utterance split.

## Results

### Headline: full-split validation (beam=4)

| α     | WER % | R_med % | P_med % | rFNR_med % |
|-------|-------|---------|---------|------------|
| 0.0 (unbiased) | 23.39 | 63.57 | **97.73** | — |
| 0.5 (biased)   | **19.32** | **81.86** | 96.62 | **50.21** |

This reproduces paper Table 7 almost exactly: **rFNR 50.21% vs 50.9%**, **P_med 97.73% → 96.62% vs 97.0% → 96.7%** — and WER additionally improves 23.39% → 19.32%.

### Which strength? (pilot, beam=4)

| α   | WER % | R_med % | P_med % | rFNR_med % |
|-----|-------|---------|---------|------------|
| 0.0 | 23.72 | 63.41 | **98.25** | — |
| 0.5 | **19.37** | 82.22 | 97.21 | 51.40 |
| 1.0 | 22.78 | **83.25** | 83.51 | **54.21** |
| 1.5 | 40.85 | 77.18 | 50.27 | 37.64 |
| 2.0 | 59.60 | 72.35 | 36.69 | 24.44 |
| 3.0 | 94.60 | 60.02 | 20.73 | -9.27 |

α=0.5 is the sweet spot: precision stays within ~1 pt of baseline while WER improves by ~4 pts. At α≥1.0 precision collapses — biasing starts inserting spurious medical terms.

### Does beam width matter? (pilot, α=0.5)

| Decoding | rFNR_med % | P_med % | WER % |
|----------|------------|---------|-------|
| greedy (full split) | 27.45 | 95.47 | 22.07 |
| beam=1 | 27.17 | **97.40** | 22.26 |
| beam=2 | 36.13 | 96.91 | 20.24 |
| beam=4 | 51.40 | 97.21 | **19.37** |
| beam=5 | 52.38 | 96.98 | 19.67 |
| beam=8 | **54.90** | 96.67 | 19.77 |

Single-hypothesis decoding (greedy, bit-identical to beam=1 when unbiased) cannot exploit a biasing trie — rFNR caps at ~27%. Beam search needs hypothesis diversity for the trie scores to change the argmax; beyond width 4 the gains flatten.

<details>
<summary>Full sweep tables (α grids per decoding configuration)</summary>

**Greedy decoding sweep (4855 utterances, `greedy_batch`)**

| α   | WER % | R_med % | P_med % | rFNR_med % |
|-----|-------|---------|---------|------------|
| 0.0 | 24.62 | 62.75 | **97.79** | — |
| 0.5 | **22.07** | **72.98** | 95.47 | **27.45** |
| 1.0 | 25.77 | 71.88 | 81.00 | 24.50 |
| 1.5 | 33.28 | 65.98 | 64.40 | 8.66 |
| 2.0 | 41.96 | 59.02 | 51.37 | -10.01 |
| 3.0 | 58.43 | 47.90 | 35.08 | -39.86 |

**Beam-size sweep (500-utterance pilot)**

| Beam | α   | WER % | R_med % | P_med % | rFNR_med % |
|------|-----|-------|---------|---------|------------|
| 1    | 0.0 | 24.67 | 63.31 | 98.09 | — |
| 1    | 0.5 | 22.26 | 73.28 | 97.40 | 27.17 |
| 1    | 1.0 | 26.19 | 71.33 | 83.51 | 21.85 |
| 1    | 1.5 | 35.80 | 61.56 | 64.00 | -4.76 |
| 1    | 2.0 | 44.16 | 56.63 | 47.13 | -18.21 |
| 1    | 3.0 | 64.85 | 48.82 | 29.39 | -39.50 |
| 2    | 0.0 | 23.50 | 64.44 | **98.58** | — |
| 2    | 0.5 | 20.24 | 77.29 | 96.91 | 36.13 |
| 2    | 1.0 | 24.05 | 77.08 | 82.69 | 35.55 |
| 2    | 1.5 | 31.24 | 73.07 | 61.29 | 24.28 |
| 2    | 2.0 | 49.79 | 64.65 | 41.22 | 0.58 |
| 2    | 3.0 | 83.27 | 55.40 | 24.37 | -25.43 |
| 4    | 0.0 | 23.72 | 63.41 | 98.25 | — |
| 4    | 0.5 | **19.37** | 82.22 | 97.21 | 51.40 |
| 4    | 1.0 | 22.78 | 83.25 | 83.51 | 54.21 |
| 4    | 1.5 | 40.85 | 77.18 | 50.27 | 37.64 |
| 4    | 2.0 | 59.60 | 72.35 | 36.69 | 24.44 |
| 4    | 3.0 | 94.60 | 60.02 | 20.73 | -9.27 |
| 5    | 0.0 | 23.97 | 63.31 | 97.93 | — |
| 5    | 0.5 | 19.67 | 82.53 | 96.98 | 52.38 |
| 5    | 1.0 | 22.95 | 83.45 | 83.37 | 54.90 |
| 5    | 1.5 | 43.09 | 78.31 | 47.48 | 40.90 |
| 5    | 2.0 | 65.35 | 73.69 | 33.27 | 28.29 |
| 5    | 3.0 | 96.71 | 60.53 | 19.60 | -7.56 |
| 8    | 0.0 | 24.25 | 63.31 | 97.93 | — |
| 8    | 0.5 | 19.77 | 83.45 | 96.67 | 54.90 |
| 8    | 1.0 | 24.20 | **85.51** | 78.42 | **60.50** |
| 8    | 1.5 | 48.10 | 82.53 | 43.22 | 52.38 |
| 8    | 2.0 | 69.98 | 77.29 | 30.51 | 38.10 |
| 8    | 3.0 | 101.72 | 66.60 | 19.03 | 8.96 |

</details>

## Key Findings

1. **The paper's result reproduces.** At α=0.5 with beam=4 on the full split, biasing halves the medical-term miss rate while leaving precision essentially unchanged — exactly the paper's "safe biasing" claim.

2. **Biasing needs hypothesis diversity.** Single-hypothesis decoding caps rFNR at ~27%; beam ≥ 4 reaches ~51–55%. Beam=4 remains the practical default, matching the decoding-throughput benchmark's recommendation.

3. **The safe window is narrow.** α=0.5 improves WER ~3–4 pts with precision within ~1.3 pts of baseline; α≥1.0 collapses precision and α 2–3 makes decoding worse than unbiased overall.

4. **Wider beams tolerate stronger biasing, via spurious terms.** beam=8 at α=1.0 reaches rFNR 60.5% but precision falls to 78.4% — recall gains come from spurious detections, the failure mode P_med is designed to catch.

5. **Unbiased behavior is insensitive to search configuration.** R_med 63.3–64.4% and P_med 97.9–98.6% across greedy and all beam sizes; beam width matters almost exclusively through its interaction with biasing.

## Methodology

- **Biasing vocabulary:** one global vocabulary from the union of the `medical_terms` column entries across the full split (2507 terms), built once; the same tree is re-built per decoding configuration
- **Strategy switching:** `model.change_decoding_strategy()` with a deep copy of the base decoding config; `boosting_tree` + `boosting_tree_alpha` set under `decoding.beam` (or `decoding.greedy`); alignments/timestamps/frame confidence disabled and `fused_batch_size=-1` to isolate decode cost
- **Data:** `radai_en` read from the local HuggingFace datasets cache; audio written to 16 kHz WAV; the unbiased baseline (α=0) uses the same decoding strategy without a boosting tree, per configuration
- **Keyterm metric:** TP/FN/FP per the paper's Section 3.1 definitions over word-level Levenshtein alignment (substitutions preferred at ties); matching is case-insensitive and punctuation-insensitive; multi-word terms matched at the term level. rFNR_med = (FNR_unbiased − FNR_biased) / FNR_unbiased (Eq. 3)
- **Runtime:** 10 GB A100 MIG partition (`CUDA_VISIBLE_DEVICES=MIG-b9e23c79-55f7-5638-823d-56a0a9b84b09`), `batch_size=8`, bfloat16 — matching the greedy-vs-beam-search throughput setup

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=MIG-b9e23c79-55f7-5638-823d-56a0a9b84b09 \
python scripts/asr_context_biasing/eval_keyterm_biasing_radai.py \
    out_dir=<output folder> \
    hf_datasets_cache=<path to the HuggingFace datasets cache> \
    n_examples=500 \
    alphas=[0.0,0.5,1.0,1.5,2.0,3.0] \
    strategies=[beam] \
    beam_sizes=[1,2,4,5,8] \
    batch_size=8 \
    compute_dtype=bfloat16
```

Knobs per results table:

| Table | `strategies` / `beam_sizes` | `n_examples` | `alphas` |
|-------|------------------------------|--------------|----------|
| Which strength? (pilot) | `[beam]` / `[4]` | `500` | default |
| Does beam width matter? (pilot) | `[beam]` / `[1,2,4,5,8]` | `500` | default |
| Greedy sweep (full split) | `[greedy]` | `0` (full) | default |
| Headline full-split validation | `[beam]` / `[4]` | `0` (full) | `[0.0,0.5]` |
