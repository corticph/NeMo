#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark sticky session capacity on MIG-partitioned GPUs.

Measures how many concurrent sticky sessions (each with its own boosting trie)
can be maintained on a single Triton endpoint, and whether decode performance
degrades as the number of concurrent tries grows.

Tests:
  1. Memory capacity: register increasing numbers of tries into a single
     GPUBiasingMultiModel, tracking GPU memory and reallocation events.
  2. Decode performance: decode a batch of LibriSpeech audio where each element
     maps to a different trie, measuring RTFx vs. baseline (no biasing).
  3. Scaled batch: scale batch_size = n_sessions to find the realtime limit.

Production target: H100 split into 7 MIGs (~10 GB each).
Local POC: A100 GPU 7 with 7x 1g.10gb MIG partitions.

Usage:
    # Full A100 (for baseline)
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_sticky_sessions.py

    # Target a MIG partition (10 GB)
    CUDA_VISIBLE_DEVICES=MIG-b9e23c79-55f7-5638-823d-56a0a9b84b09 \\
        python scripts/benchmark_sticky_sessions.py

    # Custom session counts and term count
    python scripts/benchmark_sticky_sessions.py --max-sessions 500 --term-count 10000

    # Skip decode test (memory only)
    python scripts/benchmark_sticky_sessions.py --no-decode-test
"""

import argparse
import gc
import random
import time

import torch

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.context_biasing.biasing_multi_model import GPUBiasingMultiModel
from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import (
    BoostingTreeModelConfig,
    GPUBoostingTreeModel,
)
from nemo.collections.asr.parts.submodules.tdt_beam_decoding import BeamBatchedTDTInfer


MEDICAL_WORDS = [
    "myocardial", "pulmonary", "hepatic", "renal", "cerebral", "abdominal",
    "thoracic", "cervical", "lumbar", "sacral", "pericardial", "pleural",
    "peritoneal", "meningeal", "mucosal", "epithelial", "endothelial",
    "endocardial", "myocardium", "pericardium", "pleura", "peritoneum",
    "infarction", "edema", "ischemia", "hemorrhage", "thrombosis", "embolism",
    "necrosis", "fibrosis", "atrophy", "hypertrophy", "inflammation",
    "infection", "obstruction", "perforation", "rupture", "tear", "lesion",
    "mass", "acute", "chronic", "bilateral", "severe", "mild", "recurrent",
    "progressive", "localized", "diffuse", "systemic", "anterior", "posterior",
    "superior", "inferior", "medial", "lateral", "proximal", "distal",
    "resection", "transplantation", "bypass", "catheterization", "endoscopy",
    "angioplasty", "ablation", "biopsy", "aspiration", "incision", "drainage",
    "intubation", "ventilation", "perfusion", "ligation", "anastomosis",
    "heparin", "warfarin", "metformin", "lisinopril", "atorvastatin",
    "amoxicillin", "furosemide", "albuterol", "ibuprofen", "morphine",
    "dopamine", "epinephrine", "norepinephrine", "nitroglycerin", "fentanyl",
    "pneumonia", "sepsis", "meningitis", "appendicitis", "peritonitis",
    "colitis", "pancreatitis", "hepatitis", "nephritis", "dermatitis",
    "bronchitis", "endocarditis", "osteomyelitis", "arthritis", "encephalitis",
]


def generate_phrases(n: int, seed: int = 42) -> list[str]:
    """Generate n unique medical-style multi-word phrases with no repeated words."""
    rng = random.Random(seed)
    phrases: set[str] = set()
    while len(phrases) < n:
        num_words = rng.randint(1, min(4, len(MEDICAL_WORDS)))
        words = rng.sample(MEDICAL_WORDS, num_words)
        phrases.add(" ".join(words))
    return sorted(phrases)


def format_bytes(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes}B"
    if n_bytes < 1024 * 1024:
        return f"{n_bytes / 1024:.1f}KB"
    if n_bytes < 1024 * 1024 * 1024:
        return f"{n_bytes / (1024 * 1024):.1f}MB"
    return f"{n_bytes / (1024 * 1024 * 1024):.2f}GB"


def gpu_mem(device: torch.device) -> int:
    """Current GPU memory allocated in bytes."""
    return torch.cuda.memory_allocated(device)


def gpu_mem_mb(device: torch.device) -> float:
    return gpu_mem(device) / (1024 * 1024)


def make_beam_decoder(
    model: ASRModel,
    biasing_multi_model: GPUBiasingMultiModel,
    beam_size: int = 4,
) -> BeamBatchedTDTInfer:
    """Create a BeamBatchedTDTInfer decoder that uses the given biasing_multi_model."""
    model_cfg = model.to_config_dict()
    durations = list(model_cfg["model_defaults"]["tdt_durations"])
    vocab_size = model.tokenizer.vocab_size
    decoder = BeamBatchedTDTInfer(
        decoder_model=model.decoder,
        joint_model=model.joint,
        durations=durations,
        blank_index=vocab_size,
        beam_size=beam_size,
        score_norm=True,
        return_best_hypothesis=True,
        allow_cuda_graphs=False,
        enable_per_stream_biasing=True,
    )
    decoder.decoding_computer.biasing_multi_model = biasing_multi_model
    return decoder


def load_librispeech_batch(
    num_samples: int,
    device: torch.device,
    hf_cache_dir: str = "/mnt/md0/mlrd/shared_cache/huggingface",
    pad_to_duration: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Load real speech from LibriSpeech test.clean via HF datasets.

    Returns (audio_batch, audio_lengths, avg_duration_s) where audio_batch is
    padded to [batch, max_samples] at 16 kHz.

    If pad_to_duration is set, each sample is zero-padded to exactly that duration
    (in seconds). This simulates production audio lengths regardless of the
    natural segment duration.
    """
    import os

    from datasets import load_dataset

    ds = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="test",
        cache_dir=os.path.join(hf_cache_dir, "datasets"),
        trust_remote_code=True,
    )

    target_samples = int(pad_to_duration * 16000) if pad_to_duration else None

    if pad_to_duration:
        # Use the longest segments first to maximize real speech content
        indexed = []
        for i in range(len(ds)):
            row = ds[i]
            dur = len(row["audio"]["array"]) / row["audio"]["sampling_rate"]
            indexed.append((dur, i))
        indexed.sort(reverse=True)
        indices = [idx for _, idx in indexed[:num_samples]]
    else:
        indices = list(range(num_samples))

    samples = []
    for idx in indices:
        row = ds[idx]
        audio = torch.tensor(row["audio"]["array"], dtype=torch.float32)
        sr = row["audio"]["sampling_rate"]
        if sr != 16000:
            import torchaudio

            audio = torchaudio.functional.resample(audio, sr, 16000)
        if target_samples and audio.shape[0] < target_samples:
            audio = torch.cat([audio, torch.zeros(target_samples - audio.shape[0], dtype=torch.float32)])
        samples.append(audio)

    max_len = max(s.shape[0] for s in samples)
    batch = torch.zeros(len(samples), max_len, dtype=torch.float32, device=device)
    lengths = torch.zeros(len(samples), dtype=torch.long, device=device)
    for i, s in enumerate(samples):
        batch[i, : s.shape[0]] = s.to(device)
        lengths[i] = s.shape[0]

    avg_duration = sum(s.shape[0] for s in samples) / (len(samples) * 16000)
    return batch, lengths, avg_duration


def verify_biasing(
    model: ASRModel,
    tokenizer,
    vocab_size: int,
    device: torch.device,
    beam_size: int,
    hf_cache_dir: str = "/mnt/md0/mlrd/shared_cache/huggingface",
):
    """Verify that per-stream biasing actually changes decoder output.

    Decodes one LibriSpeech sample two ways:
      1. No biasing (enable_per_stream_biasing=False)
      2. Biasing with ground-truth transcript words as keyterms
    If the outputs differ, biasing is taking effect.
    """
    import os

    from datasets import load_dataset

    print("\n=== Biasing Verification ===")
    print(f"  Audio: LibriSpeech test.clean[0]")

    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0

    ds = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="test",
        cache_dir=os.path.join(hf_cache_dir, "datasets"),
    )

    row = ds[0]
    audio = torch.tensor(row["audio"]["array"], dtype=torch.float32)
    sr = row["audio"]["sampling_rate"]
    if sr != 16000:
        import torchaudio

        audio = torchaudio.functional.resample(audio, sr, 16000)
    reference = row["text"].lower().strip()
    audio = audio.to(device)

    encoded, encoded_len = model(
        input_signal=audio.unsqueeze(0),
        input_signal_length=torch.tensor([audio.shape[0]], device=device),
    )

    # No biasing: decoder without per-stream biasing
    model_cfg = model.to_config_dict()
    durations = list(model_cfg["model_defaults"]["tdt_durations"])
    no_bias_decoder = BeamBatchedTDTInfer(
        decoder_model=model.decoder,
        joint_model=model.joint,
        durations=durations,
        blank_index=vocab_size,
        beam_size=beam_size,
        score_norm=True,
        return_best_hypothesis=True,
        allow_cuda_graphs=False,
        enable_per_stream_biasing=False,
    )

    with torch.no_grad(), torch.inference_mode():
        hyp_no = no_bias_decoder(encoder_output=encoded, encoded_lengths=encoded_len)[0]
    text_no = hyp_no[0].text if hyp_no[0].text is not None else tokenizer.ids_to_text(hyp_no[0].y_sequence.tolist())

    # With biasing: register ground-truth words as keyterms
    biasing_multi_model = GPUBiasingMultiModel(vocab_size=vocab_size, use_triton=True)
    biasing_multi_model.to(device)

    keyterms = list(set(reference.split()))
    cfg = BoostingTreeModelConfig(key_phrases_list=keyterms)
    trie = GPUBoostingTreeModel.from_config(cfg, tokenizer=tokenizer)
    model_id = biasing_multi_model.add_model(model=trie, alpha=4.0)

    bias_decoder = make_beam_decoder(model, biasing_multi_model, beam_size=beam_size)
    bias_ids = torch.tensor([model_id], dtype=torch.long, device=device)

    with torch.no_grad(), torch.inference_mode():
        hyp_yes = bias_decoder(encoder_output=encoded, encoded_lengths=encoded_len, multi_biasing_ids=bias_ids)[0]
    text_yes = hyp_yes[0].text if hyp_yes[0].text is not None else tokenizer.ids_to_text(hyp_yes[0].y_sequence.tolist())

    print(f"  Reference:  {reference}")
    print(f"  No bias:    {text_no}")
    print(f"  Biased:     {text_yes}")
    print(f"  Keyterms ({len(keyterms)}):  {keyterms[:20]}{'...' if len(keyterms) > 20 else ''}")
    print(f"  Changed:    {text_no != text_yes}")
    print()

    return text_no != text_yes


def benchmark_memory_capacity(
    model: ASRModel,
    tokenizer,
    vocab_size: int,
    device: torch.device,
    term_count: int,
    max_sessions: int,
    step_sizes: list[int],
    variable_trie_sizes: bool = False,
    min_terms: int = 50,
    max_terms: int = 10000,
):
    """Register increasing numbers of tries and track GPU memory.

    When variable_trie_sizes=True, each trie gets a random term count
    drawn uniformly from [min_terms, max_terms] instead of a fixed term_count.
    """
    print("\n=== Test 1: Memory Capacity ===")
    if variable_trie_sizes:
        print(f"  Terms per trie: variable (random {min_terms}-{max_terms})")
    else:
        print(f"  Terms per trie: {term_count}")
    print(f"  Max sessions: {max_sessions}")
    print()

    biasing_multi_model = GPUBiasingMultiModel(vocab_size=vocab_size, use_triton=True)
    biasing_multi_model.to(device)

    model_mem = gpu_mem(device)
    mm_base_mem = gpu_mem(device) - model_mem
    print(f"  Model GPU mem: {format_bytes(model_mem)}")
    print(f"  Multi-model base (pre-allocated): {format_bytes(mm_base_mem)}")
    print()

    rng = random.Random(123) if variable_trie_sizes else None

    results = []
    results.append({
        "sessions": 0,
        "mm_mem_mb": mm_base_mem / (1024 * 1024),
        "total_mem_gb": gpu_mem(device) / (1024**3),
        "marginal_mb": 0.0,
        "realloc": "base",
        "trie_stats": "",
    })

    num_registered = 0
    prev_mm_mem = gpu_mem(device)
    total_arcs = 0
    total_states = 0

    for step_sessions in step_sizes:
        if step_sessions > max_sessions:
            break

        while num_registered < step_sessions:
            if variable_trie_sizes:
                this_term_count = rng.randint(min_terms, max_terms)
            else:
                this_term_count = term_count

            phrases = generate_phrases(this_term_count, seed=42 + num_registered)
            cfg = BoostingTreeModelConfig(key_phrases_list=phrases)

            before = gpu_mem(device)
            trie = GPUBoostingTreeModel.from_config(cfg, tokenizer=tokenizer)
            model_id = biasing_multi_model.add_model(model=trie, alpha=2.0)
            after = gpu_mem(device)
            num_registered += 1
            total_arcs += trie.num_arcs
            total_states += trie.num_states

            if after - before > 1024 * 1024:
                marginal_mb = (after - prev_mm_mem) / (1024 * 1024)
                avg_terms = total_arcs // max(num_registered, 1)
                results.append({
                    "sessions": num_registered,
                    "mm_mem_mb": (after - model_mem) / (1024 * 1024),
                    "total_mem_gb": after / (1024**3),
                    "marginal_mb": marginal_mb,
                    "realloc": f"YES (+{format_bytes(after - before)})",
                    "trie_stats": f"{trie.num_states:,} st, {trie.num_arcs:,} arcs",
                })
                prev_mm_mem = after

        if num_registered == step_sessions:
            cur = gpu_mem(device)
            marginal_mb = (cur - prev_mm_mem) / (1024 * 1024) if results else 0.0
            avg_terms = total_arcs // max(num_registered, 1) if num_registered > 0 else 0
            results.append({
                "sessions": num_registered,
                "mm_mem_mb": (cur - model_mem) / (1024 * 1024),
                "total_mem_gb": cur / (1024**3),
                "marginal_mb": marginal_mb,
                "realloc": "no" if marginal_mb < 1.0 else "",
                "trie_stats": f"total: {total_arcs:,} arcs, {total_states:,} st",
            })
            prev_mm_mem = cur

            print(
                f"  {num_registered:>5} sessions | "
                f"MM mem: {(cur - model_mem) / (1024 * 1024):>8.1f} MB | "
                f"Total: {cur / (1024**3):>6.2f} GB | "
                f"Marginal: {marginal_mb:>6.1f} MB"
                f"{'  REALLOC' if marginal_mb > 1.0 else ''}"
            )
            if variable_trie_sizes and num_registered > 0:
                print(f"         avg arcs/trie: {total_arcs // num_registered:,}")

    print()
    return results, biasing_multi_model


def benchmark_decode_performance(
    model: ASRModel,
    tokenizer,
    vocab_size: int,
    device: torch.device,
    biasing_multi_model: GPUBiasingMultiModel,
    num_tries: int,
    term_count: int,
    batch_size: int,
    beam_size: int,
    session_counts: list[int],
    pad_to_duration: float | None = None,
):
    """Decode LibriSpeech audio with increasing numbers of concurrent tries to measure performance impact."""
    print("\n=== Test 2: Decode Performance ===")
    print(f"  Batch size: {batch_size}, Beam size: {beam_size}")
    if pad_to_duration:
        print(f"  Audio: LibriSpeech test.clean padded to {pad_to_duration}s")
    else:
        print(f"  Audio: LibriSpeech test.clean (real speech)")
    print(f"  Sessions to test: {session_counts}")
    print()

    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0

    audio_batch, audio_len, avg_duration = load_librispeech_batch(batch_size, device, pad_to_duration=pad_to_duration)
    audio_duration_s = avg_duration

    durations = (audio_len.float() / 16000).tolist()
    print(f"  Audio durations: {[f'{d:.1f}s' for d in durations]}")
    print()

    results = []

    for n_sessions in session_counts:
        if n_sessions > num_tries:
            print(f"  {n_sessions:>5} sessions: skipped (only {num_tries} tries registered)")
            continue

        print(f"  {n_sessions:>5} sessions: ", end="", flush=True)

        multi_biasing_ids = torch.arange(min(n_sessions, batch_size), dtype=torch.long, device=device)
        if n_sessions < batch_size:
            multi_biasing_ids = torch.cat([
                multi_biasing_ids,
                torch.full([batch_size - n_sessions], fill_value=-1, dtype=torch.long, device=device),
            ])

        decoder = make_beam_decoder(model, biasing_multi_model, beam_size=beam_size)

        def run_pipeline():
            with torch.no_grad(), torch.inference_mode():
                encoded, encoded_len = model(input_signal=audio_batch, input_signal_length=audio_len)
                hyps = decoder(
                    encoder_output=encoded,
                    encoded_lengths=encoded_len,
                    multi_biasing_ids=multi_biasing_ids,
                )[0]
            return hyps

        try:
            run_pipeline()
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            run_pipeline()
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            decode_time = t1 - t0
            rtfx = (audio_duration_s * batch_size) / decode_time if decode_time > 0 else 0.0

            peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)
            torch.cuda.reset_peak_memory_stats(device)

            print(f"decode={decode_time * 1000:.0f}ms, RTFx={rtfx:.1f}x, peak_mem={peak_mem:.2f} GB")
            results.append({
                "sessions": n_sessions,
                "decode_time_ms": decode_time * 1000,
                "rtfx": rtfx,
                "peak_mem_gb": peak_mem,
            })
        except torch.cuda.OutOfMemoryError:
            print("OOM")
            results.append({
                "sessions": n_sessions,
                "decode_time_ms": None,
                "rtfx": 0.0,
                "peak_mem_gb": None,
            })
            break

        del decoder
        gc.collect()
        torch.cuda.empty_cache()

    print()
    return results


def benchmark_scaled_batch(
    model: ASRModel,
    tokenizer,
    vocab_size: int,
    device: torch.device,
    biasing_multi_model: GPUBiasingMultiModel,
    num_tries: int,
    term_count: int,
    beam_size: int,
    session_counts: list[int],
    hf_cache_dir: str = "/mnt/md0/mlrd/shared_cache/huggingface",
):
    """Scale batch_size = n_sessions and measure RTFx to find the realtime limit.

    Each session sends audio simultaneously. We decode all N sessions in one batch.
    RTFx = total_audio_duration / decode_time. When RTFx < 1.0, we've exceeded
    the realtime capacity of the GPU.
    """
    import os

    from datasets import load_dataset

    print("\n=== Test 3: Scaled Batch (batch_size = n_sessions) ===")
    print(f"  Beam size: {beam_size}")
    print(f"  Audio: LibriSpeech test.clean (real speech)")
    print(f"  Constraint: RTFx must stay >= 1.0 for realtime serving")
    print(f"  Session counts to test: {session_counts}")
    print()

    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0

    ds = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="test",
        cache_dir=os.path.join(hf_cache_dir, "datasets"),
    )

    max_needed = max(session_counts) if session_counts else 0
    if max_needed > num_tries:
        print(f"  Note: max session count ({max_needed}) > registered tries ({num_tries})")
        print(f"  Sessions beyond {num_tries} will use multi_biasing_ids=-1 (no biasing)")
    print()

    results = []

    for n_sessions in session_counts:
        print(f"  {n_sessions:>4} sessions (batch={n_sessions}): ", end="", flush=True)

        samples = []
        for i in range(n_sessions):
            row = ds[i]
            audio = torch.tensor(row["audio"]["array"], dtype=torch.float32)
            sr = row["audio"]["sampling_rate"]
            if sr != 16000:
                import torchaudio

                audio = torchaudio.functional.resample(audio, sr, 16000)
            samples.append(audio)

        max_len = max(s.shape[0] for s in samples)
        audio_batch = torch.zeros(n_sessions, max_len, dtype=torch.float32, device=device)
        audio_len = torch.zeros(n_sessions, dtype=torch.long, device=device)
        for i, s in enumerate(samples):
            audio_batch[i, : s.shape[0]] = s.to(device)
            audio_len[i] = s.shape[0]

        total_audio_s = audio_len.float().sum().item() / 16000

        try:
            with torch.no_grad(), torch.inference_mode():
                encoded, encoded_len = model(input_signal=audio_batch, input_signal_length=audio_len)

            multi_biasing_ids = torch.arange(min(n_sessions, num_tries), dtype=torch.long, device=device)
            if n_sessions > num_tries:
                multi_biasing_ids = torch.cat([
                    multi_biasing_ids,
                    torch.full([n_sessions - num_tries], fill_value=-1, dtype=torch.long, device=device),
                ])

            decoder = make_beam_decoder(model, biasing_multi_model, beam_size=beam_size)

            def run_decode():
                with torch.no_grad(), torch.inference_mode():
                    hyps = decoder(
                        encoder_output=encoded,
                        encoded_lengths=encoded_len,
                        multi_biasing_ids=multi_biasing_ids,
                    )[0]
                return hyps

            run_decode()
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            run_decode()
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            decode_time = t1 - t0
            rtfx = total_audio_s / decode_time if decode_time > 0 else 0.0

            peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)
            torch.cuda.reset_peak_memory_stats(device)

            status = "OK" if rtfx >= 1.0 else "BELOW REALTIME"
            print(f"decode={decode_time * 1000:.0f}ms, RTFx={rtfx:.1f}x, peak={peak_mem:.2f} GB [{status}]")

            results.append({
                "sessions": n_sessions,
                "batch_size": n_sessions,
                "decode_time_ms": decode_time * 1000,
                "total_audio_s": total_audio_s,
                "rtfx": rtfx,
                "peak_mem_gb": peak_mem,
                "realtime": rtfx >= 1.0,
            })

            if rtfx < 1.0:
                print(f"  *** Realtime limit reached at {n_sessions} sessions (RTFx={rtfx:.1f}x)")
                break

            del decoder
            gc.collect()
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print("OOM")
            results.append({
                "sessions": n_sessions,
                "batch_size": n_sessions,
                "decode_time_ms": None,
                "total_audio_s": total_audio_s,
                "rtfx": 0.0,
                "peak_mem_gb": None,
                "realtime": False,
            })
            break

    print()
    return results


def print_summary(
    mem_results: list[dict],
    decode_results: list[dict],
    scaled_results: list[dict],
    model_name: str,
    device: torch.device,
    term_count: int,
    max_sessions: int,
):
    print()
    print("=" * 100)
    print("Sticky Session Capacity Benchmark")
    print("=" * 100)
    print(f"  Model:          {model_name}")
    print(f"  Device:         {device}")
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        print(f"  GPU:             {prop.name}")
        print(f"  Total GPU mem:   {prop.total_memory / (1024**3):.1f} GB")
    print(f"  Terms per trie:  {term_count}")
    print(f"  Max sessions:    {max_sessions}")
    print()

    print("Memory Capacity:")
    print(f"  {'Sessions':>8}  {'MM Mem':>10}  {'Total GPU':>10}  {'Marginal':>10}  {'Realloc':>20}  {'Trie stats'}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*20}  {'-'*30}")
    for r in mem_results:
        print(
            f"  {r['sessions']:>8}  "
            f"{r['mm_mem_mb']:>8.1f}MB  "
            f"{r['total_mem_gb']:>7.2f}GB  "
            f"{r['marginal_mb']:>8.1f}MB  "
            f"{r['realloc']:>20}  "
            f"{r['trie_stats']}"
        )

    if decode_results:
        baseline = next((r for r in decode_results if r["sessions"] == 0), None)
        baseline_time = baseline["decode_time_ms"] if baseline and baseline["decode_time_ms"] else 0.0
        print()
        print("Decode Performance:")
        print(f"  {'Sessions':>8}  {'Decode Time':>12}  {'RTFx':>8}  {'Peak Mem':>10}  {'vs Baseline':>12}")
        print(f"  {'-'*8}  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*12}")
        for r in decode_results:
            dt = f"{r['decode_time_ms']:.0f}ms" if r["decode_time_ms"] else "OOM"
            pm = f"{r['peak_mem_gb']:.2f}GB" if r["peak_mem_gb"] else "--"
            ratio = f"{r['decode_time_ms'] / baseline_time:.2f}x" if baseline_time and r["decode_time_ms"] else "--"
            print(f"  {r['sessions']:>8}  {dt:>12}  {r['rtfx']:>7.1f}x  {pm:>10}  {ratio:>12}")

    if scaled_results:
        print()
        print("Scaled Batch (batch_size = n_sessions, all sessions decode simultaneously):")
        print(f"  {'Sessions':>8}  {'Batch':>6}  {'Decode Time':>12}  {'Total Audio':>12}  {'RTFx':>8}  {'Peak Mem':>10}  {'Status'}")
        print(f"  {'-'*8}  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*20}")
        for r in scaled_results:
            dt = f"{r['decode_time_ms']:.0f}ms" if r["decode_time_ms"] else "OOM"
            ta = f"{r['total_audio_s']:.1f}s"
            pm = f"{r['peak_mem_gb']:.2f}GB" if r["peak_mem_gb"] else "--"
            status = "REALTIME" if r["realtime"] else "BELOW REALTIME"
            print(f"  {r['sessions']:>8}  {r['batch_size']:>6}  {dt:>12}  {ta:>12}  {r['rtfx']:>7.1f}x  {pm:>10}  {status}")

        realtime_max = max((r["sessions"] for r in scaled_results if r["realtime"]), default=0)
        print(f"  Max realtime sessions: {realtime_max}")

    print()
    print("Capacity estimate:")
    if mem_results:
        last = mem_results[-1]
        model_mem_gb = mem_results[0]["total_mem_gb"] if mem_results[0]["sessions"] == 0 else 0
        if device.type == "cuda":
            prop = torch.cuda.get_device_properties(device)
            total_gb = prop.total_memory / (1024**3)
            used_gb = last["total_mem_gb"]
            available_gb = total_gb - used_gb
            mm_mem_per_session = (last["mm_mem_mb"] - mem_results[0]["mm_mem_mb"]) / max(last["sessions"], 1) / 1024
            if mm_mem_per_session > 0:
                est = int(available_gb / mm_mem_per_session)
            else:
                est = "N/A (within pre-allocation)"
            print(f"  GPU total:        {total_gb:.1f} GB")
            print(f"  Current usage:    {used_gb:.2f} GB ({last['sessions']} sessions)")
            print(f"  Available:        {available_gb:.2f} GB")
            print(f"  Per-session (marginal): {mm_mem_per_session * 1024:.2f} MB")
            print(f"  Estimated max sessions:  {est}")
            print(f"  (Excludes decode working memory — actual capacity will be lower)")
    print("=" * 100)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark sticky session capacity on MIG-partitioned GPUs")
    parser.add_argument("--model", type=str, default="nvidia/parakeet-tdt-0.6b-v2")
    parser.add_argument("--term-count", type=int, default=10000, help="Terms per trie")
    parser.add_argument("--max-sessions", type=int, default=500, help="Maximum number of sessions/tries to register")
    parser.add_argument(
        "--session-steps",
        type=str,
        default="10,50,100,200,500",
        help="Comma-separated session counts for memory checkpoint reporting",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for decode test")
    parser.add_argument("--beam-size", type=int, default=4, help="Beam size for decode test")
    parser.add_argument(
        "--pad-to-duration",
        type=float,
        default=None,
        help="Pad each LibriSpeech sample to this duration (seconds) to simulate production audio lengths",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-decode-test", action="store_true", help="Skip decode performance test")
    parser.add_argument(
        "--decode-session-counts",
        type=str,
        default="0,10,50,100,200,500",
        help="Comma-separated session counts for decode performance test",
    )
    parser.add_argument(
        "--scale-batch",
        action="store_true",
        help="Run scaled batch test (batch_size=n_sessions) to find realtime limit",
    )
    parser.add_argument(
        "--scale-batch-counts",
        type=str,
        default="8,16,32,64,128,256",
        help="Comma-separated session counts for scaled batch test",
    )
    parser.add_argument(
        "--variable-trie-sizes",
        action="store_true",
        help="Use random term counts per trie (uniform from --min-terms to --max-terms)",
    )
    parser.add_argument(
        "--min-terms",
        type=int,
        default=0,
        help="Minimum terms per trie when --variable-trie-sizes is set",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=50000,
        help="Maximum terms per trie when --variable-trie-sizes is set",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    session_steps = [int(x.strip()) for x in args.session_steps.split(",")]
    decode_counts = [int(x.strip()) for x in args.decode_session_counts.split(",")]

    print(f"Device: {device}")
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        print(f"GPU: {prop.name}")
        print(f"Total memory: {prop.total_memory / (1024**3):.1f} GB")
    print(f"Model: {args.model}")
    print(f"Terms per trie: {args.term_count}")
    print(f"Max sessions: {args.max_sessions}")
    print(f"Session steps: {session_steps}")
    print(f"Decode test: {'disabled' if args.no_decode_test else 'enabled'}")
    if not args.no_decode_test:
        audio_desc = f"LibriSpeech padded to {args.pad_to_duration}s" if args.pad_to_duration else "LibriSpeech test.clean"
        print(f"  Batch: {args.batch_size}, Beam: {args.beam_size}, Audio: {audio_desc}")
        print(f"  Decode session counts: {decode_counts}")
    if args.variable_trie_sizes:
        print(f"  Variable trie sizes: {args.min_terms}-{args.max_terms} terms per trie (random)")
    if args.scale_batch:
        scale_counts = [int(x.strip()) for x in args.scale_batch_counts.split(",")]
        print(f"  Scale batch test: enabled, counts: {scale_counts}")
    print()

    print(f"Loading model: {args.model}")
    model = ASRModel.from_pretrained(model_name=args.model, map_location="cpu")
    model.to(device)
    model.eval()
    tokenizer = model.tokenizer
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer: {type(tokenizer).__name__}, vocab_size={vocab_size}")
    print()

    biasing_ok = verify_biasing(model, tokenizer, vocab_size, device, args.beam_size)
    if not biasing_ok:
        print("  WARNING: Biasing did not change the output. Results may not reflect real biasing behavior.")
    print()

    mem_results, biasing_multi_model = benchmark_memory_capacity(
        model, tokenizer, vocab_size, device,
        term_count=args.term_count,
        max_sessions=args.max_sessions,
        step_sizes=session_steps,
        variable_trie_sizes=args.variable_trie_sizes,
        min_terms=args.min_terms,
        max_terms=args.max_terms,
    )

    decode_results = []
    if not args.no_decode_test:
        actual_decode_counts = [c for c in decode_counts if c <= args.max_sessions]
        if 0 not in actual_decode_counts:
            actual_decode_counts = [0] + actual_decode_counts

        decode_results = benchmark_decode_performance(
            model, tokenizer, vocab_size, device,
            biasing_multi_model=biasing_multi_model,
            num_tries=args.max_sessions,
            term_count=args.term_count,
            batch_size=args.batch_size,
            beam_size=args.beam_size,
            session_counts=actual_decode_counts,
            pad_to_duration=args.pad_to_duration,
        )

    scaled_results = []
    if args.scale_batch:
        scale_counts = [int(x.strip()) for x in args.scale_batch_counts.split(",")]
        scaled_results = benchmark_scaled_batch(
            model, tokenizer, vocab_size, device,
            biasing_multi_model=biasing_multi_model,
            num_tries=args.max_sessions,
            term_count=args.term_count,
            beam_size=args.beam_size,
            session_counts=scale_counts,
        )

    print_summary(
        mem_results, decode_results, scaled_results, args.model, device,
        args.term_count, args.max_sessions,
    )


if __name__ == "__main__":
    main()
