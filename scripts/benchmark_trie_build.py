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

"""Benchmark boosting trie build time.

Reproduces the benchmarks from SR-2752 measuring the one-time latency of
compiling a boosting trie from key phrases at stream initialization.

Two stages are timed separately:
  1. from_config: Trie compilation only (tokenization + ContextGraph build
     + GPUBoostingTreeModel construction on CPU)
  2. add_to_multi_model: Full production path (build + register into
     GPUBiasingMultiModel on GPU, including CPU-to-GPU data transfer)

SR-2752 reference results (Lasse Borgholt):
  1000 terms  -> min 195ms, median 205ms, p95 536ms, max 549ms
  10000 terms -> min 1.27s, median 1.56s, p95 1.65s, max 1.69s

Usage:
    export CUDA_VISIBLE_DEVICES=0
    python scripts/benchmark_trie_build.py
    python scripts/benchmark_trie_build.py --term-counts 1000,10000 --iterations 30
    python scripts/benchmark_trie_build.py --model nvidia/parakeet-tdt-0.6b-v2
"""

import argparse
import random
import statistics
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.context_biasing.biasing_multi_model import (
    BiasingRequestItemConfig,
    GPUBiasingMultiModel,
)
from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import (
    BoostingTreeModelConfig,
    GPUBoostingTreeModel,
)


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


def percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile of a list of values."""
    if not data:
        return 0.0
    return float(np.percentile(np.array(data), p))


def format_duration(seconds: float) -> str:
    """Format a duration for table output (ms or s)."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def benchmark_from_config(
    phrases: list[str],
    tokenizer,
    iterations: int,
    warmup: int,
) -> tuple[list[float], int, int]:
    """Benchmark GPUBoostingTreeModel.from_config() (CPU-only trie compilation).

    Returns (times, num_states, num_arcs).
    """
    cfg = BoostingTreeModelConfig(key_phrases_list=phrases)

    sample_model = None
    for _ in range(warmup):
        sample_model = GPUBoostingTreeModel.from_config(cfg, tokenizer=tokenizer)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model = GPUBoostingTreeModel.from_config(cfg, tokenizer=tokenizer)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        sample_model = model

    num_states = sample_model.num_states if sample_model else 0
    num_arcs = sample_model.num_arcs if sample_model else 0
    return times, num_states, num_arcs


def benchmark_add_to_multi_model(
    phrases: list[str],
    tokenizer,
    device: torch.device,
    vocab_size: int,
    alpha: float,
    iterations: int,
    warmup: int,
) -> list[float]:
    """Benchmark the full production path: from_config + add_to_multi_model on GPU.

    This is the end-to-end flow that would happen at stream initialization:
    build the boosting trie, then register it into the GPUBiasingMultiModel
    (which copies trie data from CPU to GPU tensors).
    """
    biasing_multi_model = GPUBiasingMultiModel(vocab_size=vocab_size, use_triton=True)
    biasing_multi_model.to(device)

    cfg = BoostingTreeModelConfig(key_phrases_list=phrases)

    for _ in range(warmup):
        request = BiasingRequestItemConfig(boosting_model_cfg=cfg, boosting_model_alpha=alpha)
        request.add_to_multi_model(tokenizer=tokenizer, biasing_multi_model=biasing_multi_model)
        if request.multi_model_id is not None:
            biasing_multi_model.remove_model(request.multi_model_id)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        request = BiasingRequestItemConfig(boosting_model_cfg=cfg, boosting_model_alpha=alpha)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        request.add_to_multi_model(tokenizer=tokenizer, biasing_multi_model=biasing_multi_model)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if request.multi_model_id is not None:
            biasing_multi_model.remove_model(request.multi_model_id)

    return times


@dataclass
class BenchResult:
    num_terms: int
    stage: str
    times: list[float] = field(default_factory=list)
    num_states: int = 0
    num_arcs: int = 0


def print_results_table(
    results: list[BenchResult],
    model_name: str,
    device: torch.device,
    iterations: int,
    warmup: int,
    seed: int,
):
    print()
    print("=" * 110)
    print("Benchmark: Boosting Trie Build Time")
    print("=" * 110)
    print(f"  Model:      {model_name}")
    print(f"  Device:     {device}")
    print(f"  Iterations: {iterations} ({warmup} warmup)")
    print(f"  Seed:       {seed}")
    print()

    header = (
        f"{'Terms':>6}  {'Stage':<22}  {'Min':>10}  {'Median':>10}  "
        f"{'P95':>10}  {'Max':>10}  {'Trie stats':<24}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        if not r.times:
            continue
        sorted_times = sorted(r.times)
        mn = min(sorted_times)
        med = statistics.median(sorted_times)
        p95 = percentile(sorted_times, 95)
        mx = max(sorted_times)
        trie_info = f"{r.num_states:,} st, {r.num_arcs:,} arcs" if r.num_states else ""
        print(
            f"{r.num_terms:>6}  {r.stage:<22}  {format_duration(mn):>10}  {format_duration(med):>10}  "
            f"{format_duration(p95):>10}  {format_duration(mx):>10}  {trie_info:<24}"
        )

    print()
    print("SR-2752 reference (Lasse Borgholt):")
    print("  1000 terms  -> min 195ms, median 205ms, p95 536ms, max 549ms")
    print("  10000 terms -> min 1.27s, median 1.56s, p95 1.65s, max 1.69s")
    print("=" * 110)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark boosting trie build time")
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/parakeet-tdt-0.6b-v2",
        help="Pretrained model name or path to .nemo file",
    )
    parser.add_argument(
        "--term-counts",
        type=str,
        default="100,500,1000,5000,10000",
        help="Comma-separated list of term counts to benchmark",
    )
    parser.add_argument("--iterations", type=int, default=20, help="Measured iterations per benchmark")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations before timing")
    parser.add_argument(
        "--alpha", type=float, default=2.0, help="Boosting model alpha weight (for add_to_multi_model)"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto-detect)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for phrase generation")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    term_counts = [int(x.strip()) for x in args.term_counts.split(",")]

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Term counts: {term_counts}")
    print(f"Iterations: {args.iterations} ({args.warmup} warmup)")
    print(f"Alpha: {args.alpha}")
    print()

    print(f"Loading model: {args.model}")
    model = ASRModel.from_pretrained(model_name=args.model, map_location="cpu")
    model.eval()
    tokenizer = model.tokenizer
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer: {type(tokenizer).__name__}, vocab_size={vocab_size}")
    print()

    all_results: list[BenchResult] = []

    for n_terms in term_counts:
        print(f"Generating {n_terms} synthetic medical phrases (seed={args.seed})...")
        phrases = generate_phrases(n_terms, seed=args.seed)
        print(f"  Example phrases: {phrases[:3]}")
        print()

        print(f"  [{n_terms} terms] Benchmarking from_config (trie compilation)...")
        fc_times, num_states, num_arcs = benchmark_from_config(
            phrases, tokenizer, iterations=args.iterations, warmup=args.warmup,
        )
        fc_result = BenchResult(
            num_terms=n_terms, stage="from_config", times=fc_times, num_states=num_states, num_arcs=num_arcs,
        )
        all_results.append(fc_result)

        mn = min(fc_times)
        med = statistics.median(fc_times)
        p95 = percentile(sorted(fc_times), 95)
        mx = max(fc_times)
        print(
            f"    min={format_duration(mn)}, median={format_duration(med)}, "
            f"p95={format_duration(p95)}, max={format_duration(mx)}"
        )
        print(f"    trie: {num_states:,} states, {num_arcs:,} arcs")
        print()

        print(f"  [{n_terms} terms] Benchmarking add_to_multi_model (full production path)...")
        am_times = benchmark_add_to_multi_model(
            phrases, tokenizer, device, vocab_size, alpha=args.alpha,
            iterations=args.iterations, warmup=args.warmup,
        )
        am_result = BenchResult(num_terms=n_terms, stage="add_to_multi_model", times=am_times)
        all_results.append(am_result)

        mn = min(am_times)
        med = statistics.median(am_times)
        p95 = percentile(sorted(am_times), 95)
        mx = max(am_times)
        print(
            f"    min={format_duration(mn)}, median={format_duration(med)}, "
            f"p95={format_duration(p95)}, max={format_duration(mx)}"
        )

        gpu_overhead = statistics.median(am_times) - statistics.median(fc_times)
        pct = gpu_overhead / statistics.median(fc_times) * 100 if statistics.median(fc_times) > 0 else 0.0
        print(f"    GPU registration overhead: {format_duration(gpu_overhead)} ({pct:.1f}%)")
        print()

    print_results_table(
        all_results, args.model, device, args.iterations, args.warmup, args.seed,
    )


if __name__ == "__main__":
    main()
