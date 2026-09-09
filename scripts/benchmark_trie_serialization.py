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

"""Benchmark trie serialization round-trip vs. rebuild from phrases.

Answers the question: is it faster to serialize/deserialize a compiled boosting
trie than to rebuild it from key phrases?

Two serialization formats are benchmarked:
  1. state_dict + torch.save to BytesIO (lightweight, closest to a gRPC payload)
  2. .nemo file (tarball with state_dict + YAML config, the existing on-disk format)

For each format we measure:
  - Build time (baseline: from_config from phrases)
  - Serialize time + serialized size
  - Deserialize time (reconstruct GPUBoostingTreeModel)
  - Register time (add to GPUBiasingMultiModel on GPU)
  - Round-trip total (serialize + deserialize + register)

Correctness verification: after deserialization, calls model.advance() on both
the original and deserialized trie with the same random states, asserts the
scores and next_states are identical.

Usage:
    export CUDA_VISIBLE_DEVICES=0
    python scripts/benchmark_trie_serialization.py
    python scripts/benchmark_trie_serialization.py --term-counts 1000,10000 --iterations 10
"""

import argparse
import io
import os
import random
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.context_biasing.biasing_multi_model import GPUBiasingMultiModel
from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import (
    BoostingTreeConfig,
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
    if not data:
        return 0.0
    return float(np.percentile(np.array(data), p))


def format_duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def format_size(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes}B"
    if n_bytes < 1024 * 1024:
        return f"{n_bytes / 1024:.1f}KB"
    return f"{n_bytes / (1024 * 1024):.2f}MB"


def get_model_metadata(model: GPUBoostingTreeModel) -> dict:
    return {
        "num_states": model.num_states,
        "num_arcs": model.num_arcs,
        "max_order": model.max_order,
        "vocab_size": model.vocab_size,
    }


def create_model_from_metadata(metadata: dict, use_triton: bool | None = None) -> GPUBoostingTreeModel:
    cfg = OmegaConf.structured(
        BoostingTreeConfig(
            num_states=metadata["num_states"],
            num_arcs=metadata["num_arcs"],
            max_order=metadata["max_order"],
            vocab_size=metadata["vocab_size"],
            use_triton=use_triton,
        )
    )
    return GPUBoostingTreeModel(cfg=cfg)


def time_fn(fn, iterations: int, warmup: int, sync_cuda: bool = False) -> list[float]:
    for _ in range(warmup):
        fn()
    if sync_cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        if sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def benchmark_build(phrases, tokenizer, iterations, warmup):
    cfg = BoostingTreeModelConfig(key_phrases_list=phrases)
    sample = None

    def run():
        nonlocal sample
        sample = GPUBoostingTreeModel.from_config(cfg, tokenizer=tokenizer)

    times = time_fn(run, iterations, warmup)
    return times, sample


def benchmark_serialize_state_dict(model, iterations, warmup):
    metadata = get_model_metadata(model)

    buf = io.BytesIO()

    def run():
        buf.seek(0)
        buf.truncate()
        sd = model.state_dict()
        torch.save(sd, buf)

    times = time_fn(run, iterations, warmup)
    serialized = buf.getvalue()
    return times, serialized, metadata


def benchmark_serialize_nemo(model, iterations, warmup, tmpdir):
    path = os.path.join(tmpdir, "trie.nemo")

    def run():
        model.save_to(path)

    times = time_fn(run, iterations, warmup)
    nemo_size = os.path.getsize(path)
    return times, nemo_size


def benchmark_deserialize_state_dict(serialized, metadata, iterations, warmup):
    def run():
        buf = io.BytesIO(serialized)
        sd = torch.load(buf, map_location="cpu", weights_only=True)
        model = create_model_from_metadata(metadata)
        model.load_state_dict(sd)
        model._resolve_final()
        return model

    times = time_fn(run, iterations, warmup)
    return times


def benchmark_deserialize_nemo(nemo_path, vocab_size, iterations, warmup):
    def run():
        return GPUBoostingTreeModel.from_nemo(lm_path=nemo_path, vocab_size=vocab_size)

    times = time_fn(run, iterations, warmup)
    return times


def benchmark_register(model, device, vocab_size, iterations, warmup):
    biasing_multi_model = GPUBiasingMultiModel(vocab_size=vocab_size, use_triton=True)
    biasing_multi_model.to(device)

    def run():
        model_id = biasing_multi_model.add_model(model=model, alpha=2.0)
        biasing_multi_model.remove_model(model_id)

    times = time_fn(run, iterations, warmup, sync_cuda=(device.type == "cuda"))
    return times


def verify_correctness(original: GPUBoostingTreeModel, deserialized: GPUBoostingTreeModel, device: torch.device):
    original = original.to(device)
    deserialized = deserialized.to(device)
    original.eval()
    deserialized.eval()

    batch_size = 128
    states = torch.full([batch_size], fill_value=0, dtype=torch.long, device=device)

    with torch.no_grad(), torch.inference_mode():
        scores_orig, next_states_orig = original.advance(states)
        scores_deser, next_states_deser = deserialized.advance(states)

    scores_match = torch.allclose(scores_orig, scores_deser, atol=1e-6, rtol=1e-5)
    states_match = torch.equal(next_states_orig, next_states_deser)

    if not scores_match:
        max_diff = (scores_orig - scores_deser).abs().max().item()
        return False, f"Scores mismatch: max diff = {max_diff}"
    if not states_match:
        mismatches = (next_states_orig != next_states_deser).sum().item()
        return False, f"Next-states mismatch: {mismatches}/{batch_size * original.vocab_size}"
    return True, "Scores and next_states identical"


@dataclass
class StageResult:
    name: str = ""
    times: list[float] = field(default_factory=list)
    size_bytes: int = 0
    extra: str = ""


@dataclass
class TermResult:
    num_terms: int
    build: StageResult = field(default_factory=StageResult)
    ser_sd: StageResult = field(default_factory=StageResult)
    ser_nemo: StageResult = field(default_factory=StageResult)
    deser_sd: StageResult = field(default_factory=StageResult)
    deser_nemo: StageResult = field(default_factory=StageResult)
    register: StageResult = field(default_factory=StageResult)
    correctness_sd: str = ""
    correctness_nemo: str = ""


def print_results_table(
    results: list[TermResult],
    model_name: str,
    device: torch.device,
    iterations: int,
    warmup: int,
    seed: int,
):
    print()
    print("=" * 130)
    print("Benchmark: Trie Serialization Round-Trip vs. Rebuild")
    print("=" * 130)
    print(f"  Model:      {model_name}")
    print(f"  Device:     {device}")
    print(f"  Iterations: {iterations} ({warmup} warmup)")
    print(f"  Seed:       {seed}")
    print()

    for r in results:
        build_med = statistics.median(r.build.times) if r.build.times else 0.0
        print(f"  Terms: {r.num_terms}  (build median: {format_duration(build_med)}, "
              f"trie: {r.build.extra})")
        print()

        header = (
            f"    {'Stage':<32}  {'Median':>10}  {'Min':>10}  {'P95':>10}  {'Max':>10}  "
            f"{'Size':>10}  {'vs Build':>8}"
        )
        print(header)
        print("    " + "-" * (len(header) - 5))

        def print_stage(stage: StageResult, ref: float):
            if not stage.times:
                return
            sorted_t = sorted(stage.times)
            med = statistics.median(sorted_t)
            mn = min(sorted_t)
            p95 = percentile(sorted_t, 95)
            mx = max(sorted_t)
            ratio = med / ref if ref > 0 else 0.0
            sz = format_size(stage.size_bytes) if stage.size_bytes else "--"
            print(
                f"    {stage.name:<32}  {format_duration(med):>10}  {format_duration(mn):>10}  "
                f"{format_duration(p95):>10}  {format_duration(mx):>10}  {sz:>10}  {ratio:>7.3f}x"
            )

        print_stage(r.build, build_med)
        print_stage(r.ser_sd, build_med)
        print_stage(r.ser_nemo, build_med)
        print_stage(r.deser_sd, build_med)
        print_stage(r.deser_nemo, build_med)
        print_stage(r.register, build_med)

        rt_sd_med = (
            statistics.median(r.ser_sd.times) + statistics.median(r.deser_sd.times) + statistics.median(r.register.times)
            if r.ser_sd.times and r.deser_sd.times and r.register.times
            else 0.0
        )
        rt_nemo_med = (
            statistics.median(r.ser_nemo.times) + statistics.median(r.deser_nemo.times) + statistics.median(r.register.times)
            if r.ser_nemo.times and r.deser_nemo.times and r.register.times
            else 0.0
        )

        print()
        print(f"    Round-trip (state_dict):  {format_duration(rt_sd_med):>10}  "
              f"({rt_sd_med / build_med:>6.3f}x of build)" if build_med > 0 else "")
        print(f"    Round-trip (.nemo):       {format_duration(rt_nemo_med):>10}  "
              f"({rt_nemo_med / build_med:>6.3f}x of build)" if build_med > 0 else "")
        print()
        print(f"    Correctness (state_dict): {r.correctness_sd}")
        print(f"    Correctness (.nemo):       {r.correctness_nemo}")
        print()
        print("    " + "=" * 125)

    print()
    print("Summary: round-trip / build ratio at each term count")
    print("    " + "-" * 60)
    for r in results:
        build_med = statistics.median(r.build.times) if r.build.times else 0.0
        rt_sd = (
            statistics.median(r.ser_sd.times) + statistics.median(r.deser_sd.times) + statistics.median(r.register.times)
            if r.ser_sd.times and r.deser_sd.times and r.register.times else 0.0
        )
        rt_nemo = (
            statistics.median(r.ser_nemo.times) + statistics.median(r.deser_nemo.times) + statistics.median(r.register.times)
            if r.ser_nemo.times and r.deser_nemo.times and r.register.times else 0.0
        )
        ratio_sd = rt_sd / build_med if build_med > 0 else 0.0
        ratio_nemo = rt_nemo / build_med if build_med > 0 else 0.0
        print(f"    {r.num_terms:>6} terms  state_dict={ratio_sd:.3f}x  .nemo={ratio_nemo:.3f}x  "
              f"build={format_duration(build_med)}  rt_sd={format_duration(rt_sd)}  rt_nemo={format_duration(rt_nemo)}")
    print("=" * 130)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark trie serialization round-trip vs. rebuild")
    parser.add_argument("--model", type=str, default="nvidia/parakeet-tdt-0.6b-v2")
    parser.add_argument("--term-counts", type=str, default="100,1000,5000,10000")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
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
    print()

    print(f"Loading model: {args.model}")
    model = ASRModel.from_pretrained(model_name=args.model, map_location="cpu")
    model.eval()
    tokenizer = model.tokenizer
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer: {type(tokenizer).__name__}, vocab_size={vocab_size}")
    print()

    all_results: list[TermResult] = []

    with tempfile.TemporaryDirectory(prefix="trie_bench_") as tmpdir:
        for n_terms in term_counts:
            print(f"--- {n_terms} terms ---")
            phrases = generate_phrases(n_terms, seed=args.seed)
            print(f"  Generated {len(phrases)} unique phrases (seed={args.seed})")

            result = TermResult(num_terms=n_terms)

            # 1. Build (baseline)
            print(f"  Benchmarking build (from_config)...")
            build_times, built_model = benchmark_build(phrases, tokenizer, args.iterations, args.warmup)
            result.build = StageResult(
                name="build (from_config)", times=build_times,
                extra=f"{built_model.num_states:,} st, {built_model.num_arcs:,} arcs",
            )
            build_med = statistics.median(build_times)
            print(f"    median: {format_duration(build_med)}, {result.build.extra}")

            # 2. Serialize (state_dict)
            print(f"  Benchmarking serialize (state_dict)...")
            ser_sd_times, ser_sd_bytes, metadata = benchmark_serialize_state_dict(
                built_model, args.iterations, args.warmup,
            )
            result.ser_sd = StageResult(
                name="serialize (state_dict)", times=ser_sd_times, size_bytes=len(ser_sd_bytes),
            )
            print(f"    median: {format_duration(statistics.median(ser_sd_times))}, "
                  f"size: {format_size(len(ser_sd_bytes))}")

            # 3. Serialize (.nemo)
            print(f"  Benchmarking serialize (.nemo)...")
            ser_nemo_times, nemo_size = benchmark_serialize_nemo(
                built_model, args.iterations, args.warmup, tmpdir,
            )
            result.ser_nemo = StageResult(
                name="serialize (.nemo)", times=ser_nemo_times, size_bytes=nemo_size,
            )
            print(f"    median: {format_duration(statistics.median(ser_nemo_times))}, "
                  f"size: {format_size(nemo_size)}")

            # 4. Deserialize (state_dict)
            print(f"  Benchmarking deserialize (state_dict)...")
            deser_sd_times = benchmark_deserialize_state_dict(
                ser_sd_bytes, metadata, args.iterations, args.warmup,
            )
            result.deser_sd = StageResult(name="deserialize (state_dict)", times=deser_sd_times)
            print(f"    median: {format_duration(statistics.median(deser_sd_times))}")

            # 5. Deserialize (.nemo)
            nemo_path = os.path.join(tmpdir, "trie.nemo")
            print(f"  Benchmarking deserialize (.nemo)...")
            deser_nemo_times = benchmark_deserialize_nemo(
                nemo_path, vocab_size, args.iterations, args.warmup,
            )
            result.deser_nemo = StageResult(name="deserialize (.nemo)", times=deser_nemo_times)
            print(f"    median: {format_duration(statistics.median(deser_nemo_times))}")

            # 6. Register to GPUBiasingMultiModel
            print(f"  Benchmarking register to multi-model...")
            deser_model = create_model_from_metadata(metadata)
            buf = io.BytesIO(ser_sd_bytes)
            sd = torch.load(buf, map_location="cpu", weights_only=True)
            deser_model.load_state_dict(sd)
            deser_model._resolve_final()
            register_times = benchmark_register(
                deser_model, device, vocab_size, args.iterations, args.warmup,
            )
            result.register = StageResult(name="register to multi-model", times=register_times)
            print(f"    median: {format_duration(statistics.median(register_times))}")

            # 7. Correctness verification (state_dict)
            print(f"  Verifying correctness (state_dict)...")
            ok, detail = verify_correctness(built_model, deser_model, device)
            result.correctness_sd = f"{'PASS' if ok else 'FAIL'}: {detail}"
            print(f"    {result.correctness_sd}")

            # 8. Correctness verification (.nemo)
            print(f"  Verifying correctness (.nemo)...")
            nemo_model = GPUBoostingTreeModel.from_nemo(lm_path=nemo_path, vocab_size=vocab_size)
            ok_nemo, detail_nemo = verify_correctness(built_model, nemo_model, device)
            result.correctness_nemo = f"{'PASS' if ok_nemo else 'FAIL'}: {detail_nemo}"
            print(f"    {result.correctness_nemo}")

            print()
            all_results.append(result)

    print_results_table(all_results, args.model, device, args.iterations, args.warmup, args.seed)


if __name__ == "__main__":
    main()
