#!/usr/bin/env python
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Benchmark ASR decoding throughput: greedy vs beam search across beam widths.

Measures RTFx (Real-Time Factor) = total_audio_duration / decode_time for each
configuration and prints a comparison table.  Optionally saves results to JSON.

Usage examples
--------------

# Minimal: greedy vs batched beam (malsd_batch) at beam widths 1,2,4,8
python scripts/speech_recognition/benchmark_decoding_throughput.py \\
    --pretrained-name nvidia/parakeet-tdt-0.6b-v2 \\
    --dataset-manifest /path/to/manifest.json

# Sample-level beam search instead of batched
python scripts/speech_recognition/benchmark_decoding_throughput.py \\
    --pretrained-name stt_en_conformer_transducer_small \\
    --dataset-manifest /path/to/manifest.json \\
    --beam-strategy beam \\
    --beam-widths 2,4,8

# CTC model
python scripts/speech_recognition/benchmark_decoding_throughput.py \\
    --pretrained-name nvidia/stt_en_conformer_ctc_large \\
    --dataset-manifest /path/to/manifest.json

# Hybrid model — benchmark the CTC decoder
python scripts/speech_recognition/benchmark_decoding_throughput.py \\
    --pretrained-name stt_en_fastconformer_hybrid_large_pc \\
    --dataset-manifest /path/to/manifest.json \\
    --decoder-type ctc

# Save results to JSON
python scripts/speech_recognition/benchmark_decoding_throughput.py \\
    --pretrained-name nvidia/parakeet-tdt-0.6b-v2 \\
    --dataset-manifest /path/to/manifest.json \\
    --output-json results.json
"""

import argparse
import gc
import glob
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.submodules.multitask_decoding import MultiTaskDecoding
from nemo.collections.asr.parts.utils.transcribe_utils import get_inference_dtype
from nemo.utils import logging
from nemo.utils.timers import SimpleTimer

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    """Throughput result for a single decoding configuration."""

    config_name: str
    strategy: str
    beam_width: Optional[int]
    search_type: str
    decode_time_s: float
    decode_time_std_s: float
    rtfx: float
    rtfx_std: float
    speedup_vs_greedy: float


# ---------------------------------------------------------------------------
# Model-type detection
# ---------------------------------------------------------------------------


def detect_model_type(model: ASRModel) -> str:
    """Return ``"hybrid"``, ``"rnnt"``, or ``"ctc"``.

    Uses attribute presence rather than ``isinstance`` to avoid importing
    every concrete model class and to remain forward-compatible with new
    subclasses.
    """
    # Hybrid models expose both a ``decoding`` (RNNT) and a ``ctc_decoding`` attribute.
    if hasattr(model, "ctc_decoding"):
        return "hybrid"
    # Transducer models (RNNT, TDT, multiblank) have a ``joint`` module.
    if hasattr(model, "joint"):
        return "rnnt"
    return "ctc"


def get_default_beam_strategy(model_type: str, decoder_type: str) -> str:
    """Return the recommended batched beam-search strategy for the model type."""
    if decoder_type == "ctc":
        return "beam_batch"
    return "malsd_batch"


def get_valid_beam_strategies(model_type: str, decoder_type: str) -> List[str]:
    """Return the list of beam-search strategies valid for this model type."""
    if decoder_type == "ctc":
        return ["beam", "beam_batch"]
    return ["beam", "malsd_batch", "maes_batch"]


# ---------------------------------------------------------------------------
# Decoding-config construction
# ---------------------------------------------------------------------------


def _get_current_decoding_cfg(model: ASRModel, decoder_type: str) -> Any:
    """Retrieve the current decoding OmegaConf from the model."""
    if decoder_type == "ctc" and hasattr(model, "ctc_decoding"):
        return model.cfg.aux_ctc.decoding
    return model.cfg.decoding


def build_decoding_cfg(
    model: ASRModel,
    strategy: str,
    beam_width: Optional[int],
    decoder_type: str,
) -> OmegaConf:
    """Build a decoding config that preserves existing settings and overrides only strategy / beam width."""
    current = _get_current_decoding_cfg(model, decoder_type)
    cfg_dict = OmegaConf.to_container(current, resolve=True)

    cfg_dict["strategy"] = strategy

    if beam_width is not None and strategy not in ("greedy", "greedy_batch"):
        if "beam" not in cfg_dict or cfg_dict["beam"] is None:
            cfg_dict["beam"] = {}
        cfg_dict["beam"]["beam_size"] = beam_width

    # Disable alignment / timestamp / confidence overhead during benchmarking.
    cfg_dict["preserve_alignments"] = False
    cfg_dict["compute_timestamps"] = False
    if "confidence_cfg" in cfg_dict and cfg_dict["confidence_cfg"] is not None:
        cfg_dict["confidence_cfg"]["preserve_frame_confidence"] = False

    # transcribe_speech.py sets this to -1 for RNNT decoding.
    if "fused_batch_size" in cfg_dict:
        cfg_dict["fused_batch_size"] = -1

    return OmegaConf.create(cfg_dict)


def apply_decoding_strategy(
    model: ASRModel,
    strategy: str,
    beam_width: Optional[int],
    decoder_type: str,
) -> str:
    """Switch the model's decoding strategy and return a human-readable config name."""
    cfg = build_decoding_cfg(model, strategy, beam_width, decoder_type)

    if hasattr(model, "ctc_decoding"):
        model.change_decoding_strategy(cfg, decoder_type=decoder_type, verbose=False)
    else:
        model.change_decoding_strategy(cfg, verbose=False)

    if strategy in ("greedy", "greedy_batch"):
        return "greedy_batch"
    if beam_width is not None:
        return f"{strategy}_b{beam_width}"
    return strategy


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def load_audio_paths_and_duration(
    dataset_manifest: Optional[str],
    audio_dir: Optional[str],
    audio_key: str,
) -> tuple:
    """Return (audio_paths, total_duration_s) from a manifest or audio directory.

    When loading from an audio directory, durations are read from the audio files
    using the torchaudio backend.
    """
    audio_paths: List[str] = []
    total_duration = 0.0

    if dataset_manifest is not None:
        with open(dataset_manifest, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                audio_paths.append(item[audio_key])
                if "duration" in item:
                    total_duration += float(item["duration"])

        if total_duration == 0.0:
            logging.info("Manifest lacks 'duration' fields — computing from audio files.")
            total_duration = _compute_total_duration_from_files(audio_paths)
    else:
        extensions = ("*.wav", "*.flac", "*.mp3", "*.ogg")
        for ext in extensions:
            audio_paths.extend(glob.glob(os.path.join(audio_dir, ext)))
        audio_paths = sorted(audio_paths)
        if not audio_paths:
            raise ValueError(f"No audio files found in {audio_dir}")
        total_duration = _compute_total_duration_from_files(audio_paths)

    return audio_paths, total_duration


def _compute_total_duration_from_files(audio_paths: List[str]) -> float:
    """Sum audio durations by reading file metadata (no full decode)."""
    total = 0.0
    try:
        import torchaudio

        for path in audio_paths:
            info = torchaudio.info(path)
            total += info.num_frames / info.sample_rate
    except Exception:
        logging.warning("torchaudio not available — duration will be estimated from file sizes.")
        for path in audio_paths:
            total += os.path.getsize(path) / (16000 * 2)  # rough estimate: 16kHz mono 16-bit
    return total


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


def run_timed_transcription(
    model: ASRModel,
    audio_paths: List[str],
    batch_size: int,
    warmup_steps: int,
    run_steps: int,
    device: torch.device,
) -> np.ndarray:
    """Run timed transcription and return an array of per-run wall-clock seconds."""
    timer = SimpleTimer()
    measurements: List[float] = []

    for step in range(warmup_steps + run_steps):
        is_warmup = step < warmup_steps
        label = f"warmup {step + 1}/{warmup_steps}" if is_warmup else f"run {step - warmup_steps + 1}/{run_steps}"

        timer.reset()
        timer.start(device=device)
        model.transcribe(audio=audio_paths, batch_size=batch_size, verbose=False)
        timer.stop(device=device)

        elapsed = timer.total_sec()
        logging.info(f"    [{label}] {elapsed:.3f}s")

        if not is_warmup:
            measurements.append(elapsed)

    return np.array(measurements)


def cleanup_between_runs():
    """Release GPU memory between benchmark configurations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------


def print_results_table(results: List[BenchResult], total_duration: float):
    """Print an aligned comparison table to stdout."""
    header = f"{'Strategy':<18} {'Beam':>5} {'Decode Time (s)':>18} {'RTFx':>10} {'vs Greedy':>10}"
    sep = "-" * len(header)
    print(f"\n{'':=^{len(header)}}")
    print(f"{'Decoding Throughput Benchmark':^{len(header)}}")
    print(f"{'Total audio: {:.1f}s | Batch runs: mean ± std':^{len(header)}}".format(total_duration))
    print(f"{'':=^{len(header)}}")
    print(header)
    print(sep)

    for r in results:
        beam_str = "-" if r.beam_width is None else str(r.beam_width)
        time_str = f"{r.decode_time_s:.3f} ± {r.decode_time_std_s:.3f}"
        rtfx_str = f"{r.rtfx:.1f}" + (f" ± {r.rtfx_std:.1f}" if r.rtfx_std > 0 else "")
        speedup_str = f"{r.speedup_vs_greedy:.2f}x"
        print(f"{r.config_name:<18} {beam_str:>5} {time_str:>18} {rtfx_str:>10} {speedup_str:>10}")

    print(sep)
    print()


def save_results_json(results: List[BenchResult], output_path: str):
    """Save benchmark results as JSON."""
    data = [asdict(r) for r in results]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ASR decoding throughput: greedy vs beam search across beam widths.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--pretrained-name", type=str, help="Pretrained model name from NGC/NeMo.")
    model_group.add_argument("--model-path", type=str, help="Path to a .nemo checkpoint file.")

    # Data
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--dataset-manifest", type=str, help="Path to a JSON manifest file.")
    data_group.add_argument("--audio-dir", type=str, help="Directory containing audio files.")

    # Beam search
    parser.add_argument(
        "--beam-widths",
        type=str,
        default="1,2,4,8",
        help="Comma-separated beam widths to benchmark (default: 1,2,4,8).",
    )
    parser.add_argument(
        "--beam-strategy",
        type=str,
        default=None,
        help=(
            "Beam search strategy. "
            "RNNT/TDT: 'beam' (sample-level), 'malsd_batch', 'maes_batch' (default: malsd_batch). "
            "CTC: 'beam' (sample-level), 'beam_batch' (default: beam_batch)."
        ),
    )

    # Hybrid
    parser.add_argument(
        "--decoder-type",
        type=str,
        default=None,
        choices=["rnnt", "ctc"],
        help="For hybrid models: which decoder to benchmark (default: rnnt).",
    )

    # Inference
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference (default: 32).")
    parser.add_argument("--warmup-steps", type=int, default=1, help="Warmup runs before timing (default: 1).")
    parser.add_argument("--run-steps", type=int, default=3, help="Timed runs for averaging (default: 3).")
    parser.add_argument(
        "--compute-dtype",
        type=str,
        default=None,
        choices=["float32", "float16", "bfloat16"],
        help="Compute dtype (default: auto-select best for device).",
    )
    parser.add_argument("--cuda", type=int, default=None, help="CUDA device index (default: 0 if available).")

    # Output
    parser.add_argument("--output-json", type=str, default=None, help="Path to save results as JSON.")
    parser.add_argument("--audio-key", type=str, default="audio_filepath", help="Manifest key for audio path.")

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Device setup ---
    if args.cuda is not None:
        device = torch.device(f"cuda:{args.cuda}")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    logging.info(f"Device: {device}")
    torch.set_float32_matmul_precision("high")

    # --- Load model ---
    map_location = device
    if args.pretrained_name:
        logging.info(f"Loading pretrained model: {args.pretrained_name}")
        model = ASRModel.from_pretrained(model_name=args.pretrained_name, map_location=map_location)
    else:
        logging.info(f"Loading model from: {args.model_path}")
        model = ASRModel.restore_from(args.model_path, map_location=map_location)

    model.eval()

    # --- Compute dtype ---
    compute_dtype = get_inference_dtype(args.compute_dtype, device)
    if compute_dtype != torch.float32:
        model.to(compute_dtype)
    logging.info(f"Compute dtype: {compute_dtype}")

    # --- Detect model type ---
    model_type = detect_model_type(model)
    decoder_type = args.decoder_type or ("rnnt" if model_type == "hybrid" else model_type)

    if model_type == "hybrid" and args.decoder_type is None:
        logging.info("Hybrid model detected — defaulting to RNNT decoder. Use --decoder-type ctc to benchmark CTC.")

    logging.info(f"Model type: {model_type} | Decoder: {decoder_type}")

    # --- Check for MultiTaskDecoding (AED) — not supported in this script ---
    if hasattr(model, "decoding") and isinstance(model.decoding, MultiTaskDecoding):
        logging.error(
            "AED / multitask decoding is not supported by this benchmark script. "
            "Use transcribe_speech.py with calculate_rtfx=True instead."
        )
        sys.exit(1)

    # --- Determine beam strategy ---
    if args.beam_strategy is None:
        beam_strategy = get_default_beam_strategy(model_type, decoder_type)
    else:
        beam_strategy = args.beam_strategy

    valid_strategies = get_valid_beam_strategies(model_type, decoder_type)
    if beam_strategy not in valid_strategies:
        logging.error(
            f"Beam strategy '{beam_strategy}' is not valid for {decoder_type} models. "
            f"Valid strategies: {valid_strategies}"
        )
        sys.exit(1)

    beam_widths = [int(x) for x in args.beam_widths.split(",")]
    logging.info(f"Beam strategy: {beam_strategy} | Beam widths: {beam_widths}")

    # --- Load audio ---
    audio_paths, total_duration = load_audio_paths_and_duration(
        dataset_manifest=args.dataset_manifest,
        audio_dir=args.audio_dir,
        audio_key=args.audio_key,
    )
    logging.info(f"Audio files: {len(audio_paths)} | Total duration: {total_duration:.2f}s")

    # --- Build benchmark configurations ---
    # Each entry: (config_name, strategy, beam_width)
    configs: List[tuple] = [("greedy_batch", "greedy_batch", None)]
    for bw in beam_widths:
        configs.append((f"{beam_strategy}_b{bw}", beam_strategy, bw))

    # --- Run benchmarks ---
    results: List[BenchResult] = []
    greedy_rtfx: Optional[float] = None

    for config_name, strategy, beam_width in configs:
        logging.info(f"\n{'─' * 60}")
        logging.info(f"Benchmarking: {config_name}")
        logging.info(f"  strategy={strategy}, beam_width={beam_width}")

        apply_decoding_strategy(model, strategy, beam_width, decoder_type)
        cleanup_between_runs()

        measurements = run_timed_transcription(
            model=model,
            audio_paths=audio_paths,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            run_steps=args.run_steps,
            device=device,
        )

        mean_time = float(measurements.mean())
        std_time = float(measurements.std()) if len(measurements) > 1 else 0.0
        rtfx_values = total_duration / measurements
        mean_rtfx = float(rtfx_values.mean())
        std_rtfx = float(rtfx_values.std()) if len(measurements) > 1 else 0.0

        if greedy_rtfx is None:
            greedy_rtfx = mean_rtfx
        speedup = mean_rtfx / greedy_rtfx if greedy_rtfx > 0 else 0.0

        result = BenchResult(
            config_name=config_name,
            strategy=strategy,
            beam_width=beam_width,
            search_type=beam_strategy if beam_width is not None else "greedy_batch",
            decode_time_s=mean_time,
            decode_time_std_s=std_time,
            rtfx=mean_rtfx,
            rtfx_std=std_rtfx,
            speedup_vs_greedy=speedup,
        )
        results.append(result)

        logging.info(f"  Mean time: {mean_time:.3f}s | RTFx: {mean_rtfx:.1f} | vs greedy: {speedup:.2f}x")

    # --- Print results ---
    print_results_table(results, total_duration)

    # --- Save JSON ---
    if args.output_json:
        save_results_json(results, args.output_json)


if __name__ == "__main__":
    main()
