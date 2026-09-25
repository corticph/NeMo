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

"""
Evaluate keyterm (context) biasing with greedy and/or beam-search decoding on the MedRad radiology dictation
dataset, reproducing the keyterm-biasing experiment of arXiv:2605.16545v2 ("Symphony for Speech-to-Text",
Sections 3.1 and 4.4).

The model is biased with a single global vocabulary built from the union of the `medical_terms` column entries
of the full `radai_en` split, decoded with greedy decoding and/or `malsd_batch` beam search with GPU
phrase-boosting (GPU-PB) shallow fusion at several biasing strengths (`boosting_tree_alpha`), and scored with
keyterm recall and precision (R_med / P_med) computed from a word-level Levenshtein alignment between reference
and hypothesis, plus WER. The unbiased baseline (alpha=0) uses the same decoding strategy without a boosting
tree, so the only variable across runs is the biasing strength. For every strength, the script reports the
relative reduction in medical-term false-negative rate (rFNR_med) with respect to the unbiased baseline run and
the medical-term precision under biasing, to verify that biasing does not introduce spurious medical terms.

Keyterm counting follows the paper's definitions (Section 3.1): TP is a vocabulary-term occurrence in the
reference whose aligned hypothesis tokens form the same term; FN is a reference occurrence whose aligned tokens
are missing or form a different term; FP is a hypothesis occurrence whose aligned reference tokens are missing or
form a different term. Matching is case-insensitive and punctuation-insensitive, and multi-word terms are matched
at the term level.

Data loading matches the MedRad dataset used by the paper: the `radai_en` dataset (config `en`, default split
`test_spoken`) is read from the local HuggingFace datasets cache (columns: `audio`, `id`, `transcript`,
`medical_terms`). Audio for the selected subset is written as WAV files under `<out_dir>/audio` and a NeMo-style
JSON manifest is produced, so a run can be repeated with `dataset_manifest` without re-preparing data.

USAGE

python eval_keyterm_biasing_radai.py \
    out_dir=<output folder> \
    hf_datasets_cache=<path to the HuggingFace datasets cache> \
    pretrained_name=nvidia/parakeet-tdt-0.6b-v2 \
    n_examples=500 \
    alphas=[0.0,0.5,1.0,1.5,2.0,3.0] \
    strategies=[greedy,beam] \
    beam_sizes=[1,2,4,5,8]

`strategies` selects the decoding configurations to evaluate: `greedy` runs batched greedy decoding
(`greedy_batch`, the greedy strategy that supports boosting for TDT models), `beam` runs `{beam_strategy}`
beam search at each size in `beam_sizes`. The two can be combined in one run; rFNR is baselined within each
decoding configuration independently.

The biasing strength used by the paper is not reported, hence the sweep over `alphas`. After inspecting
`<out_dir>/results.csv`, re-run on the full split at the chosen strength, e.g.

python eval_keyterm_biasing_radai.py out_dir=<output folder> hf_datasets_cache=<cache> n_examples=0 alphas=[1.0]

Use `n_examples=0` to evaluate the full split, and `dataset_manifest=<path>` to reuse a previously prepared
manifest (skips data preparation; the biasing vocabulary is then taken from the manifest itself).

To match the greedy-vs-beam-search throughput benchmark (scripts/speech_recognition/benchmark_decoding_throughput.py),
runs use batch_size=8 on the 10 GB A100 MIG partition with bfloat16 compute:

CUDA_VISIBLE_DEVICES=MIG-b9e23c79-55f7-5638-823d-56a0a9b84b09 \
python eval_keyterm_biasing_radai.py \
    out_dir=<output folder> \
    hf_datasets_cache=<path to the HuggingFace datasets cache> \
    n_examples=500 \
    batch_size=8 \
    strategies=[greedy,beam] \
    beam_sizes=[1,2,4,5,8] \
    compute_dtype=bfloat16
"""

import ast
import copy
import csv
import glob
import json
import os
import re
from dataclasses import dataclass, field

import soundfile as sf
import torch
from datasets import Dataset
from omegaconf import MISSING, OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import BoostingTreeModelConfig
from nemo.collections.asr.parts.utils.transcribe_utils import get_inference_dtype
from nemo.core.config import hydra_runner
from nemo.utils import logging


@dataclass
class RadaiKeytermBiasingConfig:
    out_dir: str = MISSING
    model_path: str | None = None
    pretrained_name: str = "nvidia/parakeet-tdt-0.6b-v2"
    split: str = "test_spoken"
    hf_datasets_cache: str | None = None
    dataset_manifest: str | None = None
    n_examples: int = 500
    seed: int = 1234
    alphas: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    strategies: list[str] = field(default_factory=lambda: ["beam"])
    beam_sizes: list[int] = field(default_factory=lambda: [4])
    batch_size: int = 8
    beam_strategy: str = "malsd_batch"
    compute_dtype: str = "bfloat16"
    device: str | None = None
    context_score: float = 1.0
    depth_scaling: float = 2.0
    bpe_mode: str = "case_insensitive"
    text_column: str = "transcript"
    medical_terms_column: str = "medical_terms"
    id_column: str = "id"


@hydra_runner(config_name="RadaiKeytermBiasingConfig", schema=RadaiKeytermBiasingConfig)
def main(cfg: RadaiKeytermBiasingConfig):
    device = (
        torch.device(cfg.device)
        if cfg.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    os.makedirs(cfg.out_dir, exist_ok=True)

    manifest_path = cfg.dataset_manifest or os.path.join(cfg.out_dir, "manifest.json")
    if cfg.dataset_manifest is None:
        dataset = _load_radai_split(cfg.split, cfg.hf_datasets_cache)
        vocab = _build_vocab(dataset[cfg.medical_terms_column])
        subset = _select_subset(dataset, cfg.n_examples, cfg.seed)
        records = _write_audio_and_collect(subset, cfg, os.path.join(cfg.out_dir, "audio"))
        _write_jsonl(manifest_path, records)
        logging.info(
            f"Loaded split `{cfg.split}` with {len(dataset)} utterances; biasing vocabulary size {len(vocab)}"
        )
    else:
        records = _read_jsonl(manifest_path)
        vocab = _build_vocab([record["medical_terms"] for record in records])
        logging.info(f"Loaded {len(records)} utterances from manifest; biasing vocabulary size {len(vocab)}")

    vocab_path = os.path.join(cfg.out_dir, "key_terms.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for term in vocab:
            f.write(term + "\n")

    asr_model = _load_model(cfg, device)
    base_decoding_cfg = copy.deepcopy(asr_model.cfg.decoding)

    results = []
    run_configs = []
    for strategy in cfg.strategies:
        if strategy == "greedy":
            run_configs.append(("greedy", None))
        elif strategy == "beam":
            run_configs.extend(("beam", beam_size) for beam_size in cfg.beam_sizes)
        else:
            raise ValueError(f"Unknown strategy `{strategy}`; expected `beam` or `greedy`")

    for mode, beam_size in run_configs:
        for alpha in cfg.alphas:
            if mode == "greedy":
                logging.info(f"Decoding with greedy decoding boosting_tree_alpha={alpha}")
            else:
                logging.info(f"Decoding with {cfg.beam_strategy} beam_size={beam_size} boosting_tree_alpha={alpha}")
            asr_model.change_decoding_strategy(
                _make_decoding_cfg(base_decoding_cfg, vocab, alpha, mode, beam_size, cfg)
            )
            texts = asr_model.transcribe(
                audio=[record["audio_filepath"] for record in records], batch_size=cfg.batch_size
            )
            for record, text in zip(records, texts):
                record["pred_text"] = _pred_text(text)
            suffix = "" if beam_size is None else beam_size
            _write_jsonl(os.path.join(cfg.out_dir, f"preds_{mode}{suffix}_alpha_{alpha}.jsonl"), records)
            metrics = compute_keyterm_biasing_metrics(records, vocab)
            metrics["alpha"] = alpha
            metrics["mode"] = mode
            metrics["beam_size"] = beam_size
            results.append(metrics)
            finalize_rfnr(results)
            label = "greedy" if mode == "greedy" else f"beam_size={beam_size}"
            logging.info(
                f"{label} alpha={alpha}: WER={metrics['wer_pct']:.2f}% R_med={metrics['R_med_pct']:.2f}% "
                f"P_med={metrics['P_med_pct']:.2f}% rFNR_med={metrics['rFNR_med_pct']:.2f}%"
            )

    results_path = os.path.join(cfg.out_dir, "results.csv")
    _write_results(results, results_path)
    _print_summary(results, results_path)
    logging.info("Done!")


def compute_keyterm_biasing_metrics(records: list[dict], vocab: list[str]) -> dict:
    """
    Compute WER and keyterm recall/precision (R_med, P_med) per arXiv:2605.16545v2 Section 3.1, over one
    global keyterm vocabulary `vocab`, from word-level Levenshtein alignments of every record.

    TP: a vocabulary-term occurrence in the reference whose aligned hypothesis tokens form the same term.
    FN: a vocabulary-term occurrence in the reference whose aligned hypothesis tokens are missing or form a
    different term (recall failure).
    FP: a vocabulary-term occurrence in the hypothesis whose aligned reference tokens are missing or form a
    different term (precision failure).

    Returns a dict with counts (tp, fn, fp, num_ref_terms, num_hyp_terms), rates (wer_pct, R_med_pct,
    P_med_pct, FDR_med_pct, FNR_med_pct) and word totals (subs, dels, ins, num_ref_words).
    """
    term_set = {_term_tokens(term) for term in vocab}
    tp = fn = fp = sub = dele = ins = num_ref_terms = num_hyp_terms = 0
    num_ref_words = 0
    for record in records:
        ref_tokens = _tokenize(record["text"])
        hyp_tokens = _tokenize(record["pred_text"])
        alignment = _levenshtein_align(ref_tokens, hyp_tokens)

        num_ref_words += len(ref_tokens)
        for ref_idx, hyp_idx in alignment:
            if ref_idx is None:
                ins += 1
            elif hyp_idx is None:
                dele += 1
            elif ref_tokens[ref_idx] != hyp_tokens[hyp_idx]:
                sub += 1

        ref_to_hyp = {ref_idx: hyp_idx for ref_idx, hyp_idx in alignment if ref_idx is not None}
        hyp_to_ref = {hyp_idx: ref_idx for ref_idx, hyp_idx in alignment if hyp_idx is not None}

        for term in term_set:
            for start, end in _find_occurrences(ref_tokens, term):
                num_ref_terms += 1
                aligned = [hyp_tokens[ref_to_hyp[idx]] for idx in range(start, end) if ref_to_hyp.get(idx) is not None]
                if aligned == list(term):
                    tp += 1
                else:
                    fn += 1
            for start, end in _find_occurrences(hyp_tokens, term):
                num_hyp_terms += 1
                aligned = [ref_tokens[hyp_to_ref[idx]] for idx in range(start, end) if hyp_to_ref.get(idx) is not None]
                if len(aligned) != len(term) or aligned != list(term):
                    fp += 1

    num_errors = sub + dele + ins
    wer = num_errors / num_ref_words if num_ref_words else 0.0
    R_med = tp / (tp + fn) if (tp + fn) else float("nan")
    P_med = tp / (tp + fp) if (tp + fp) else float("nan")
    return {
        "alpha": None,
        "wer_pct": 100.0 * wer,
        "R_med_pct": 100.0 * R_med,
        "P_med_pct": 100.0 * P_med,
        "FDR_med_pct": 100.0 * (1.0 - P_med),
        "FNR_med_pct": 100.0 * (1.0 - R_med),
        "rFNR_med_pct": float("nan"),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "num_ref_terms": num_ref_terms,
        "num_hyp_terms": num_hyp_terms,
        "subs": sub,
        "dels": dele,
        "ins": ins,
        "num_ref_words": num_ref_words,
    }


def finalize_rfnr(results: list[dict]) -> list[dict]:
    """
    Add the relative FNR reduction rFNR_med(α) = (FNR_unbiased - FNR_biased) / FNR_unbiased (Eq. 3 of the paper)
    to every result, using the alpha=0 (unbiased) entry of the same decoding strategy (greedy, or beam search
    with the same beam size) as the baseline. When several decoding configurations are evaluated, rFNR is
    computed independently within each configuration group.
    """
    for result in results:
        result.setdefault("rFNR_med_pct", float("nan"))
    group_keys = []
    for result in results:
        key = (result.get("mode", "beam"), result.get("beam_size"))
        if key not in group_keys:
            group_keys.append(key)
    for mode, beam_size in group_keys:
        group = [
            result for result in results if result.get("mode", "beam") == mode and result.get("beam_size") == beam_size
        ]
        unbiased = next((result for result in group if result["alpha"] == 0.0), group[0] if group else None)
        if unbiased is None:
            continue
        fnr_unbiased = unbiased["FNR_med_pct"]
        if fnr_unbiased <= 0.0:
            continue
        for result in group:
            result["rFNR_med_pct"] = 100.0 * (fnr_unbiased - result["FNR_med_pct"]) / fnr_unbiased
    return results


def _levenshtein_align(ref_tokens: list[str], hyp_tokens: list[str]) -> list[tuple[int | None, int | None]]:
    """
    Word-level Levenshtein alignment. Returns one (ref_index, hyp_index) pair per alignment column, with the
    tokens at those indices: (ref_index, hyp_index) for matches and substitutions, (ref_index, None) for
    deletions and (None, hyp_index) for insertions. Ties prefer substitutions, then deletions.
    """
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ref_tok = ref_tokens[i - 1]
        row, prev_row = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev_row[j - 1] + (ref_tok != hyp_tokens[j - 1]), prev_row[j] + 1, row[j - 1] + 1)

    alignment: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (ref_tokens[i - 1] != hyp_tokens[j - 1]):
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((i - 1, None))
            i -= 1
        else:
            alignment.append((None, j - 1))
            j -= 1
    alignment.reverse()
    return alignment


_NON_WORD_CHARS = re.compile(r"[^a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation (apostrophes kept) and split into word tokens."""
    return _NON_WORD_CHARS.sub(" ", text.lower()).split()


def _term_tokens(term: str) -> tuple[str, ...]:
    return tuple(_tokenize(term))


def _find_occurrences(tokens: list[str], term: tuple[str, ...]) -> list[tuple[int, int]]:
    if not term or len(term) > len(tokens):
        return []
    return [(i, i + len(term)) for i in range(len(tokens) - len(term) + 1) if tuple(tokens[i : i + len(term)]) == term]


def _build_vocab(medical_terms_columns) -> list[str]:
    vocab = {term for value in medical_terms_columns for term in _normalize_medical_terms(value)}
    return sorted(vocab)


def _normalize_medical_terms(value) -> list[str]:
    """
    Normalize a `medical_terms` value into a flat list of terms. Some rows of the `radai_en` dataset store the
    terms as a Python list repr string (e.g. "['term1', 'term2']") inside the sequence; parse those.
    """
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    terms = []
    for item in items:
        if not isinstance(item, str):
            terms.append(item)
            continue
        stripped = item.strip()
        if stripped.startswith("["):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                terms.append(item)
                continue
            if isinstance(parsed, (list, tuple)):
                terms.extend(parsed)
            else:
                terms.append(parsed)
        else:
            terms.append(item)
    return [str(term) for term in terms if term]


def _load_radai_split(split: str, hf_datasets_cache: str | None) -> Dataset:
    cache_dir = hf_datasets_cache or os.environ.get(
        "HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets")
    )
    candidates = glob.glob(os.path.join(cache_dir, "radai_en", "**", f"*{split}.arrow"), recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"No cached `radai_en` split arrow found at `{os.path.join(cache_dir, 'radai_en', '**', '*' + split + '.arrow')}`. "
            "Set `hf_datasets_cache` to the HuggingFace datasets cache containing the `radai_en` dataset."
        )
    path = max(candidates, key=os.path.getsize)
    return Dataset.from_file(path)


def _select_subset(dataset: Dataset, n_examples: int, seed: int) -> Dataset:
    if n_examples <= 0 or n_examples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(n_examples))


def _write_audio_and_collect(subset: Dataset, cfg: RadaiKeytermBiasingConfig, audio_dir: str) -> list[dict]:
    os.makedirs(audio_dir, exist_ok=True)
    records = []
    for row in subset:
        uid = str(row[cfg.id_column])
        audio_filepath = os.path.join(audio_dir, f"{re.sub(r'[^A-Za-z0-9._-]+', '_', uid)}.wav")
        if not os.path.exists(audio_filepath):
            audio = row["audio"]
            sf.write(audio_filepath, audio["array"], audio["sampling_rate"], subtype="PCM_16")
        records.append(
            {
                "audio_filepath": audio_filepath,
                "text": row[cfg.text_column],
                "medical_terms": _normalize_medical_terms(row[cfg.medical_terms_column]),
                "id": uid,
            }
        )
    return records


def _pred_text(output) -> str:
    """
    Extract the best transcript from a transcribe output element, which may be a string, a Hypothesis, an
    NBestHypotheses (beam search returns n-best lists ordered best-first), or a nested list of those.
    """
    if hasattr(output, "n_best_hypotheses"):
        output = output.n_best_hypotheses[0]
    while isinstance(output, (list, tuple)):
        output = output[0]
    return output.text if hasattr(output, "text") else str(output)


def _load_model(cfg: RadaiKeytermBiasingConfig, device: torch.device) -> ASRModel:
    torch.set_float32_matmul_precision("high")
    if cfg.model_path is not None:
        asr_model = ASRModel.restore_from(restore_path=cfg.model_path, map_location=device)
    else:
        asr_model = ASRModel.from_pretrained(model_name=cfg.pretrained_name, map_location=device)
    asr_model.eval()
    compute_dtype = get_inference_dtype(cfg.compute_dtype, device)
    if compute_dtype != torch.float32:
        asr_model.to(compute_dtype)
    logging.info(f"Device: {device} | compute dtype: {compute_dtype}")
    return asr_model


def _make_decoding_cfg(
    base_decoding_cfg,
    vocab: list[str],
    alpha: float,
    mode: str,
    beam_size: int | None,
    cfg: RadaiKeytermBiasingConfig,
):
    decoding_cfg = copy.deepcopy(base_decoding_cfg)
    with open_dict(decoding_cfg):
        if mode == "greedy":
            decoding_cfg.strategy = "greedy_batch"
            if "greedy" not in decoding_cfg or decoding_cfg.greedy is None:
                decoding_cfg.greedy = {}
            fusion_cfg = decoding_cfg.greedy
        else:
            decoding_cfg.strategy = cfg.beam_strategy
            if "beam" not in decoding_cfg or decoding_cfg.beam is None:
                decoding_cfg.beam = {}
            decoding_cfg.beam.beam_size = beam_size
            fusion_cfg = decoding_cfg.beam
        boosting_cfg = BoostingTreeModelConfig()
        if alpha > 0.0:
            boosting_cfg = BoostingTreeModelConfig(
                key_phrases_list=list(vocab),
                context_score=cfg.context_score,
                depth_scaling=cfg.depth_scaling,
                bpe_mode=cfg.bpe_mode,
            )
        fusion_cfg.boosting_tree = OmegaConf.structured(boosting_cfg)
        fusion_cfg.boosting_tree_alpha = alpha
        decoding_cfg.preserve_alignments = False
        decoding_cfg.compute_timestamps = False
        if "confidence_cfg" in decoding_cfg and decoding_cfg.confidence_cfg is not None:
            decoding_cfg.confidence_cfg.preserve_frame_confidence = False
        if "fused_batch_size" in decoding_cfg:
            decoding_cfg.fused_batch_size = -1
    return decoding_cfg


def _write_jsonl(path: str, records: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


_RESULT_COLUMNS = [
    "mode",
    "beam_size",
    "alpha",
    "wer_pct",
    "R_med_pct",
    "P_med_pct",
    "FNR_med_pct",
    "rFNR_med_pct",
    "FDR_med_pct",
    "tp",
    "fn",
    "fp",
    "num_ref_terms",
    "num_hyp_terms",
    "subs",
    "dels",
    "ins",
    "num_ref_words",
]


def _write_results(results: list[dict], path: str):
    results = finalize_rfnr(results)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_RESULT_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow({column: result.get(column, "") for column in _RESULT_COLUMNS})


def _print_summary(results: list[dict], results_path: str):
    results = finalize_rfnr(results)
    logging.info(
        f"{'strategy':>9} {'alpha':>8} {'WER %':>8} {'R_med %':>8} {'P_med %':>8} {'FNR_med %':>10} {'rFNR_med %':>11}"
    )
    for result in results:
        mode = result.get("mode", "beam")
        strategy = "greedy" if mode == "greedy" else f"beam{result['beam_size']}"
        logging.info(
            f"{strategy:>9} {result['alpha']:>8} {result['wer_pct']:>8.2f} {result['R_med_pct']:>8.2f} "
            f"{result['P_med_pct']:>8.2f} {result['FNR_med_pct']:>10.2f} {result['rFNR_med_pct']:>11.2f}"
        )
    logging.info("Reference: arXiv:2605.16545v2 Table 7 (Symphony, MedRad): rFNR_med=50.9%, P_med 97.0% -> 96.7%")
    logging.info(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
