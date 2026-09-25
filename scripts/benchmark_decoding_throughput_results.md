# Decoding Throughput Benchmark: Greedy vs Beam Search

## Overview

**Question:** What is the throughput cost of switching from greedy to beam-search decoding, and which beam width should we use for key-term biasing?

- **Context:** Key-term biasing requires beam-search decoding (`malsd_batch` or `maes_batch`) to expand and rescore hypotheses against a biasing trie. Greedy decoding (`greedy_batch`) cannot apply per-stream biasing because it maintains only a single hypothesis per sample.
- **Model:** `nvidia/parakeet-tdt-0.6b-v2` (Parakeet TDT 0.6B)
- **Dataset:** LibriSpeech clean validation (200 samples, 1282.1s of audio)
- **Hardware:** A100-SXM4-80GB MIG 1g.10gb (10.2 GB), bf16 compute
- **Batch size:** 8
- **Warmup:** 1 run; **Timed runs:** 3 (mean ± std)
- **Metric:** RTFx = total_audio_duration / decode_time (higher = faster)
- **Script:** `scripts/speech_recognition/benchmark_decoding_throughput.py`

## Results

| Strategy          | Beam | Decode Time (s)   | RTFx              | vs Greedy |
|-------------------|------|--------------------|--------------------|-----------|
| greedy_batch      | —    | 4.503 ± 0.143      | 285.0 ± 8.9       | 1.00x     |
| malsd_batch_b1    | 1    | 4.739 ± 0.029      | 270.6 ± 1.7       | 0.95x     |
| malsd_batch_b2    | 2    | 5.029 ± 0.034      | 255.0 ± 1.7       | 0.89x     |
| malsd_batch_b4    | 4    | 5.381 ± 0.089      | 238.3 ± 4.0       | 0.84x     |
| malsd_batch_b8    | 8    | 5.822 ± 0.059      | 220.2 ± 2.2       | 0.77x     |

## Key Findings

1. **Greedy decoding is fastest** at 285x RTFx — well above real-time.

2. **Beam=1 adds only 5% overhead** (271 vs 285 RTFx) despite not expanding the search space. This is the fixed cost of the beam infrastructure (hypothesis bookkeeping, sorting, scoring).

3. **Beam scaling is smooth and roughly linear (~6-7% per doubling):**
   - Beam 1 -> 2: -6% RTFx (271 -> 255)
   - Beam 2 -> 4: -6% RTFx (255 -> 238)
   - Beam 4 -> 8: -7% RTFx (238 -> 220)

4. **Beam=8 is 23% slower than greedy** (0.77x) but still achieves 220x RTFx — far above real-time.

5. **Scaling is sub-linear** — doubling the beam width does not double decode time. The batched `malsd_batch` strategy with CUDA graphs keeps the overhead manageable even at beam=8.

6. **Recommendation: beam=4 is a practical default for key-term biasing.** It retains 84% of greedy throughput (238x RTFx) while providing enough hypothesis diversity for effective biasing. Beam=8 buys marginal biasing improvement at ~8% additional cost.

## Methodology

- **Script:** `scripts/speech_recognition/benchmark_decoding_throughput.py`
- **Timer:** `SimpleTimer` with CUDA synchronization (`torch.cuda.synchronize`) at start/end of each run
- **Warmup:** 1 untimed run per configuration to stabilize CUDA graphs and cuDNN
- **Strategy switching:** `model.change_decoding_strategy()` with `OmegaConf` config merge, preserving all model-specific decoding settings
- **Alignment/timestamp/confidence:** Disabled during benchmarking to isolate pure decode cost
- **Data:** 200 files from `openslr/librispeech_asr` (clean, validation split), written to 16kHz WAV

## Reproduction

```bash
# Download LibriSpeech samples and build manifest
python -c "
from datasets import load_dataset
import soundfile as sf, os, json
ds = load_dataset('openslr/librispeech_asr', 'clean', split='validation')
os.makedirs('/tmp/libri_bench', exist_ok=True)
manifest = []
for i in range(200):
    sample = ds[i]
    path = f'/tmp/libri_bench/sample_{i:04d}.wav'
    sf.write(path, sample['audio']['array'], sample['audio']['sampling_rate'])
    manifest.append({'audio_filepath': path, 'text': sample['text'],
                     'duration': len(sample['audio']['array']) / sample['audio']['sampling_rate']})
with open('/tmp/libri_bench/manifest.json', 'w') as f:
    for item in manifest: f.write(json.dumps(item) + '\n')
"

# Run benchmark on MIG (batch=8)
CUDA_VISIBLE_DEVICES=MIG-b9e23c79-55f7-5638-823d-56a0a9b84b09 \
python scripts/speech_recognition/benchmark_decoding_throughput.py \
    --pretrained-name nvidia/parakeet-tdt-0.6b-v2 \
    --dataset-manifest /tmp/libri_bench/manifest.json \
    --beam-widths 1,2,4,8 \
    --batch-size 8 \
    --warmup-steps 1 \
    --run-steps 3 \
    --compute-dtype bfloat16 \
    --output-json /tmp/libri_bench/results.json
```

## Raw Data

```json
[
  {
    "config_name": "greedy_batch",
    "strategy": "greedy_batch",
    "beam_width": null,
    "search_type": "greedy_batch",
    "decode_time_s": 4.503336945,
    "decode_time_std_s": 0.14314007922769848,
    "rtfx": 284.97209987681037,
    "rtfx_std": 8.901361453439687,
    "speedup_vs_greedy": 1.0
  },
  {
    "config_name": "malsd_batch_b1",
    "strategy": "malsd_batch",
    "beam_width": 1,
    "search_type": "malsd_batch",
    "decode_time_s": 4.739338399,
    "decode_time_std_s": 0.029230029751211967,
    "rtfx": 270.5779575661071,
    "rtfx_std": 1.6624371552585678,
    "speedup_vs_greedy": 0.9494794984233502
  },
  {
    "config_name": "malsd_batch_b2",
    "strategy": "malsd_batch",
    "beam_width": 2,
    "search_type": "malsd_batch",
    "decode_time_s": 5.028680636666666,
    "decode_time_std_s": 0.03360954281668498,
    "rtfx": 254.98256869992024,
    "rtfx_std": 1.6960359752466469,
    "speedup_vs_greedy": 0.8947249027744982
  },
  {
    "config_name": "malsd_batch_b4",
    "strategy": "malsd_batch",
    "beam_width": 4,
    "search_type": "malsd_batch",
    "decode_time_s": 5.381394977333333,
    "decode_time_std_s": 0.08913067836337137,
    "rtfx": 238.2950981969291,
    "rtfx_std": 3.986474769765137,
    "speedup_vs_greedy": 0.8361825235381651
  },
  {
    "config_name": "malsd_batch_b8",
    "strategy": "malsd_batch",
    "beam_width": 8,
    "search_type": "malsd_batch",
    "decode_time_s": 5.822052974333333,
    "decode_time_std_s": 0.058817942867761734,
    "rtfx": 220.23680508353275,
    "rtfx_std": 2.230411644971014,
    "speedup_vs_greedy": 0.7728259146748374
  }
]
```
