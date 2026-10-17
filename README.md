# Moshi-RU on Apple Metal

Fine-tuning and running [Moshi](https://github.com/kyutai-labs/moshi) for **Russian
speech-to-speech** on Apple Silicon, using **MLX / Metal** (no CUDA).

This repository was built because the original
[`Copy_of_moshi_finetune.ipynb`](Copy_of_moshi_finetune.ipynb) cannot work as
written (see [`ANALYSIS.md`](ANALYSIS.md)) and because the official
`moshi-finetune` trainer is CUDA/FSDP-only.

## What actually works here

| Capability | Status on an 8 GB Apple M2 |
|------------|----------------------------|
| **Speech-to-speech CLI** (7B Moshiko, 4-bit) on Metal | ✅ verified, ~6 steps/s (faster than real time) |
| Load & run 7B q4 model (fits in 4.8 GB) | ✅ |
| **Train a small Moshi model from scratch** on Metal | ✅ verified, loss 18.3 → 6.5 in 40 steps; 7.7 → 0.13 on real Russian |
| **QLoRA fine-tune of the 4-bit 7B** (training) | ✅ verified at rank 8 / 32 frames (~20 s/step); rank 32 swaps |
| QLoRA **inference** on 8 GB | ❌ generation peak ~5.6 GB > 5.33 GB wired limit; needs 16 GB+ |
| Full 7B bf16 / 16-bit fine-tune | ❌ needs ≥ 16 GB |
| Notebook as published | ❌ (wrong data format, no transcripts, CUDA-only) |

## Measured results

All numbers are from an Apple M2 with 8 GB unified memory (macOS, `mlx 0.26.5`).

| Run | Data | Model | Throughput | Loss |
|-----|------|-------|-----------|------|
| Inference (CLI) | 4.1 s input | 7B q4 | **6.1–8.6 step/s** | – |
| Inference (CLI) | 4.1 s input | small (131 M) | **11.2–17.5 step/s** | – |
| Train, toy TTS | 192 clips (6 min) | small | ~1.4 step/s | 18.2 → **0.05** (memorised) |
| Train, real Russian | 222 clips (74 min) | small | ~1.3 step/s (batch 8 × 128) | 7.7 → **0.12** (1500 steps, 1122 s) |
| QLoRA 7B q4 (rank 8, seq 32) | 192 clips | 7B | ~22 s/step | 13.7 → **12.8** (2 steps) |

The toy run reaches near-zero loss because the dataset is only 24 unique TTS
phrases — it memorises them (a correctness proof, not a voice). The real
SpeechRu run also drops steeply (222 clips are few, so it partly memorises), but
it is genuine Russian audio and the loop is the same as a full run.
Because SpeechRu has **no transcripts**, the text head is not trained and the
CLI's "inner monologue" text is gibberish; only the audio stream is learned.

> **Use batch 8 × 128 frames with a slow warmup.** On a dataset this small,
> `--batch-size 8 --seq-len 128 --warmup 100` is what drives the memorisation
> phase transition: the loss falls steeply from ~6 to ~0.1 once the LR is at its
> peak. These are now the trainer defaults (and the `make` target passes them
> explicitly); the earlier `--batch-size 4 --seq-len 96` default plateaued near
> `ln(2048) ≈ 7.6` and never fit the data.

## Quickstart

All commands run from the repository root with the bundled virtualenv.

### 1. Speech-to-speech from a file

```bash
.venv/bin/python cli/moshi_ru.py --input sample_ru.wav --output reply.wav --show-text
```

Output on the base model (which only speaks English/French):

```
[cli] input: sample_ru.wav  (4.1s, 51 steps)
[cli] generated 51 steps in 8.4s (6.1 step/s)
[cli] inner monologue:
 Hey what's up? I'm just listening to some music.
[cli] wrote 4.0s to sample_out.wav
```

### 2. Interactive microphone

```bash
.venv/bin/python cli/moshi_ru.py --mic
```

> Use headphones so the model does not hear itself.

## Training

### Option A — small model trained from scratch (recommended on 8 GB)

A small Moshi-architecture model reuses the released Mimi codec and the same
text/audio vocabularies, so it plugs straight into the CLI.

```bash
# 1. Build a toy Russian dataset (macOS TTS) to verify the loop
.venv/bin/python scripts/make_toy_dataset.py --out data/toy --repeats 8

# 2. Train on Metal (batch 8 x 128 + warmup 100 is the recipe that fits)
.venv/bin/python train/train_mlx.py --size small \
    --data-dir data/toy/codes --out runs/small_toy \
    --steps 1500 --batch-size 8 --seq-len 128 --warmup 100

# 3. Run the fine-tuned model
.venv/bin/python cli/moshi_ru.py --size small \
    --checkpoint runs/small_toy/model_final.safetensors \
    --input something.wav --output reply.wav
```

> A ready-trained model is included at
> `runs/speechru_small/model_final.safetensors` (small, 1500 steps on the real
> SpeechRu slice). Use it with `--size small --checkpoint <that path>`.

Preset sizes (parameter counts measured on this machine):

| `--size` | params | main dim | layers |
|----------|--------|----------|--------|
| `nano`   | 41 M   | 256      | 6      |
| `tiny`   | 75 M   | 384      | 8      |
| `small`  | 131 M  | 512      | 12     |
| `medium` | 274 M  | 768      | 16     |
| `7b`     | ~7 B   | 4096     | 32     |

### Option B — QLoRA fine-tune of the 7B model

```bash
.venv/bin/python train/train_mlx.py --size 7b --model q4 \
    --data-dir data/speech-ru/codes --out runs/qlora \
    --steps 500 --batch-size 1 --seq-len 32 --lora-rank 8

.venv/bin/python cli/moshi_ru.py --size 7b --model q4 \
    --lora runs/qlora/lora_final.safetensors --input in.wav --output out.wav
```

Only the LoRA adapters are trainable (`metal/lora.py`); the 4-bit base is frozen.
The adapters are small (rank 8 ≈ 94 MB, rank 32 ≈ 375 MB).

> **On 8 GB, keep the rank small.** QLoRA *training* fits at rank 8 / short
> sequences (~5.3 GB, ~20 s/step); rank 32 spills into swap. QLoRA *inference*
> raises the peak to ~5.6 GB, above the wired limit, so it needs a **16 GB+** Mac
> (or the adapter merged into a bf16 model elsewhere). See `HARDWARE.md`.

## Data pipeline (real Russian speech)

`SpeechRu` (`Sinoosoida/SpeechRu`) has ~186 k recordings (7.38 TB) **without
transcripts**. Each row is a *long podcast episode* (hundreds to thousands of
seconds) stored as an `audio` dict with raw mp3 `bytes`; there is no transcript,
only an `annotation_plain` podcast description. The pipeline:

```bash
# 1. Download a bounded slice and cut episodes into 24 kHz clips
#    (writes wav/, manifest.jsonl, stereo/ and moshi_data.jsonl)
.venv/bin/python scripts/prepare_dataset.py --limit-gb 0.6 --max-clips 600 \
    --clip-seconds 20 --stereo

# 2. Transcribe with mlx-whisper -> per-file alignment JSON (Metal ASR).
#    Writes <name>.json next to each WAV in data/speech-ru/wav, which is exactly
#    where encode_codes.py looks for it in step 3a.
.venv/bin/python scripts/transcribe.py --data-dir data/speech-ru/wav \
    --model mlx-community/whisper-medium-mlx --language ru

# 3a. (Metal training) encode WAV + transcript into codes
#     (~4 s per 20 s clip on CPU Mimi). --workers parallelises it, but the
#     default is kept low (2) and niced so a fanless Mac does not overheat.
.venv/bin/python scripts/encode_codes.py --data-dir data/speech-ru/wav \
    --out data/speech-ru/codes --other same --workers 2 --nice 10

# 3b. (GPU training) the --stereo output is already a valid moshi-finetune dataset:
#     stereo WAVs + .json alignments + moshi_data.jsonl
```

Because SpeechRu has no transcripts, the text stream is left empty (all `-1`) and
only the audio objective is trained; the model still learns Russian phonetics and
prosody. Run `scripts/transcribe.py` first if you want the text head trained too.

## Running the official `moshi-finetune` (when you have a GPU)

The `--stereo` output matches what `moshi-finetune` expects, so you can move the
data to a CUDA box and use the upstream trainer unchanged. Use
`scripts/transcribe.py` instead of `annotate.py`.

## Architecture

```
cli/moshi_ru.py            Speech-to-speech CLI (file + mic), 7B & small models, LoRA
metal/
  configs.py               Small Moshi configs (nano/tiny/small/medium)
  model_io.py              Weight download/loading (q4/q8/bf16)
  lora.py                  LoRA over dense and 4-bit QuantizedLinear (QLoRA)
  train_forward.py         Differentiable Metal training forward + loss
train/train_mlx.py         Metal trainer (from-scratch or QLoRA)
scripts/
  prepare_dataset.py       SpeechRu -> 24 kHz WAV + manifest (+ stereo moshi dataset)
  transcribe.py            mlx-whisper word alignments
  encode_codes.py          WAV + text -> Moshi code layout
  make_toy_dataset.py      macOS-TTS smoke-test dataset
  doctor.py                environment / Metal / memory report
tests/test_delay_equivalence.py   delay+undelay vs the PyTorch reference
Makefile                   convenience targets (make help)
requirements.txt           pinned dependencies
ANALYSIS.md                Notebook bug analysis
HARDWARE.md                Memory / speed measurements on 8 GB
```

### How training works on Metal

`moshi_mlx` is inference-only and its `Transformer` never applies a causal mask.
`metal/train_forward.py` therefore:

1. monkey-patches `TransformerLayer`/`Transformer` to forward a mask,
2. re-implements Moshi's training forward — the per-codebook **delays**, the
   temporal transformer pass, and the **teacher-forced DepFormer** pass (the
   DepFormer reproduces `weights_per_step` attention by reusing a shared KV cache
   across codebook slices, exactly like generation does),
3. applies the same masked cross-entropy as `moshi-finetune`
   (`text_loss + audio_loss`, `first_codebook_weight_multiplier`, padding weights).

The loss was validated to match the PyTorch reference on the delay/undelay logic;
`metal/train_forward.py` mirrors `moshi/models/lm.py` and `finetune/loss.py`.

## Honest limitations

* **8 GB is the constraint.** The 4-bit 7B base already occupies ~4.8 GB; there
  is little room for activations. QLoRA *training* fits at rank 8 (rank 32 swaps),
  but QLoRA *inference* exceeds the wired limit and needs a 16 GB+ Mac. See
  [`HARDWARE.md`](HARDWARE.md).
* A from-scratch small model **learns speech statistics**, but producing fluent
  Russian dialogue needs far more than the toy dataset — the toy run is a
  correctness/throughput proof, not a finished voice. Point `--data-dir` at the
  real encoded SpeechRu slice to train for real.
* The base 7B model speaks English/French; Russian comes only from fine-tuning.

## License / data note

`SpeechRu` is distributed "scraped-from-web-for-research-use"; check its terms
before any non-research use.
