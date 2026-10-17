# Analysis of `Copy_of_moshi_finetune.ipynb`

This document explains what the original notebook does, why it cannot succeed as
written, and what this repository replaces it with.

## 1. What the notebook is

An 18-cell Colab-style notebook that tries to fine-tune **Moshi 7B** for Russian
speech. The important cells are:

| Cell | Purpose |
|------|---------|
| 11 | Config: dataset id, output dirs, `audio_paths`, JSONL path. |
| 12 | Downloads the `Sinoosoida/SpeechRu` dataset (audio + metadata). |
| 13 | "Training" cell — writes a JSONL manifest and invokes `moshi-finetune`. |
| 14 | A hack cell that patches a file to re-raise a swallowed traceback (evidence the author hit an error they could not see). |
| 17 | Inference cell (uses the PyTorch `moshi` runner). |

## 2. Concrete bugs

### 2.1 The dataset has no transcripts

`SpeechRu` exposes, among others, the columns `audio`, `annotation_plain`,
`name`, `duration_sec`. `annotation_plain` is a **podcast description** (title /
summary of the episode), *not* a word-level transcript of the speech.

Cell 13 nevertheless builds records like:

```json
{"audio_path": ".../clip.mp3", "text": "<annotation_plain>"}
```

Moshi's fine-tuning expects a **word-level alignment** of the *spoken* audio
(`{"alignments": [[word, [start, end], speaker], ...]}`), plus codes with the
delays applied. Feeding a podcast blurb as the transcript teaches the text head
to predict unrelated text — the model cannot converge.

**Fix:** run real ASR with word timestamps (`scripts/transcribe.py`, using
`mlx-whisper`) and treat `annotation_plain` as a `description`, never as `text`.

### 2.2 Wrong audio format

The notebook writes **mono `.mp3`** files. `moshi-finetune` requires:

* **24 kHz** PCM (Mimi's sample rate),
* **stereo** WAV — channel 0 = Moshi's own audio ("main"), channel 1 = the user,
* a sidecar `*.json` alignment file next to each WAV,
* a JSONL of `{"path": ..., "duration": ...}` (not `{"audio_path","text"}`).

**Fix:** `scripts/prepare_dataset.py` writes 24 kHz WAV (optionally stereo) and
a `{"path","duration",...}` JSONL.

### 2.3 Mono encodes only half the model

Moshi is a **full-duplex** model: the temporal transformer consumes the *user*
codebooks while the DepFormer generates *Moshi's* codebooks. With mono audio and
no second stream the "user" channels are undefined, so the duplex objective is
never learned.

**Fix:** provide a real stereo source, or (for monologue language adaptation)
duplicate the speaker onto the main channel and silence/zero the other channel.

### 2.4 The whole dataset is 7.38 TB of long podcasts

Cell 12 downloads the entire dataset. Even a single epoch is impossible on
consumer hardware — and every row is a **full-length podcast episode** (hundreds
to thousands of seconds), far longer than any training sequence.

**Fix:** `scripts/prepare_dataset.py` streams a bounded slice (`--limit-gb`,
`--max-clips`) and cuts each episode into short `--clip-seconds` segments. Each
row's audio lives in an `audio` dict with raw mp3 `bytes`.

### 2.5 The training backend cannot run on this machine

`moshi-finetune` is CUDA/FSDP-bound:

* `train.py` -> `assert torch.cuda.is_available()`, `.cuda()`, `torch.distributed`
  with NCCL, bf16 weights (~14 GB for 7B).
* It therefore cannot run on an 8 GB Apple Silicon Mac at all.

**Fix:** this repository ships a Metal-native training path (see `README.md`).

### 2.6 Cell 14 is a symptom, not a feature

Cell 14 monkey-patches a library file so a previously swallowed exception surfaces
again. It is a debugging workaround, and it means the notebook as published never
completed the run in the environment it was written for.

## 3. Verdict

The notebook is a reasonable *sketch* of a Moshi fine-tune, but it:

1. has no usable transcripts,
2. produces data in the wrong format (mono MP3 + wrong JSONL schema),
3. tries to download 7.38 TB,
4. targets a CUDA-only trainer.

None of these can be fixed by changing hyper-parameters. This repository
re-implements the whole path so that it (a) produces correct data and (b) runs
on Apple Metal.

See `README.md` for the replacement pipeline and `HARDWARE.md` for the
feasibility numbers on an 8 GB M2.
