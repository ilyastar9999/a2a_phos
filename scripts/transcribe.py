#!/usr/bin/env python3
"""Transcribe WAV files on Apple Metal with ``mlx-whisper``.

Replaces ``moshi-finetune/annotate.py`` (which is CUDA + ``whisper_timestamped``)
and produces the **same alignment JSON** the reference pipeline expects:

    {"alignments": [[word, [start, end], "SPEAKER_MAIN"], ...],
     "text_conditions": {...}}

The output file is written next to each WAV as ``<name>.json`` (the location the
reference interleaver reads).

Example::

    python scripts/transcribe.py --data-dir data/speech-ru/wav \
        --model mlx-community/whisper-medium-mlx --language ru
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

SAMPLE_RATE = 24000
WHISPER_RATE = 16000


def load_16k_mono(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != WHISPER_RATE:
        # Polyphase resampling; no scipy dependency on the pure-python side is
        # required because numpy is enough for a simple linear interpolation.
        n = int(round(len(mono) * WHISPER_RATE / sr))
        idx = np.linspace(0, len(mono) - 1, n)
        mono = np.interp(idx, np.arange(len(mono)), mono).astype(np.float32)
    return mono


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, help="Directory of .wav files.")
    ap.add_argument("--model", default="mlx-community/whisper-medium-mlx",
                    help="mlx-whisper model repo. Use large-v3 for best Russian accuracy.")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--speaker", default="SPEAKER_MAIN")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import mlx_whisper

    wavs = sorted(args.data_dir.rglob("*.wav"))
    if not wavs:
        raise SystemExit(f"no .wav files under {args.data_dir}")

    for i, wav in enumerate(wavs):
        out = wav.with_suffix(".json")
        if out.exists() and not args.overwrite:
            continue
        audio = load_16k_mono(wav)
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=args.model,
            language=args.language,
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        alignments = []
        for seg in result.get("segments", []):
            for word in seg.get("words", []):
                text = word.get("word", "").strip()
                if not text:
                    continue
                alignments.append([text, [float(word["start"]), float(word["end"])], args.speaker])
        payload = {"alignments": alignments, "text": result.get("text", "").strip()}
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[asr] {i+1}/{len(wavs)} {wav.name}: {len(alignments)} words", flush=True)


if __name__ == "__main__":
    main()
