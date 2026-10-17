#!/usr/bin/env python3
"""Build a tiny Russian speech dataset by synthesising phrases with macOS TTS.

This is a *smoke-test* dataset: it lets the Metal trainer and the CLI be verified
end-to-end without downloading the multi-terabyte SpeechRu corpus.  It uses the
macOS ``say`` command with the Russian voice ``Milena`` and then encodes the
audio with Mimi exactly like :mod:`scripts.encode_codes`.

For real training use :mod:`scripts.prepare_dataset` (SpeechRu) and
:mod:`scripts.transcribe` (mlx-whisper transcripts) instead.

Example::

    python scripts/make_toy_dataset.py --out data/toy --repeats 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import rustymimi  # noqa: E402
import sentencepiece  # noqa: E402

from metal import model_io  # noqa: E402
from scripts.encode_codes import build_codes  # noqa: E402

PHRASES = [
    "Привет, как дела?",
    "Сегодня хорошая погода.",
    "Я люблю слушать музыку.",
    "Расскажи мне интересную историю.",
    "Москва — большой и красивый город.",
    "Давай пойдём гулять в парк.",
    "Что ты делаешь сегодня вечером?",
    "Мне нравится изучать иностранные языки.",
    "Это очень вкусный чай.",
    "У меня есть собака и кошка.",
    "Завтра будет солнечный день.",
    "Пожалуйста, говори медленнее.",
    "Я живу в России уже много лет.",
    "Который сейчас час?",
    "Спасибо большое за помощь.",
    "Мы встретимся завтра утром.",
    "Я хочу выучить русский язык.",
    "Как пройти до ближайшей станции метро?",
    "Этот фильм был очень интересным.",
    "Давай приготовим что-нибудь вкусное.",
    "Мне нужно купить хлеб и молоко.",
    "Она работает врачом в больнице.",
    "Мы часто ездим на природу летом.",
    "Дети играют во дворе.",
]


def synth(phrase: str, voice: str, out_wav: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "speech.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), phrase], check=True)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@24000", "-c", "1", str(aiff), str(out_wav)],
            check=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/toy"))
    ap.add_argument("--voice", default="Milena")
    ap.add_argument("--repeats", type=int, default=1, help="Encode each phrase this many times.")
    ap.add_argument("--other", default="same", choices=["same", "silence", "zero"])
    ap.add_argument("--mimi-weights", default=None)
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    codes_dir = args.out / "codes"
    wav_dir = args.out / "wav"
    codes_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    repo = model_io.DEFAULT_REPOS["q4"]
    mimi_path = model_io._download(repo, model_io.MIMI_FILE, args.mimi_weights)
    tok_path = model_io._download(repo, model_io.TOKENIZER_FILE, args.tokenizer)
    text_tok = sentencepiece.SentencePieceProcessor(tok_path)

    manifest = []
    idx = 0
    for rep in range(args.repeats):
        for phrase in PHRASES:
            wav = wav_dir / f"toy_{idx:05d}.wav"
            codes = None
            last_exc = None
            for attempt in range(3):
                try:
                    synth(phrase, args.voice, wav)
                    # NOTE: rustymimi.Tokenizer.reset() does not clear the encoder
                    # buffer, so a fresh Tokenizer is created for every utterance.
                    tok = rustymimi.Tokenizer(str(mimi_path), num_codebooks=8)
                    codes = build_codes(tok, text_tok, wav, phrase, None, other=args.other)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
            if codes is None:
                print(f"[toy] skip {idx} ({phrase!r}): {last_exc}")
                idx += 1
                continue
            np.savez_compressed(codes_dir / f"toy_{idx:05d}.npz", codes=codes.astype(np.int32))
            manifest.append({"codes": str(codes_dir / f"toy_{idx:05d}.npz"), "text": phrase,
                             "frames": int(codes.shape[1])})
            idx += 1

    import json
    (args.out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n", encoding="utf-8"
    )
    total = sum(m["frames"] for m in manifest) / 12.5
    print(f"[toy] wrote {len(manifest)} clips ({total:.1f}s of audio) to {codes_dir}")


if __name__ == "__main__":
    main()
