#!/usr/bin/env python3
"""Encode WAV audio + a transcript into the code layout Moshi trains on.

The produced ``codes`` array has shape ``[1 + n_q, T]``:

* channel 0       -> text tokens (0 = padding, -1 = "no text here")
* channels 1..8   -> *main* speaker audio codebooks (what the DepFormer predicts)
* channels 9..16  -> *other* speaker audio codebooks (model input)

This mirrors ``moshi-finetune``'s ``InterleavedTokenizer`` (text padding = 0,
zero padding = -1, ``keep_main_only``) but runs entirely on Metal via
``rustymimi`` instead of a CUDA Mimi.

Because SpeechRu has no transcripts we do not attempt forced alignment here; the
transcript tokens are spread evenly across the clip.  If you do have word
timings, pass ``--alignments file.json`` with
``{"alignments": [[word, [start, end], "SPEAKER_MAIN"], ...]}`` and they will be
used exactly like the reference interleaver.

Examples::

    python scripts/encode_codes.py --wav clip.wav --text "привет мир" --out codes/clip.npz
    python scripts/encode_codes.py --wav clip.wav --alignments clip.json --out codes/clip.npz
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import rustymimi  # noqa: E402
import sentencepiece  # noqa: E402
import soundfile as sf  # noqa: E402

from metal import model_io  # noqa: E402

SAMPLE_RATE = 24000
FRAME_RATE = 12.5  # 24000 / 1920
FRAME_SIZE = 1920
TEXT_PADDING = 0
ZERO_PADDING = -1


def load_mono(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != target_sr:
        raise SystemExit(f"{path}: expected {target_sr} Hz, got {sr} (resample first)")
    return data.mean(axis=1)


def load_stereo(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return ``[2, samples]`` (left = main, right = other)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != target_sr:
        raise SystemExit(f"{path}: expected {target_sr} Hz, got {sr}")
    if data.shape[1] == 1:
        return np.stack([data[:, 0], data[:, 0]], axis=0)
    return data[:, :2].T


def encode_channel(tok: rustymimi.Tokenizer, samples: np.ndarray) -> np.ndarray:
    """Encode mono samples ``[T]`` -> codes ``[8, F]`` on Metal."""
    tok.reset()
    # rustymimi needs a whole number of 1920-sample frames.
    pad = (-len(samples)) % FRAME_SIZE
    if pad:
        samples = np.pad(samples, (0, pad))
    codes = tok.encode(samples[None, None, :].astype(np.float32))
    return np.array(codes)[0]


def build_text_stream(
    tokenizer: sentencepiece.SentencePieceProcessor,
    text: str | None,
    alignments: list | None,
    num_frames: int,
) -> np.ndarray:
    """Return an int array ``[num_frames]`` of text tokens."""
    if text is None and alignments is None:
        return np.full(num_frames, ZERO_PADDING, dtype=np.int32)

    tokens = np.full(num_frames, TEXT_PADDING, dtype=np.int32)

    if alignments is not None:
        # Reference-style placement: each word's tokens start at its start frame.
        for word, (start, end), _speaker in sorted(alignments, key=lambda a: a[1][0]):
            t0 = int(start * FRAME_RATE)
            toks = tokenizer.encode(word.strip())
            for j, tk in enumerate(toks):
                if t0 + j < num_frames:
                    tokens[t0 + j] = tk
        return tokens

    # No alignment available: spread the tokens evenly across the clip.
    toks = tokenizer.encode(text)
    if not toks:
        return tokens
    for j, tk in enumerate(toks):
        frame = min(num_frames - 1, int(j * num_frames / len(toks)))
        tokens[frame] = tk
    return tokens


def build_codes(
    tok: rustymimi.Tokenizer,
    text_tokenizer: sentencepiece.SentencePieceProcessor,
    wav_path: Path,
    text: str | None,
    alignments: list | None,
    other: str = "same",
) -> np.ndarray:
    stereo = load_stereo(wav_path)
    main = encode_channel(tok, stereo[0])                      # [8, F]
    num_frames = main.shape[1]

    if other == "same":
        # Re-encoding the same audio would advance rustymimi's encoder state a
        # second time; when the channels are identical (mono input) just reuse
        # the main codes.
        if stereo.shape[0] > 1 and not np.array_equal(stereo[0], stereo[1]):
            other_codes = encode_channel(tok, stereo[1])
        else:
            other_codes = main
    elif other == "silence":
        other_codes = encode_channel(tok, np.zeros_like(stereo[1]))
    elif other == "zero":
        other_codes = np.full_like(main, ZERO_PADDING)
    else:
        raise ValueError(f"unknown --other {other!r}")

    f = min(main.shape[1], other_codes.shape[1])
    main, other_codes = main[:, :f], other_codes[:, :f]

    text_codes = build_text_stream(text_tokenizer, text, alignments, f)[None]  # [1,F]
    codes = np.concatenate([text_codes.astype(np.int32), main.astype(np.int32),
                            other_codes.astype(np.int32)], axis=0)
    return codes


def _make_tokenizer(mimi_weights: str | None):
    repo = model_io.DEFAULT_REPOS["q4"]
    mimi_path = model_io._download(repo, model_io.MIMI_FILE, mimi_weights)
    return mimi_path, rustymimi.Tokenizer(str(mimi_path), num_codebooks=8)


# --------------------------------------------------------------------------- #
# Bulk encoding (optionally parallel).
# --------------------------------------------------------------------------- #
_WORKER: dict = {}


def _init_worker(mimi_path: str, tok_path: str, other: str) -> None:
    _WORKER["mimi_path"] = mimi_path
    _WORKER["sp"] = sentencepiece.SentencePieceProcessor(tok_path)
    _WORKER["other"] = other


def _encode_one(job: tuple[str, str]) -> str:
    wav = Path(job[0])
    out = job[1]
    alignments = None
    sidecar = wav.with_suffix(".json")
    if sidecar.exists():
        alignments = json.loads(sidecar.read_text())["alignments"]
    # A fresh Tokenizer per utterance works around rustymimi's un-resettable
    # encoder buffer (see README).
    tok = rustymimi.Tokenizer(_WORKER["mimi_path"], num_codebooks=8)
    codes = build_codes(tok, _WORKER["sp"], wav, None, alignments, other=_WORKER["other"])
    np.savez_compressed(out, codes=codes.astype(np.int32))
    return out


def _has_text_tokens(path: Path) -> bool:
    try:
        codes = np.load(path)["codes"]
        return int((codes[0] > 0).sum()) > 0
    except Exception:
        return False



def _encode_jobs(wavs: list[Path], out_dir: Path, skip_with_text: bool = True) -> list[tuple[str, str]]:
    jobs = []
    for w in wavs:
        out = out_dir / f"{w.stem}.npz"
        if skip_with_text and out.exists() and _has_text_tokens(out):
            continue
        jobs.append((str(w), str(out)))
    return jobs
def encode_directory(data_dir: Path, out_dir: Path, mimi_path: Path, tok_path: Path,
                     other: str, workers: int = 2, nice: int = 10) -> None:
    """Encode every ``*.wav`` under ``data_dir`` into ``out_dir/<stem>.npz``.

    A sidecar ``<stem>.json`` with ``{"alignments": ...}`` (see
    :mod:`scripts.transcribe`) is used for the text stream if present.

    ``nice`` lowers the scheduling priority of this process (and, by
    inheritance, every worker) so a long encode does not pin all cores and
    overheat a fanless machine. The default worker count is deliberately small
    for the same reason — raise ``--workers`` on a desktop with cooling.
    """
    if nice:
        try:
            os.nice(nice)
        except OSError:
            pass
    out_dir.mkdir(parents=True, exist_ok=True)
    wavs = sorted(data_dir.rglob("*.wav"))
    if not wavs:
        raise SystemExit(f"no .wav files under {data_dir}")
    jobs = _encode_jobs(wavs, out_dir, skip_with_text=True)

    if workers <= 1:
        _init_worker(str(mimi_path), str(tok_path), other)
        for i, job in enumerate(jobs):
            _encode_one(job)
            if (i + 1) % 50 == 0:
                print(f"[encode] {i+1}/{len(jobs)}", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(str(mimi_path), str(tok_path), other)) as ex:
            for i, _ in enumerate(ex.map(_encode_one, jobs, chunksize=4)):
                if (i + 1) % 50 == 0:
                    print(f"[encode] {i+1}/{len(jobs)}", flush=True)
    print(f"[encode] encoded {len(jobs)} files -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", type=Path, default=None, help="Single WAV to encode.")
    ap.add_argument("--data-dir", type=Path, default=None, help="Directory of WAVs to encode in bulk.")
    ap.add_argument("--out", type=Path, required=True, help="Output .npz (single) or directory (bulk).")
    ap.add_argument("--text", default=None, help="Transcript (used if --alignments is absent).")
    ap.add_argument("--alignments", type=Path, default=None, help="JSON with word alignments.")
    ap.add_argument("--other", default="same", choices=["same", "silence", "zero"],
                    help="What to put in the 'other speaker' channels.")
    ap.add_argument("--workers", type=int, default=2,
                    help="Parallel workers for --data-dir (encoding is CPU-bound). "
                         "Kept low by default to avoid overheating fanless machines.")
    ap.add_argument("--nice", type=int, default=10,
                    help="Unix nice level applied to the encoder (0-19).")
    ap.add_argument("--mimi-weights", default=None)
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args()

    repo = model_io.DEFAULT_REPOS["q4"]
    mimi_path = model_io._download(repo, model_io.MIMI_FILE, args.mimi_weights)
    tok_path = model_io._download(repo, model_io.TOKENIZER_FILE, args.tokenizer)

    if args.data_dir is not None:
        encode_directory(args.data_dir, args.out, Path(mimi_path), Path(tok_path),
                         args.other, workers=args.workers, nice=args.nice)
        return

    if args.wav is None:
        raise SystemExit("provide --wav or --data-dir")

    text_tok = sentencepiece.SentencePieceProcessor(tok_path)
    tok = rustymimi.Tokenizer(str(mimi_path), num_codebooks=8)
    alignments = None
    if args.alignments:
        alignments = json.loads(args.alignments.read_text())["alignments"]

    codes = build_codes(tok, text_tok, args.wav, args.text, alignments, other=args.other)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, codes=codes.astype(np.int32))
    print(f"[encode] {args.wav} -> {args.out}  codes={codes.shape} "
          f"({codes.shape[1] / FRAME_RATE:.1f}s, text_tokens={int((codes[0] > 0).sum())})")


if __name__ == "__main__":
    main()
