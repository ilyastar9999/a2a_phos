#!/usr/bin/env python3
"""Prepare the `Sinoosoida/SpeechRu` dataset for Moshi fine-tuning.

Fixes several problems with the original notebook:

* the notebook wrote `.mp3` files; we decode and write proper 24 kHz PCM WAV
  files (the sample rate Mimi requires);
* the notebook used the podcast ``annotation_plain`` metadata *as if it were a
  transcript*.  SpeechRu has **no transcripts**; this script stores that text
  under a clearly-named ``description`` field and leaves ``text`` empty.  Use
  ``scripts/transcribe.py`` to produce real transcripts.
* the notebook produced mono files with a ``{audio_path, text}`` manifest, which
  Moshi cannot consume.  Moshi needs **stereo** audio (left = model, right =
  user) and a ``{path, duration}`` JSONL plus a sidecar alignment JSON.

SpeechRu rows are **long podcast episodes** (hundreds to thousands of seconds),
so each episode is cut into fixed ``--clip-seconds`` segments before writing.

Outputs (under ``--out``)::

    out/
      wav/audio_000000.wav        # 24 kHz mono, 1 clip
      manifest.jsonl              # {"path","duration","episode","description"}
      (optional) stereo/... + moshi_data.jsonl  # when --stereo

Examples::

    # quick smoke test: ~0.5 GB of shards -> up to 2000 twenty-second clips
    python scripts/prepare_dataset.py --limit-gb 0.5 --max-clips 2000 --stereo

    # larger conversion
    python scripts/prepare_dataset.py --limit-gb 50 --max-clips 50000 --stereo
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import sphn  # noqa: E402
from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402

REPO = "Sinoosoida/SpeechRu"
SAMPLE_RATE = 24000


def log(msg: str) -> None:
    print(msg, flush=True)


def decode_audio(audio_bytes: bytes, target_sr: int) -> np.ndarray:
    """Decode compressed audio bytes to float32 ``[channels, samples]`` at ``target_sr``."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        pcm, _sr = sphn.read(tmp.name, sample_rate=target_sr)
    return np.asarray(pcm, dtype=np.float32)


def already_processed(out_dir: Path) -> int:
    wav_dir = out_dir / "wav"
    if not wav_dir.exists():
        return 0
    return sum(1 for _ in wav_dir.glob("*.wav"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--out", type=Path, default=Path("data/speech-ru"))
    ap.add_argument("--limit-gb", type=float, default=0.5, help="Stop after downloading this many GB.")
    ap.add_argument("--max-clips", type=int, default=2000, help="Stop after writing this many clips.")
    ap.add_argument("--clip-seconds", type=float, default=20.0, help="Length of each training clip.")
    ap.add_argument("--min-clip-seconds", type=float, default=4.0, help="Drop trailing segments shorter than this.")
    ap.add_argument("--min-rms", type=float, default=0.005, help="Skip near-silent segments.")
    ap.add_argument("--stereo", action="store_true",
                    help="Also write stereo WAVs (speaker duplicated) + moshi_data.jsonl.")
    ap.add_argument("--keep-parquet", action="store_true", help="Do not delete downloaded parquet shards.")
    args = ap.parse_args()

    out_dir: Path = args.out
    wav_dir = out_dir / "wav"
    parquet_dir = out_dir / "parquet"
    wav_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd  # imported late; heavy

    files = [f for f in list_repo_files(args.repo, repo_type="dataset") if f.endswith(".parquet")]
    if not files:
        raise SystemExit("no parquet files found in the dataset repo")
    files.sort()
    log(f"[data] {len(files)} parquet shards; limit={args.limit_gb} GB, max_clips={args.max_clips}, "
        f"clip={args.clip_seconds}s")

    max_bytes = int(args.limit_gb * 1024**3)
    clip_len = int(args.clip_seconds * SAMPLE_RATE)
    min_len = int(args.min_clip_seconds * SAMPLE_RATE)
    downloaded = 0
    count = already_processed(out_dir)
    started = count
    manifest_path = out_dir / "manifest.jsonl"
    manifest = manifest_path.open("a", encoding="utf-8")

    stereo_dir = out_dir / "stereo"
    moshi_jsonl_path = out_dir / "moshi_data.jsonl"
    moshi_lines: list[str] = []
    if args.stereo:
        stereo_dir.mkdir(parents=True, exist_ok=True)

    try:
        for shard in files:
            if downloaded >= max_bytes or count >= args.max_clips:
                break
            log(f"[data] downloading {shard}")
            local = Path(hf_hub_download(args.repo, shard, repo_type="dataset", local_dir=parquet_dir))
            downloaded += local.stat().st_size

            df = pd.read_parquet(local)
            for _, row in df.iterrows():
                if count >= args.max_clips:
                    break
                audio = row.get("audio")
                if not isinstance(audio, dict) or not audio.get("bytes"):
                    continue
                try:
                    pcm = decode_audio(audio["bytes"], SAMPLE_RATE)
                except Exception as exc:  # noqa: BLE001
                    log(f"[data]   skip episode: decode failed ({exc})")
                    continue
                if pcm.ndim == 1:
                    pcm = pcm[None]
                total = pcm.shape[-1]
                n_clips = total // clip_len
                for j in range(n_clips):
                    if count >= args.max_clips:
                        break
                    seg = pcm[:, j * clip_len : (j + 1) * clip_len]
                    if seg.shape[-1] < min_len:
                        continue
                    mono = seg.mean(axis=0)
                    if float(np.sqrt(np.mean(mono**2))) < args.min_rms:
                        continue

                    stem = f"audio_{count:07d}"
                    sf.write(wav_dir / f"{stem}.wav", mono, SAMPLE_RATE, subtype="PCM_16")
                    duration = seg.shape[-1] / SAMPLE_RATE
                    record = {
                        "path": str((wav_dir / f"{stem}.wav").resolve()),
                        "duration": round(duration, 4),
                        "episode": int(row.get("id", -1)) if "id" in row else -1,
                        "name": str(row.get("name", "")),
                        "description": str(row.get("annotation_plain", "") or ""),
                    }
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

                    if args.stereo:
                        left = seg[0]
                        right = seg[1] if seg.shape[0] > 1 else seg[0]
                        sf.write(stereo_dir / f"{stem}.wav", np.stack([left, right], axis=1),
                                 SAMPLE_RATE, subtype="PCM_16")
                        moshi_lines.append(json.dumps(
                            {"path": str((stereo_dir / f"{stem}.wav").resolve()),
                             "duration": round(duration, 4)}, ensure_ascii=False))

                    count += 1
                    if count % 200 == 0:
                        manifest.flush()
                        log(f"[data]   {count} clips ...")

            manifest.flush()
            if not args.keep_parquet:
                local.unlink(missing_ok=True)

        if args.stereo:
            with moshi_jsonl_path.open("w", encoding="utf-8") as fobj:
                fobj.write("\n".join(moshi_lines) + ("\n" if moshi_lines else ""))
    finally:
        manifest.close()

    log(f"[data] done: {count - started} new clips (total {count}). manifest: {manifest_path}")
    if args.stereo:
        log(f"[data] stereo dir: {stereo_dir}, jsonl: {moshi_jsonl_path}")
        log("[data] NOTE: run scripts/transcribe.py to create per-file alignment JSONs before moshi-finetune.")

    # Write a manifest backup so it can be regenerated cheaply.
    (out_dir / "clip_seconds.txt").write_text(str(args.clip_seconds))


if __name__ == "__main__":
    main()
