#!/usr/bin/env python3
"""Moshi speech-to-speech CLI running on Apple Metal (MLX).

Examples
--------
Offline (file -> file)::

    python cli/moshi_ru.py --input sample_ru.wav --output reply.wav

Interactive microphone (requires a loopback / headphones)::

    python cli/moshi_ru.py --mic

Use a smaller, Metal-trainable model and a fine-tuned checkpoint::

    python cli/moshi_ru.py --size small --checkpoint runs/small/step_2000.safetensors \
        --input in.wav --output out.wav

Apply a LoRA adapter on top of the 4-bit 7B base::

    python cli/moshi_ru.py --size 7b --model q4 --lora runs/qlora/lora.safetensors \
        --input in.wav --output out.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as a plain script: ``python cli/moshi_ru.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import mlx.core as mx  # noqa: E402
import rustymimi  # noqa: E402
import sentencepiece  # noqa: E402
import sphn  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from moshi_mlx import models, utils  # noqa: E402
from metal import configs  # noqa: E402
from metal import lora as lora_lib  # noqa: E402
from metal import model_io  # noqa: E402

SAMPLE_RATE = 24000
FRAME_SIZE = 1920  # 80 ms at 24 kHz, the Mimi frame size


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def configure_memory(wired_gb: float = 6.0, cache_mb: int = 256) -> None:
    """Cap MLX's GPU allocations so generation does not thrash on small Macs.

    The requested wired limit is clamped to ``max_recommended_working_set_size``
    (on an 8 GB M2 that is ~5.33 GB).
    """
    max_ws = 0
    try:
        max_ws = int(mx.metal.device_info().get("max_recommended_working_set_size", 0))
    except Exception:  # noqa: BLE001
        pass
    try:
        target = int(wired_gb * 1024**3)
        if max_ws:
            target = min(target, max_ws)
        mx.set_wired_limit(target)
    except Exception:  # noqa: BLE001
        pass
    try:
        mx.set_cache_limit(cache_mb * 1024**2)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def build_model(args):
    """Build the LM, load weights, and return (model, text_tokenizer, audio_tokenizer, cfg)."""
    cfg = configs.get_config(args.size)

    if args.size == "7b":
        model, text_tok, audio_tok, cfg = model_io.load_moshi(
            quantized=args.model,
            repo=args.repo,
            moshi_weights=args.moshi_weights,
            mimi_weights=args.mimi_weights,
            tokenizer=args.tokenizer,
            config_path=args.config,
            dtype="bfloat16",
        )
        if args.lora:
            n = lora_lib.load_lora(model, args.lora, rank=args.lora_rank, scale=args.lora_scale)
            log(f"[cli] applied LoRA to {n} layers from {args.lora}")
            # The frozen base already nearly fills the 8 GB GPU budget; release
            # any temporaries from adapter loading before generation starts.
            try:
                mx.clear_cache()
            except Exception:  # noqa: BLE001
                pass
        return model, text_tok, audio_tok, cfg

    # Small custom model: fresh weights unless a checkpoint is provided.
    dtype = getattr(mx, "bfloat16")
    model = models.Lm(cfg)
    model.set_dtype(dtype)

    if args.checkpoint:
        log(f"[cli] loading checkpoint {args.checkpoint}")
        model.load_weights(str(args.checkpoint), strict=False)
    else:
        log("[cli] WARNING: no --checkpoint given; using randomly initialised weights.")
    if args.lora:
        n = lora_lib.load_lora(model, args.lora, rank=args.lora_rank, scale=args.lora_scale)
        log(f"[cli] applied LoRA to {n} layers from {args.lora}")

    # Mimi + text tokenizer still come from the released repo (they are shared).
    repo = args.repo or model_io.DEFAULT_REPOS["q4"]
    mimi_path = model_io._download(repo, model_io.MIMI_FILE, args.mimi_weights)
    tok_path = model_io._download(repo, model_io.TOKENIZER_FILE, args.tokenizer)
    text_tok = sentencepiece.SentencePieceProcessor(tok_path)
    n_q = max(cfg.generated_codebooks, cfg.other_codebooks) or configs.AUDIO_CODEBOOKS
    audio_tok = rustymimi.Tokenizer(str(mimi_path), num_codebooks=n_q)
    return model, text_tok, audio_tok, cfg


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
class MoshiSession:
    """Stateful speech-to-speech generator (one continuous dialogue)."""

    def __init__(self, model, text_tokenizer, audio_tokenizer, cfg, *, temp=0.8,
                 top_k_text=25, top_k_audio=250, cfg_coef=1.0, max_steps=4000):
        self.model = model
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.cfg = cfg
        self.generated_codebooks = cfg.generated_codebooks
        self.other_codebooks = cfg.other_codebooks
        self.gen = models.LmGen(
            model=model,
            max_steps=max_steps,
            text_sampler=utils.Sampler(top_k=top_k_text, temp=temp),
            audio_sampler=utils.Sampler(top_k=top_k_audio, temp=temp),
            cfg_coef=cfg_coef,
            check=False,
        )
        self.ct = None
        if model.condition_provider is not None:
            self.ct = model.condition_provider.condition_tensor("description", "very_good")
        model.warmup(self.ct)
        try:
            mx.clear_cache()
        except Exception:  # noqa: BLE001
            pass
        self.text_chunks: list[str] = []

    def step(self, pcm_frame: np.ndarray):
        """pcm_frame: float32 array of shape (1, FRAME_SIZE). Returns decoded pcm or None."""
        other = self.audio_tokenizer.encode_step(pcm_frame[None, 0:1])
        other = mx.array(other).transpose(0, 2, 1)[:, :, : self.other_codebooks]
        text_token = self.gen.step(other[0], self.ct)
        text_token = int(text_token[0].item())
        if text_token not in (0, 3):
            piece = self.text_tokenizer.id_to_piece(text_token).replace("\u2581", " ")
            self.text_chunks.append(piece)

        audio_tokens = self.gen.last_audio_tokens()
        out_pcm = None
        if audio_tokens is not None and self.generated_codebooks > 0:
            tokens = np.array(audio_tokens[:, :, None]).astype(np.uint32)
            out_pcm = self.audio_tokenizer.decode_step(tokens)
        return out_pcm

    def text(self) -> str:
        return "".join(self.text_chunks)


def run_file(args):
    model, text_tok, audio_tok, cfg = build_model(args)

    in_pcms, sr = sphn.read(args.input, sample_rate=SAMPLE_RATE)
    if in_pcms.ndim == 2 and in_pcms.shape[0] > 1:
        in_pcms = in_pcms[:1]
    n_steps = in_pcms.shape[-1] // FRAME_SIZE
    if args.max_steps:
        n_steps = min(n_steps, args.max_steps)
    if n_steps == 0:
        raise SystemExit("input audio is shorter than one frame (80 ms)")
    log(f"[cli] input: {args.input}  ({in_pcms.shape[-1]/SAMPLE_RATE:.1f}s, {n_steps} steps)")

    sess = MoshiSession(model, text_tok, audio_tok, cfg, temp=args.temp,
                        top_k_text=args.top_k_text, top_k_audio=args.top_k_audio,
                        cfg_coef=args.cfg, max_steps=n_steps + 5)

    out_chunks = []
    t0 = time.time()
    report_every = max(1, n_steps // 10)
    for idx in range(n_steps):
        pcm = in_pcms[:, idx * FRAME_SIZE : (idx + 1) * FRAME_SIZE].astype(np.float32)
        if pcm.shape[-1] < FRAME_SIZE:
            pcm = np.pad(pcm, ((0, 0), (0, FRAME_SIZE - pcm.shape[-1])))
        out = sess.step(pcm)
        if out is not None:
            out_chunks.append(out)
        if (idx + 1) % report_every == 0:
            dt = time.time() - t0
            log(f"[cli]   step {idx + 1}/{n_steps} ({dt:.1f}s, {(idx + 1) / dt:.2f} step/s)")
    dt = time.time() - t0
    log(f"[cli] generated {n_steps} steps in {dt:.1f}s ({n_steps/dt:.1f} step/s)")

    if args.show_text:
        log("[cli] inner monologue:")
        print(sess.text())

    if args.output:
        if not out_chunks:
            raise SystemExit("model produced no audio")
        out_pcm = np.concatenate(out_chunks, axis=-1)
        rustymimi.write_wav(args.output, out_pcm[0, 0], sample_rate=SAMPLE_RATE)
        log(f"[cli] wrote {out_pcm.shape[-1]/SAMPLE_RATE:.1f}s to {args.output}")


def run_mic(args):
    import asyncio
    import queue

    import sounddevice as sd

    model, text_tok, audio_tok, cfg = build_model(args)
    sess = MoshiSession(model, text_tok, audio_tok, cfg, temp=args.temp,
                        top_k_text=args.top_k_text, top_k_audio=args.top_k_audio,
                        cfg_coef=args.cfg, max_steps=args.max_steps or 4000)

    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()

    def on_input(indata, frames, time_info, status):  # noqa: ARG001
        in_q.put_nowait(indata[:, 0].astype(np.float32).copy())

    def on_output(outdata, frames, time_info, status):  # noqa: ARG001
        try:
            chunk = out_q.get_nowait()
            outdata[:, 0] = chunk
        except queue.Empty:
            outdata.fill(0)

    in_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=FRAME_SIZE, callback=on_input)
    out_stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=FRAME_SIZE, callback=on_output)

    log("[cli] microphone mode - speak into the mic (Ctrl-C to stop).")
    log("[cli] Tip: use headphones so the model does not hear itself.")

    def worker():
        while True:
            pcm = in_q.get()
            out = sess.step(pcm[None, :])
            if out is not None:
                out_q.put_nowait(out[0, 0])

    import threading

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        with in_stream, out_stream:
            while True:
                time.sleep(0.2)
    except KeyboardInterrupt:
        log("\n[cli] stopped.")
        if args.show_text:
            print(sess.text())


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="moshi_ru",
        description="Moshi speech-to-speech on Apple Metal (MLX).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--size", default="7b", choices=["7b", "nano", "tiny", "small", "medium"],
                   help="Model size. '7b' uses pretrained Moshiko; the others are small trainable configs.")
    p.add_argument("--input", help="Input wav file (offline mode).")
    p.add_argument("--output", help="Output wav file (offline mode).")
    p.add_argument("--mic", action="store_true", help="Run interactive microphone mode.")

    # 7B loading options.
    p.add_argument("--model", default="q4", choices=["q4", "q8", "bf16"],
                   help="Quantisation of the pretrained 7B model.")
    p.add_argument("--repo", default=None, help="Override the Hugging Face repo id.")
    p.add_argument("--moshi-weights", default=None, help="Local path to Moshi weights.")
    p.add_argument("--mimi-weights", default=None, help="Local path to Mimi weights.")
    p.add_argument("--tokenizer", default=None, help="Local path to the text tokenizer.")
    p.add_argument("--config", default=None, help="Local LM config json (newer models).")
    p.add_argument("--checkpoint", default=None, help="Checkpoint for a small model.")

    # LoRA.
    p.add_argument("--lora", default=None, help="Path to an MLX LoRA adapter safetensors.")
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-scale", type=float, default=2.0)

    # Sampling.
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-k-text", type=int, default=25)
    p.add_argument("--top-k-audio", type=int, default=250)
    p.add_argument("--cfg", type=float, default=1.0, help="Classifier-free-guidance coefficient.")
    p.add_argument("--max-steps", type=int, default=0, help="Cap generation length (0 = unlimited).")

    p.add_argument("--show-text", action="store_true", help="Print the model's inner monologue.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    mx.random.seed(299792458)
    # The frozen 4.8 GB base already nearly fills the 5.33 GB GPU budget. A LoRA
    # adapter pushes peak memory above it, so for LoRA we let macOS swap instead
    # of failing the Metal command buffer.
    if args.lora:
        try:
            mx.set_wired_limit(0)
        except Exception:  # noqa: BLE001
            pass
        try:
            mx.set_cache_limit(256 * 1024**2)
        except Exception:  # noqa: BLE001
            pass
    else:
        configure_memory()
    if args.mic:
        run_mic(args)
    elif args.input:
        run_file(args)
    else:
        build_parser().print_help()
        raise SystemExit("Provide --input/--output or --mic.")


if __name__ == "__main__":
    main()
