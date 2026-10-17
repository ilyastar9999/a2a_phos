"""Metal (MLX) trainer for Moshi-family speech models.

Two modes:

* ``--size nano|tiny|small|medium`` trains a small Moshi-architecture model
  *from scratch* on pre-encoded codes.  This is the path that actually fits on an
  8 GB Apple Silicon Mac.
* ``--size 7b --model q4`` performs QLoRA fine-tuning of the frozen 4-bit 7B
  Moshiko model (only the LoRA adapters are trained).

Codes are ``.npz`` files with a single ``codes`` array of shape ``[1 + n_q, T]``
(see :mod:`scripts.encode_codes`).  Loss matches ``moshi-finetune``: masked
cross-entropy summed over the text head and the DepFormer audio head.

Example::

    python train/train_mlx.py --size nano --data-dir data/toy/codes \
        --out runs/nano --steps 400 --batch-size 4 --seq-len 96
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten, tree_map  # noqa: E402

from metal import configs, model_io  # noqa: E402
from metal import lora as lora_lib  # noqa: E402
from metal.train_forward import MoshiForward, moshi_loss_components  # noqa: E402
from moshi_mlx import models  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


class CodeDataset:
    """Random fixed-length windows from a directory of ``.npz`` code files."""

    def __init__(self, files: list[Path], seq_len: int, seed: int = 0):
        if not files:
            raise SystemExit("no .npz files found")
        self.files = files
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def sample(self, batch: int) -> mx.array:
        out = []
        for _ in range(batch):
            codes = np.load(self.files[self.rng.integers(len(self.files))])["codes"]
            k, t = codes.shape
            if t >= self.seq_len:
                s = int(self.rng.integers(0, t - self.seq_len + 1))
                crop = codes[:, s : s + self.seq_len]
            else:
                crop = np.full((k, self.seq_len), -1, dtype=np.int32)
                crop[:, :t] = codes
            out.append(crop)
        return mx.array(np.stack(out).astype(np.int32))

    def __len__(self) -> int:
        return len(self.files)


def build_model(args):
    if args.size == "7b":
        log(f"[train] loading 4-bit 7B base ({args.model}) and applying LoRA")
        model, _, _, cfg = model_io.load_moshi(quantized=args.model)
        n = lora_lib.apply_lora(model, rank=args.lora_rank, scale=args.lora_scale)
        log(f"[train] LoRA applied to {n} layers (rank={args.lora_rank}, scale={args.lora_scale})")
        trainable = tree_flatten(model.trainable_parameters())
        n_train = sum(v.size for _, v in trainable)
        log(f"[train] trainable parameters: {n_train/1e6:.2f}M")
        return model, cfg, True

    cfg = configs.small_moshi_config(args.size)
    model = models.Lm(cfg)
    model.set_dtype(mx.bfloat16)
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    log(f"[train] small model '{args.size}': {n_params/1e6:.1f}M parameters")
    return model, cfg, False


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="nano", choices=["7b", "nano", "tiny", "small", "medium"])
    ap.add_argument("--model", default="q4", choices=["q4", "q8", "bf16"], help="Base precision for --size 7b.")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/default"))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=128,
                    help="Frames per sample (80 ms each; 128 = 10.2 s). Small "
                         "batch/seq windows never fit tiny datasets; see README.")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-scale", type=float, default=2.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--cooldown-every", type=int, default=0,
                    help="Sleep for --cooldown seconds every N steps, to keep a "
                         "fanless machine from overheating (0 = disabled).")
    ap.add_argument("--cooldown", type=float, default=3.0,
                    help="Seconds to sleep during a --cooldown-every pause.")
    ap.add_argument("--resume", default=None, help="Checkpoint (small) or LoRA adapter (7b) to resume from.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    try:
        mx.set_cache_limit(256 * 1024**2)
    except Exception:  # noqa: BLE001
        pass
    mx.random.seed(args.seed)

    files = sorted(args.data_dir.glob("*.npz"))
    dataset = CodeDataset(files, args.seq_len, seed=args.seed)
    log(f"[train] {len(dataset)} clips from {args.data_dir}")

    model, cfg, is_lora = build_model(args)

    if args.resume:
        if is_lora:
            lora_lib.load_lora(model, args.resume, apply=False)
        else:
            model.load_weights(str(args.resume), strict=True)
        log(f"[train] resumed from {args.resume}")

    def loss_fn(model, codes):
        total, _, _ = moshi_loss_components(MoshiForward(model), codes)
        return total

    value_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    def save(tag: str) -> None:
        if is_lora:
            lora_lib.save_lora(model, args.out / f"lora_{tag}.safetensors")
        else:
            model.save_weights(str(args.out / f"model_{tag}.safetensors"))
        cfg_dict = {
            "size": args.size, "model": args.model, "lora": is_lora,
            "lora_rank": args.lora_rank, "lora_scale": args.lora_scale, "step": tag,
        }
        (args.out / "config.json").write_text(json.dumps(cfg_dict, indent=2))
        log(f"[train] saved checkpoint {tag} -> {args.out}")

    t0 = time.time()
    running = None
    for step in range(args.steps):
        opt.learning_rate = cosine_lr(step, args.steps, args.lr, args.warmup)
        acc = None
        step_loss = 0.0
        for _ in range(args.grad_accum):
            codes = dataset.sample(args.batch_size)
            loss, grads = value_and_grad(model, codes)
            acc = grads if acc is None else tree_map(lambda a, b: a + b, acc, grads)
            step_loss += float(loss)
            mx.eval(loss)
        acc = tree_map(lambda a: a / args.grad_accum, acc)
        if args.grad_clip:
            acc, _ = optim.clip_grad_norm(acc, args.grad_clip)
        opt.update(model, acc)
        mx.eval(model.parameters(), opt.state)

        if args.cooldown_every and (step + 1) % args.cooldown_every == 0 and args.cooldown > 0:
            time.sleep(args.cooldown)

        step_loss /= args.grad_accum
        running = step_loss if running is None else 0.9 * running + 0.1 * step_loss

        if (step + 1) % args.log_every == 0 or step == 0:
            # Component losses for visibility.
            eval_codes = dataset.sample(args.batch_size)
            _, tl, al = moshi_loss_components(MoshiForward(model), eval_codes)
            mx.eval(tl, al)
            dt = time.time() - t0
            try:
                active = mx.get_active_memory()
            except AttributeError:
                active = mx.metal.get_active_memory()
            log(
                f"[train] step {step+1:5d}/{args.steps}  loss={step_loss:.4f} (ema {running:.4f})  "
                f"text={float(tl):.4f} audio={float(al):.4f}  lr={opt.learning_rate:.2e}  "
                f"{dt:.0f}s  {float(active)/1e9:.2f}GB"
            )

        if args.save_every and (step + 1) % args.save_every == 0:
            save(str(step + 1))

    save("final")
    log(f"[train] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
