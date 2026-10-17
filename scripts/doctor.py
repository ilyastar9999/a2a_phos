#!/usr/bin/env python3
"""Print an environment / capability report for Metal Moshi.

Checks MLX + Metal, the GPU memory budget, whether the pretrained repos are
cached, and runs a one-batch forward pass on the smallest config so you know the
toolchain works before downloading anything large.

    python scripts/doctor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402


def human(n: int) -> str:
    return f"{n / 1024**3:.2f} GiB"


def main() -> None:
    print("== MLX / Metal ==")
    print("mlx version      :", getattr(mx, "__version__", "unknown"))
    print("metal available  :", mx.metal.is_available())

    info = mx.metal.device_info()
    print("device           :", info.get("device_name"))
    print("architecture     :", info.get("architecture"))
    print("memory size      :", human(int(info.get("memory_size", 0))))
    max_ws = int(info.get("max_recommended_working_set_size", 0))
    print("max working set  :", human(max_ws))
    print("=> 7B q4 needs ~4.8 GiB; QLoRA leaves ~%.2f GiB for activations" % ((max_ws - 4.9e9) / 1e9))

    print("\n== Hugging Face cache ==")
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
        for repo in sorted(cache.repos, key=lambda r: r.repo_id):
            if "moshiko" in repo.repo_id or "moshi" in repo.repo_id:
                print(f"  {repo.repo_id:35s} {human(repo.size_on_disk)}")
    except Exception as exc:  # noqa: BLE001
        print("  (could not scan cache:", exc, ")")

    print("\n== Small-model forward pass ==")
    from metal import configs
    from metal.train_forward import MoshiForward, moshi_loss_components
    from moshi_mlx import models

    cfg = configs.small_moshi_config("nano")
    model = models.Lm(cfg)
    model.set_dtype(mx.bfloat16)
    codes = mx.random.randint(0, 2048, (1, 1 + cfg.audio_codebooks, 32)).astype(mx.int32)
    total, text, audio = moshi_loss_components(MoshiForward(model), codes)
    mx.eval(total, text, audio)
    print(f"  nano loss on random input: total={float(total):.2f} "
          f"text={float(text):.2f} audio={float(audio):.2f}")
    print("  OK - Metal training forward works.")

    print("\n== CLI check ==")
    cli = Path(__file__).resolve().parents[1] / "cli" / "moshi_ru.py"
    print(f"  run:  .venv/bin/python {cli.relative_to(cli.parents[1])} --input <wav> --output <wav>")


if __name__ == "__main__":
    main()
