#!/usr/bin/env python3
"""Load the 7B q4 model and report MLX memory at each stage, then run a few steps."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

try:
    mx.set_wired_limit(5_000_000_000)
except Exception as exc:  # noqa: BLE001
    print("[mem] could not set wired limit:", exc, flush=True)
mx.set_cache_limit(256 * 1024**2)


def mem(tag):
    print(
        f"[mem] {tag:32s} active={mx.metal.get_active_memory()/1e9:6.2f}GB "
        f"peak={mx.metal.get_peak_memory()/1e9:6.2f}GB cache={mx.metal.get_cache_memory()/1e9:6.2f}GB",
        flush=True,
    )


from metal import model_io  # noqa: E402

mem("start")
t0 = time.time()
model, text_tok, audio_tok, cfg = model_io.load_moshi(quantized="q4")
print(f"[mem] load took {time.time()-t0:.1f}s", flush=True)
mem("after load")
model.warmup(None)
mem("after warmup")
mx.metal.clear_cache()
mem("after clear_cache")
