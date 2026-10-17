#!/usr/bin/env python3
"""Numerically verify the MLX delay/undelay helpers against the PyTorch reference.

The reference functions are tiny (moshi/moshi/moshi/models/lm_utils.py); they are
inlined here so the test does not need the whole ``moshi`` package (which needs
``einops`` etc.).

    python tests/test_delay_equivalence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from metal.train_forward import delay_sequence, undelay_sequence  # noqa: E402


def _delay_sequence(delays, tensor, padding):  # reference
    B, K, T = tensor.shape
    outs = []
    for k, d in enumerate(delays):
        line = tensor[:, k].roll(d, dims=1)
        if d > 0:
            line[:, :d] = padding[:, k]
        outs.append(line)
    return torch.stack(outs, dim=1)


def _undelay_sequence(delays, tensor, fill=float("nan")):  # reference
    B, K, T, *_ = tensor.shape
    mask = torch.ones(B, K, T, dtype=torch.bool)
    outs = []
    if all(d == 0 for d in delays):
        return tensor, mask
    for k, d in enumerate(delays):
        line = tensor[:, k].roll(-d, dims=1)
        if d > 0:
            line[:, -d:] = fill
            mask[:, k, -d:] = 0
        outs.append(line)
    return torch.stack(outs, dim=1), mask


def check(delays, B=2, T=13, C=7):
    rng = np.random.default_rng(0)
    K = len(delays)
    t = mx.array(rng.integers(0, 50, size=(B, K, T)).astype(np.int32))
    pad = mx.array(rng.integers(0, 50, size=(B, K, 1)).astype(np.int32))
    mine = np.array(delay_sequence(delays, t, pad))
    ref = _delay_sequence(delays, torch.tensor(np.array(t)), torch.tensor(np.array(pad))).numpy()
    ok_delay = np.array_equal(mine, ref)

    t2 = mx.array(rng.standard_normal((B, K, T, C)).astype(np.float32))
    mine2, mask2 = undelay_sequence(delays, t2, float("nan"))
    ref2, maskr = _undelay_sequence(delays, torch.tensor(np.array(t2)))
    ok_undelay = np.allclose(np.array(mine2), ref2.numpy(), equal_nan=True)
    ok_mask = np.array_equal(np.array(mask2), maskr.numpy())
    status = "OK " if (ok_delay and ok_undelay and ok_mask) else "FAIL"
    print(f"[{status}] delays={delays} delay={ok_delay} undelay={ok_undelay} mask={ok_mask}")
    return ok_delay and ok_undelay and ok_mask


def main() -> None:
    configs = [
        [0, 1, 1, 1, 1, 1, 1, 1],                       # main-only, 8 codebooks
        [0] + [0, 1, 1, 1, 1, 1, 1, 1] * 2,             # Moshiko v0.1, 16 codebooks
        [0, 0, 0, 0],                                    # all-zero (early return path)
        [0, 2, 3, 1, 4],                                 # irregular delays
    ]
    ok = all(check(d) for d in configs)
    print("ALL PASS" if ok else "SOME FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
