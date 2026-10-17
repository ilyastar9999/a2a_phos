"""LoRA adapters for MLX Moshi models (frozen base + trainable low-rank updates).

Works on top of both dense (``nn.Linear``) and quantised (``nn.QuantizedLinear``)
base layers, i.e. it enables *QLoRA*-style fine-tuning of the 4-bit Moshi model
on Apple Metal.

Adapter files are plain MLX safetensors containing only the trainable LoRA
parameters (dotted parameter paths), so they are small (a few MB).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class LoRALinear(nn.Module):
    """Wrap a frozen linear layer with a trainable low-rank update.

    ``y = base(x) + scale * lora_b(lora_a(x))``
    """

    def __init__(self, base: nn.Module, rank: int, scale: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        in_dims, out_dims = _linear_dims(base)
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.rank = rank
        self.scale = scale
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.lora_a = nn.Linear(in_dims, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_dims, bias=False)
        # Start as an identity mapping: zero the up-projection.
        self.lora_b.weight = mx.zeros_like(self.lora_b.weight)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        xs = self.lora_a(x)
        if self.dropout is not None:
            xs = self.dropout(xs)
        return y + self.scale * self.lora_b(xs)


def _linear_dims(linear: nn.Module) -> tuple[int, int]:
    """Return ``(in_dims, out_dims)`` for a dense or quantised linear layer."""
    if hasattr(linear, "scales"):  # nn.QuantizedLinear
        group_size = linear.group_size
        out_dims = linear.scales.shape[0]
        in_dims = linear.scales.shape[1] * group_size
        return in_dims, out_dims
    out_dims, in_dims = linear.weight.shape
    return in_dims, out_dims


def _resolve_parent(root: nn.Module, path: str):
    parts = path.split(".")
    obj = root
    for p in parts[:-1]:
        if isinstance(obj, list):
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    return obj, parts[-1]


def _set_module(root: nn.Module, path: str, value: nn.Module):
    parent, leaf = _resolve_parent(root, path)
    if isinstance(parent, list):
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


DEFAULT_TARGET_SUFFIXES = (
    "self_attn.in_proj",
    "self_attn.out_proj",
    "gating.linear_in",
    "gating.linear_out",
)


def apply_lora(
    model: nn.Module,
    rank: int = 64,
    scale: float = 2.0,
    dropout: float = 0.0,
    prefixes: tuple[str, ...] = ("transformer.layers.", "depformer."),
    suffixes: tuple[str, ...] = DEFAULT_TARGET_SUFFIXES,
) -> int:
    """Freeze ``model`` and inject LoRA adapters on matching linear layers.

    Returns the number of adapted layers.
    """
    model.freeze()
    targets = []
    for name, module in model.named_modules():
        if not name:
            continue
        if not any(name.startswith(p) for p in prefixes):
            continue
        if not name.endswith(suffixes):
            continue
        if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
            targets.append(name)

    for name in targets:
        parent, leaf = _resolve_parent(model, name)
        module = parent[int(leaf)] if isinstance(parent, list) else getattr(parent, leaf)
        _set_module(model, name, LoRALinear(module, rank=rank, scale=scale, dropout=dropout))

    # Freeze everything (including the wrapped base layers), then re-enable grads
    # on the adapter submodules only.  MLX stores the "no-grad" set with names
    # relative to each owning module, so the adapters are unfrozen directly.
    model.freeze()

    def _unfreeze_adapters(_name, module):
        if isinstance(module, LoRALinear):
            module.lora_a.unfreeze()
            module.lora_b.unfreeze()

    model.apply_to_modules(_unfreeze_adapters)
    return len(targets)


def lora_parameters(model: nn.Module):
    return model.trainable_parameters()


def save_lora(model: nn.Module, path: str | Path) -> None:
    flat = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(path), flat)
    n = sum(v.size for v in flat.values())
    print(f"[lora] saved {len(flat)} tensors ({n/1e6:.2f}M params) to {path}")


def load_lora(model: nn.Module, path: str | Path, apply: bool = True, **apply_kwargs) -> int:
    """Apply LoRA to ``model`` (if needed) and load the adapter weights."""
    if apply:
        count = apply_lora(model, **apply_kwargs)
        # nn.Linear defaults to float32; on a bf16/quantised base that promotes
        # every activation to float32 and can blow the GPU budget at inference.
        # Match the base model dtype so inference stays in bf16.
        def _cast(_name, m):
            if isinstance(m, LoRALinear):
                ld = m.base.scales.dtype if hasattr(m.base, "scales") else m.base.weight.dtype
                m.lora_a.set_dtype(ld)
                m.lora_b.set_dtype(ld)

        model.apply_to_modules(_cast)
    else:
        count = 0
    weights = mx.load(str(path))
    model.load_weights(list(weights.items()), strict=False)
    return count
