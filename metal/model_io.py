"""Load MLX Moshi models on Apple Metal.

The released MLX repos (`kyutai/moshiko-mlx-q4`, `-q8`, `-bf16`) do **not** ship a
`config.json`; the model configuration is the hard-coded `config_v0_1()` (Moshiko
v0.1). Newer Kyutai models ship a `config.json` and are loaded with
`LmConfig.from_config_dict`.

This module centralises the loading logic so the CLI and the trainer share it.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import rustymimi
import sentencepiece
from huggingface_hub import hf_hub_download

from moshi_mlx import models
from moshi_mlx.utils import loaders as mlx_loaders

# Map the short quantisation name to the official Hugging Face repo.
DEFAULT_REPOS = {
    "q4": "kyutai/moshiko-mlx-q4",
    "q8": "kyutai/moshiko-mlx-q8",
    "bf16": "kyutai/moshiko-mlx-bf16",
}

# File names inside the released repos.
QUANT_FILE = {"q4": "model.q4.safetensors", "q8": "model.q8.safetensors"}
BF16_FILE = "model.safetensors"
MIMI_FILE = "tokenizer-e351c8d8-checkpoint125.safetensors"
TOKENIZER_FILE = "tokenizer_spm_32k_3.model"


def _download(repo: str, filename: str, local: str | None = None) -> str:
    """Return a local path for `filename`, downloading from `repo` if needed."""
    if local is not None:
        return str(mlx_loaders.hf_get(local))
    return hf_hub_download(repo, filename)


def resolve_lm_config(config_path: str | None):
    """Return an :class:`models.LmConfig`.

    ``None`` -> the hard-coded Moshiko v0.1 config (used by the released MLX
    repos). Otherwise load a JSON config dict as shipped with newer models.
    """
    if config_path is None:
        return models.config_v0_1()
    with open(config_path, "r", encoding="utf-8") as fobj:
        return models.LmConfig.from_config_dict(json.load(fobj))


def load_moshi(
    quantized: str | None = "q4",
    repo: str | None = None,
    moshi_weights: str | None = None,
    mimi_weights: str | None = None,
    tokenizer: str | None = None,
    config_path: str | None = None,
    dtype: str = "bfloat16",
    verbose: bool = True,
):
    """Load a Moshi LM with its text/audio tokenizers.

    Returns ``(model, text_tokenizer, audio_tokenizer, lm_config)``.
    """
    if quantized not in (None, "q4", "q8", "bf16"):
        raise ValueError(f"invalid quantized value {quantized!r}")
    if repo is None and quantized is not None:
        repo = DEFAULT_REPOS[quantized]

    lm_config = resolve_lm_config(config_path)

    model = models.Lm(lm_config)
    model.set_dtype(getattr(mx, dtype))

    weights_name = None
    if quantized == "q4":
        weights_name = QUANT_FILE["q4"]
    elif quantized == "q8":
        weights_name = QUANT_FILE["q8"]
    elif quantized == "bf16":
        weights_name = BF16_FILE

    moshi_path = _download(repo, weights_name, moshi_weights) if weights_name else moshi_weights
    mimi_path = _download(repo, MIMI_FILE, mimi_weights)
    tokenizer_path = _download(repo, TOKENIZER_FILE, tokenizer)

    if quantized in ("q4", "q8"):
        bits = 4 if quantized == "q4" else 8
        group_size = 32 if bits == 4 else 64
        nn.quantize(model, bits=bits, group_size=group_size)

    if verbose:
        print(f"[model_io] loading weights {moshi_path}")
    model.load_weights(moshi_path, strict=True)

    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

    generated = lm_config.generated_codebooks
    other = lm_config.other_codebooks
    mimi_n_q = max(generated, other) if max(generated, other) > 0 else lm_config.audio_codebooks
    audio_tokenizer = rustymimi.Tokenizer(str(mimi_path), num_codebooks=mimi_n_q)

    return model, text_tokenizer, audio_tokenizer, lm_config


def num_codebooks(lm_config) -> int:
    return max(lm_config.generated_codebooks, lm_config.other_codebooks, 1)
