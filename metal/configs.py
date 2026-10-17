"""Model configuration presets.

Two families:

* ``7b`` – the official Moshiko v0.1 config (``models.config_v0_1``), used with
  pretrained weights. Fine-tuning this on 8 GB is only possible in 4-bit QLoRA.
* ``nano``/``tiny``/``small``/``medium`` – small Moshi-architecture models that
  are realistically trainable **from scratch** on Apple Metal. They reuse the
  released Mimi codec and the same text/audio vocabularies, so they are fully
  compatible with the CLI and the 7B data pipeline.
"""

from __future__ import annotations

from moshi_mlx import models
from moshi_mlx.models.lm import DepFormerConfig, LmConfig
from moshi_mlx.modules.transformer import TransformerConfig

# Vocabulary sizes used by the released Moshiko v0.1 models / tokenizers.
TEXT_CARD = 32000
AUDIO_CARD = 2048  # -> audio_vocab_size = 2048 + 1
AUDIO_CODEBOOKS = 16
DEP_Q = 8
# ([0] + [1]*7) * 2 : text delay then 7 audio codebooks, repeated for user/model.
AUDIO_DELAYS = [0] + [1] * 7 + [0] + [1] * 7

SMALL_PRESETS = {
    "nano": dict(d_model=256, num_heads=4, num_layers=6, dep_d_model=128, dep_heads=4, dep_layers=2),
    "tiny": dict(d_model=384, num_heads=6, num_layers=8, dep_d_model=192, dep_heads=4, dep_layers=3),
    "small": dict(d_model=512, num_heads=8, num_layers=12, dep_d_model=256, dep_heads=8, dep_layers=4),
    "medium": dict(d_model=768, num_heads=12, num_layers=16, dep_d_model=384, dep_heads=8, dep_layers=4),
}


def _transformer_cfg(
    *,
    d_model: int,
    num_heads: int,
    num_layers: int,
    dim_feedforward: int,
    max_seq_len: int = 4096,
    context: int = 750,
    max_period: int = 10000,
    positional_embedding: str = "rope",
    layer_scale=None,
    gating: bool = True,
) -> TransformerConfig:
    return TransformerConfig(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        causal=True,
        norm_first=True,
        bias_ff=False,
        bias_attn=False,
        layer_scale=layer_scale,
        context=context,
        max_period=max_period,
        use_conv_block=False,
        use_conv_bias=True,
        cross_attention=False,
        gating=gating,
        norm="rms_norm",
        positional_embedding=positional_embedding,
        conv_layout=False,
        conv_kernel_size=3,
        kv_repeat=1,
        max_seq_len=max_seq_len,
    )


def small_moshi_config(size: str = "small") -> LmConfig:
    """Build a small (sub-7B) Moshi LM config trainable on Metal."""
    if size not in SMALL_PRESETS:
        raise ValueError(f"unknown size {size!r}; choose from {sorted(SMALL_PRESETS)}")
    p = SMALL_PRESETS[size]

    transformer = _transformer_cfg(
        d_model=p["d_model"],
        num_heads=p["num_heads"],
        num_layers=p["num_layers"],
        dim_feedforward=4 * p["d_model"],
        max_seq_len=4096,
        context=750,
        max_period=10000,
        positional_embedding="rope",
    )
    depformer = DepFormerConfig(
        transformer=_transformer_cfg(
            d_model=p["dep_d_model"],
            num_heads=p["dep_heads"],
            num_layers=p["dep_layers"],
            dim_feedforward=4 * p["dep_d_model"],
            max_seq_len=4096,
            context=DEP_Q,
            max_period=8,
            positional_embedding="none",
            gating=True,
        ),
        num_slices=DEP_Q,
    )
    return LmConfig(
        transformer=transformer,
        depformer=depformer,
        text_in_vocab_size=TEXT_CARD + 1,
        text_out_vocab_size=TEXT_CARD,
        audio_vocab_size=AUDIO_CARD + 1,
        audio_codebooks=AUDIO_CODEBOOKS,
        audio_delays=AUDIO_DELAYS,
        conditioners={},
    )


def get_config(size: str) -> LmConfig:
    """Return an :class:`LmConfig` for a named size."""
    if size in ("7b", "v0.1", "moshiko"):
        return models.config_v0_1()
    return small_moshi_config(size)


def estimate_params(cfg: LmConfig) -> int:
    """Rough parameter count for logging."""
    total = 0
    for t in (cfg.transformer, cfg.depformer.transformer):
        d = t.d_model
        ff = t.dim_feedforward
        n = t.num_layers
        # approx: attn (4*d^2 incl qkv+out) + mlp (2*d*ff-ish for gated)
        attn = 4 * d * d
        mlp = 3 * d * ff  # gating uses ~2*d*(2*ff/3) in + d*ff out
        total += n * (attn + mlp)
    total += cfg.depformer.num_slices * (
        4 * cfg.depformer.transformer.d_model * cfg.depformer.transformer.d_model
    )
    total += cfg.text_in_vocab_size * cfg.transformer.d_model
    total += cfg.transformer.d_model * cfg.text_out_vocab_size
    total += cfg.audio_codebooks * cfg.audio_vocab_size * cfg.transformer.d_model
    return total
