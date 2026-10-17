"""Metal (MLX) re-implementation of Moshi's *training* forward pass.

The released ``moshi_mlx`` package is inference-only: its ``Transformer`` never
applies a causal mask (generation feeds one token at a time through a KV cache,
which is implicitly causal).  For training we need to process a whole sequence at
once, so we

1. monkey-patch ``TransformerLayer``/``Transformer`` to accept and forward a
   ``mask`` argument (the underlying ``Attention`` already supports one);
2. re-implement ``LMModel.forward`` -- the delay / undelay bookkeeping, the
   temporal transformer pass, and the teacher-forced DepFormer pass;
3. implement the same masked cross-entropy used by ``moshi-finetune``.

Everything operates on MLX arrays and is differentiable, so it works for both
the small from-scratch models and a 4-bit 7B QLoRA.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from moshi_mlx.models.lm import Lm, LmConfig
from moshi_mlx.modules import transformer as _tf


# --------------------------------------------------------------------------- #
# Patch the transformer so it can consume a causal mask.
# --------------------------------------------------------------------------- #
def _install_mask_patch() -> None:
    if getattr(_tf.TransformerLayer.__call__, "_moshi_masked", False):
        return

    def layer_call(self, xs, cache, cross_attention_src=None, mask=None):  # noqa: ANN001
        n1 = self.norm1(xs)
        n1 = self.self_attn(n1, cache=cache.self_attn, mask=mask)
        xs = xs + self.layer_scale_1(n1)
        if self.cross_attention is not None:
            xs = self._cross_attention_block(xs, cache, cross_attention_src)
        xs = xs + self.layer_scale_2(self.gating(self.norm2(xs)))
        return xs

    def transformer_call(self, xs, cache, cross_attention_src=None, mask=None):  # noqa: ANN001
        for layer, c in zip(self.layers, cache):
            xs = layer(xs, cache=c, cross_attention_src=cross_attention_src, mask=mask)
        return xs

    layer_call._moshi_masked = True
    transformer_call._moshi_masked = True
    _tf.TransformerLayer.__call__ = layer_call
    _tf.Transformer.__call__ = transformer_call


_install_mask_patch()


def causal_mask(length: int, dtype=mx.float32) -> mx.array:
    """Additive causal mask of shape ``[1, 1, length, length]``."""
    idx = mx.arange(length)
    mask = mx.where(idx[:, None] >= idx[None, :], mx.zeros((), dtype), mx.full((), -math.inf, dtype))
    return mask[None, None]


# --------------------------------------------------------------------------- #
# Delay / undelay helpers (mirrors moshi/models/lm_utils.py).
# --------------------------------------------------------------------------- #
def delay_sequence(delays: list[int], tensor: mx.array, padding: mx.array) -> mx.array:
    """``tensor`` [B,K,T] + ``padding`` [B,K,1]; shift codebook ``k`` right by ``delays[k]``."""
    B, K, T = tensor.shape
    outs = []
    for k, delay in enumerate(delays):
        line = tensor[:, k]
        if delay > 0:
            pad = mx.broadcast_to(padding[:, k], (B, delay))
            line = mx.concatenate([pad, line[:, : T - delay]], axis=1)
        outs.append(line)
    return mx.stack(outs, axis=1)


def undelay_sequence(delays: list[int], tensor: mx.array, fill_value: float):
    """Inverse of :func:`delay_sequence` over the time axis (axis 2). Returns (tensor, mask)."""
    B, K, T = tensor.shape[0], tensor.shape[1], tensor.shape[2]
    trailing = tensor.shape[3:]
    if all(d == 0 for d in delays):
        return tensor, mx.ones((B, K, T), dtype=mx.bool_)
    outs = []
    for k, delay in enumerate(delays):
        line = tensor[:, k]
        if delay > 0:
            fill = mx.full((B, delay, *trailing), fill_value, dtype=tensor.dtype)
            line = mx.concatenate([line[:, delay:], fill], axis=1)
        outs.append(line)
    out = mx.stack(outs, axis=1)
    d = mx.array(list(delays))
    t = mx.arange(T)
    mask = t[None, :] < (T - d)[:, None]           # [K, T]: True on valid positions
    return out, mx.broadcast_to(mask[None], (B, K, T))


# --------------------------------------------------------------------------- #
# Forward pass
# --------------------------------------------------------------------------- #
class MoshiForward:
    def __init__(self, model: Lm):
        self.model = model
        cfg: LmConfig = model.cfg
        self.cfg = cfg
        self.n_q = cfg.audio_codebooks
        self.dep_q = cfg.depformer.num_slices
        self.audio_offset = 1
        self.zero_token = -1
        # text delay + per-audio-codebook delays (length: 1 + n_q).
        self.delays = [0] + list(cfg.audio_delays)
        self.text_initial = cfg.text_out_vocab_size          # card_text
        self.audio_initial = cfg.audio_vocab_size - 1        # card

    def _initial_token(self, batch: int) -> mx.array:
        text = mx.full((batch, 1, 1), self.text_initial, dtype=mx.int32)
        audio = mx.full((batch, self.n_q, 1), self.audio_initial, dtype=mx.int32)
        return mx.concatenate([text, audio], axis=1)

    def _forward_text(self, sequence: mx.array) -> tuple[mx.array, mx.array]:
        model = self.model
        inp = model.text_emb(sequence[:, 0])
        for cb in range(self.n_q):
            inp = inp + model.audio_embs[cb](sequence[:, cb + 1])
        cache = model.transformer.make_rot_cache()
        mask = causal_mask(sequence.shape[2], inp.dtype)
        out = model.transformer(inp, cache=cache, mask=mask)
        out = model.out_norm(out)
        text_logits = model.text_linear(out)  # [B, S, text_out]
        return out, text_logits

    def _forward_depformer(self, sequence: mx.array, transformer_out: mx.array) -> mx.array:
        """Teacher-forced DepFormer. ``sequence`` is the delayed codes shifted by one."""
        slices = self.model.depformer.slices
        B, T, _ = transformer_out.shape
        D = self.cfg.depformer.transformer.d_model
        cache = slices[0].transformer.make_cache()
        # The DepFormer cache is reused across the ``dep_q`` codebook slices: each
        # slice appends exactly one position.  Shrink the allocation step, otherwise
        # the default (256) preallocates gigabytes for a batch of B*T rows.
        for c in cache:
            c.self_attn.step = max(1, len(slices))
        logits = []
        for cb, sl in enumerate(slices):
            tok = sequence[:, cb]                       # [B, T]
            xs = (sl.linear_in(transformer_out) + sl.emb(tok)).reshape(B * T, 1, D)
            out = sl.transformer(xs, cache=cache)       # [B*T, 1, D]
            logits.append(sl.linear_out(out)[:, 0].reshape(B, T, -1))
        return mx.stack(logits, axis=1)                  # [B, dep_q, T, card]

    def __call__(self, codes: mx.array):
        """``codes`` [B, 1+n_q, T] -> (audio_logits, audio_mask, text_logits, text_mask)."""
        B, K, T = codes.shape
        initial = self._initial_token(B)
        delayed = delay_sequence(self.delays, codes, initial)
        delayed = mx.concatenate([initial, delayed], axis=2)  # [B,K,T+1]

        transformer_out, text_logits = self._forward_text(delayed[:, :, :-1])
        logits = self._forward_depformer(delayed[:, :, 1:], transformer_out)

        audio_logits, audio_mask = undelay_sequence(
            self.delays[self.audio_offset : self.audio_offset + self.dep_q], logits, float("nan")
        )
        audio_mask = audio_mask & (codes[:, self.audio_offset : self.audio_offset + self.dep_q] != self.zero_token)

        text_logits = text_logits[:, None]                    # [B,1,T,text_out]
        text_logits, text_mask = undelay_sequence(self.delays[:1], text_logits, float("nan"))
        text_mask = text_mask & (codes[:, :1] != self.zero_token)
        return audio_logits, audio_mask, text_logits, text_mask


# --------------------------------------------------------------------------- #
# Loss (mirrors moshi-finetune/loss.py)
# --------------------------------------------------------------------------- #
def _masked_ce(logits: mx.array, targets: mx.array, mask: mx.array,
               *, first_cb_weight: float = 1.0, padding_ids: set[int] | None = None,
               padding_weight: float = 1.0) -> mx.array:
    # logits: [B,K,T,C]; targets/mask: [B,K,T]
    B, K, T, C = logits.shape
    logits = logits.reshape(B * K * T, C).astype(mx.float32)
    targets = targets.reshape(-1).astype(mx.int32)
    weights = mask.reshape(-1).astype(mx.float32)

    if first_cb_weight != 1.0:
        w = weights.reshape(B, K, T)
        w = w.at[:, 0].multiply(first_cb_weight)
        weights = w.reshape(-1)
    if padding_ids and padding_weight != 1.0:
        for pid in padding_ids:
            weights = mx.where(targets == pid, weights * padding_weight, weights)

    safe_targets = mx.where(weights > 0, targets, mx.zeros_like(targets))
    per = nn.losses.cross_entropy(logits, safe_targets, reduction="none")
    per = mx.where(weights > 0, per * weights, mx.zeros_like(per))
    return per.sum() / mx.maximum(weights.sum(), 1.0)


def moshi_loss_components(fwd: MoshiForward, codes: mx.array, *,
                          first_codebook_weight_multiplier: float = 1.0,
                          text_padding_weight: float = 1.0,
                          text_padding_ids: set[int] | None = None):
    """Return ``(total, text_loss, audio_loss)``."""
    audio_logits, audio_mask, text_logits, text_mask = fwd(codes)
    text_loss = _masked_ce(
        text_logits, codes[:, :1], text_mask,
        padding_ids=text_padding_ids, padding_weight=text_padding_weight,
    )
    audio_loss = _masked_ce(
        audio_logits, codes[:, 1 : 1 + fwd.dep_q], audio_mask,
        first_cb_weight=first_codebook_weight_multiplier,
    )
    return text_loss + audio_loss, text_loss, audio_loss


def moshi_loss(fwd: MoshiForward, codes: mx.array, *,
               first_codebook_weight_multiplier: float = 1.0,
               text_padding_weight: float = 1.0,
               text_padding_ids: set[int] | None = None) -> mx.array:
    return moshi_loss_components(
        fwd, codes,
        first_codebook_weight_multiplier=first_codebook_weight_multiplier,
        text_padding_weight=text_padding_weight,
        text_padding_ids=text_padding_ids,
    )[0]
