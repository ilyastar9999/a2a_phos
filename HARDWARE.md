# Hardware feasibility (Apple M2, 8 GB)

Measured on the development machine with `mx.metal.device_info()` and MLX's
memory counters.

## The machine

| Property | Value |
|----------|-------|
| Chip | Apple M2 |
| Unified memory | 8.00 GiB (`memory_size = 8589934592`) |
| `max_recommended_working_set_size` | 5.33 GiB (`5726633984`) |
| MLX | 0.26.5, Metal available |

The `max_recommended_working_set_size` is the hard ceiling MLX will wire into
the GPU: **~5.33 GiB**. Any attempt to set a larger wired limit raises
`ValueError: Setting a wired limit larger than the maximum working set size is not allowed.`

## The 7B model in 4-bit (inference) — fits

```
[mem] after load       active=0.00GB peak=0.00GB   (weights load lazily)
[mem] after warmup     active=4.77GB peak=4.91GB
```

So the 4-bit 7B Moshiko model + a short generation fits under the 5.33 GiB
ceiling with ~0.4 GiB of headroom. There is essentially **no headroom for
training activations**.

### Inference speed

```
[cli] generated 51 steps in 8.4s (6.1 step/s)
```

Each step is 80 ms of audio, so 6.1 step/s ≈ **7.6× faster than real time** for
this short clip. Longer contexts slow down as the KV cache grows, but the q4 7B
model is comfortably usable for streaming on this machine.

## Small models — fits easily, and trains

| `--size` | params | training memory | notes |
|----------|--------|-----------------|-------|
| `nano`   | 41 M   | ~0.5 GB | ~4 steps/s at batch 4 × 96 frames |
| `tiny`   | 75 M   | ~0.8 GB | |
| `small`  | 131 M  | ~1.2 GB | |
| `medium` | 274 M  | ~2.5 GB | |

A 40-step sanity run on `nano` (batch 4, 96 frames) went from **loss 18.32 → 6.54**
in 10 s, with `text` loss 10.59 → 1.54 and `audio` loss 7.69 → 5.63.

## Why the PyTorch path cannot run here

`moshi-finetune` requires:

* `torch.cuda.is_available()` and calls `.cuda()` throughout,
* `torch.distributed` with NCCL,
* bf16 7B weights ≈ 14 GB (plus optimizer state and activations).

None of these are available on 8 GB Apple Silicon. The only viable local path is
MLX/Metal with a quantised base — which is what this repository implements.

## Data-encoding throughput

Mimi encoding (`rustymimi`) runs on the **CPU**, not the GPU. Measured here:

* a fresh `rustymimi.Tokenizer` costs ~0.23 s (weights are small),
* encoding one 20 s clip costs ~4 s single-threaded.

So `scripts/encode_codes.py` is the slow step for large datasets. It supports
`--workers N` (multiprocessing); on an otherwise idle machine this scales nearly
linearly across cores. Run it *after* training, or on a separate machine, to avoid
CPU contention.

`rustymimi.Tokenizer.reset()` does **not** clear the encoder's internal buffer, so
the bulk encoder deliberately creates a fresh tokenizer per utterance (after
~8192 frames the buffer would otherwise overflow).

## QLoRA: measured

QLoRA (`--size 7b --model q4`) keeps the frozen base at **4.81 GB**. The LoRA
adapters add little; the runtime cost is what matters.

**Training** (batch 1, 32 frames, 2 steps):

| rank | trainable | adapter size | active mem | time/step |
|------|-----------|--------------|-----------|-----------|
| 32   | 93.7 M    | 375 MB       | 6.68 GB (swaps) | 43–88 s |
| 8    | 23.4 M    | 94 MB        | **5.28 GB** (fits) | 22–69 s |

So **QLoRA fine-tuning runs on 8 GB** at rank 8 and short sequences — slowly, and
rank 32 tips into swap. Loss fell 13.67 → 12.8 in the two-step smoke test.

**Inference** is the harder problem: the frozen base alone peaks at 4.91 GB, and
adding the adapters raises the generation peak to **5.64 GB**, above the 5.33 GB
wired ceiling → the Metal command buffer fails with `Insufficient Memory`. If the
wired limit is relaxed so macOS can swap, generation *runs* end-to-end (verified:
a 0.5 s clip produced a WAV), but it thrashes: **0.03 step/s** (≈ 30 s per 80 ms
frame) instead of the base model's 8.6 step/s. Practical QLoRA **inference**
therefore needs a **16 GB+** Mac (or the adapter merged into a bf16 model on a
bigger machine). On 8 GB the recommended local path remains the small
from-scratch model.
