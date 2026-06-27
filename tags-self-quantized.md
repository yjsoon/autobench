---
layout: default
title: Tags · Self-Quantized
permalink: /tags/self-quantized/
---

# Self-Quantized

Models we **quantized ourselves** (the `Self-Quantized` tag) — here, REAP-pruned MoE
checkpoints taken to **NVFP4 W4A16** (compressed-tensors) so they serve on a single DGX
Spark. This page is the full recipe; the runnable tooling is in
[`recipes/`]({{ '/recipes/' | relative_url }}). [← all tag kinds]({{ '/tags/' | relative_url }})

## Why W4A16 NVFP4

Weights → NVFP4 (E2M1, block-16, FP8-E4M3 micro-scale + FP32 global scale ≈ 4.5 bits/value);
activations stay BF16. It is the **highest-accuracy FP4 option and needs no calibration
forward pass** — a pure weight transform. On the GB10 the NVFP4 win is **memory-bandwidth**
(via the marlin dequant kernel), not FP4 compute, so W4A16 keeps ~2% accuracy vs W4A4's >4%
on an already double-compressed (REAP) model — worth far more than the modest W4A4 speedup.

## The memory wall (and why off-the-shelf tools fail)

REAP MoE checkpoints exceed the box's 121 GB RAM (GLM-Air 159 GB BF16; MiniMax ~260 GB after
FP8→BF16 dequant). Both stock quantizers fall over:

- **NVIDIA ModelOpt** `mtq.quantize` eagerly materializes the **whole** model in RAM during
  the quantize step → OOM-killed (peaked 111 GB and climbing on GLM-Air). `max_memory` caps
  and accelerate disk-offload bound the *load* but the *quantize* pass pulls everything back
  in. (SIGKILL leaves no traceback; `nohup` block-buffers Python stdout, so deaths look
  silent — use `python -u`.)
- **LLM Compressor** bounds the load fine (disk offload genuinely engages), but
  compressed_tensors 0.17's `from_accelerate` converter asserts every offloaded param is on
  the `meta` device and rejects accelerate's offload layout — fails whether the overflow tier
  is CPU or disk.

## The fix — a shard-by-shard streaming quantizer

[`recipes/streaming_quantize.py`]({{ '/recipes/streaming_quantize.py' | relative_url }})
never loads the full model. It processes **one shard at a time** and, for each Linear weight,
reuses compressed_tensors' own primitives so the on-disk format is byte-identical to what
vLLM's `compressed-tensors` loader expects:

- `generate_gparam` → fp32 per-tensor global scale,
- `compute_dynamic_scales_and_zp` → per-block-16 fp8_e4m3 micro-scale,
- `quantize` + `pack_fp4_to_uint8` → packed `weight_packed` (uint8).

Output is compressed-tensors `nvfp4-pack-quantized` (`weight_packed` uint8 / `weight_scale`
fp8_e4m3 / `weight_global_scale` fp32). Peak RAM ≈ **one shard (~5 GB)** + GPU working set, so
it scales to any size (~30 s/shard, ~16 min for GLM-Air). It accepts **BF16 and block-FP8**
sources — block-FP8 (DeepSeek-style `weight` + `weight_scale_inv` [128,128]) is dequantized to
BF16 on the fly, so no giant BF16 intermediate is written. `lm_head`, MoE routers (`*.gate`),
embeddings and any MTP/nextn layer stay dense (BF16) and are listed in the config `ignore`.

### ⚠️ The correctness rule that bites: fused-layer shared global scale

vLLM **fuses** parallel projections and uses **one** NVFP4 `weight_global_scale` per fused
group: `q_proj+k_proj+v_proj → qkv_proj`; `gate_proj+up_proj → gate_up_proj` (per dense MLP,
per expert, per shared-expert; SwiGLU `w1`+`w3`). If each weight gets its own global scale,
vLLM applies the **first** member's scale to all of them → non-first members (k, v, up) are
dequantized wrong → **coherent-looking garbage** ("exact exact exact…"), even though the
checkpoint loads fine and per-weight round-trip error looks normal (~10%, which *is* normal
for 4-bit). vLLM warns at load: *"the weight global scale is different for parallel layers"* /
*"w1_weight_global_scale must match w3_weight_global_scale"* — heed it.

The fix is **two passes**: pass 1 collects per-group min/max; pass 2 quantizes each fused group
against a **shared** global scale (max amax over the group). The per-block `weight_scale` can
stay per-tensor (rows concatenate fine under fusion); only the global scale must be shared.

```python
# pass 1 → one shared fp32 global scale per fused projection group
def group_global_scales(stats: dict) -> dict:
    groups = defaultdict(list)
    for base in stats:                         # base = "...self_attn.q_proj", etc.
        groups[fusion_group_key(base)].append(base)
    out = {}
    for members in groups.values():
        gmin = min(stats[b][0] for b in members)
        gmax = max(stats[b][1] for b in members)
        gs = generate_gparam(torch.tensor(gmin), torch.tensor(gmax))  # fp32 [1]
        for b in members:
            out[b] = gs                        # SAME object shared across the group
    return out

# vLLM's fusion map. MLA (DeepSeek) fuses differently (q_a/q_b, kv_a/kv_b) — revisit per arch.
def fusion_group_key(base: str) -> str:
    parent, leaf = base.rsplit(".", 1)
    if leaf in ("q_proj", "k_proj", "v_proj") and parent.endswith(".self_attn"):
        return parent + "::QKV"
    if leaf in ("gate_proj", "up_proj"):
        return parent + "::GATEUP"
    if leaf in ("w1", "w3"):
        return parent + "::W13"
    return base                                # o_proj, down_proj/w2, MLA latent → singletons
```

## Run it

```bash
# env: llmcompressor 0.12 / compressed_tensors 0.17 / torch 2.12+cu130 (see
# recipes/quantizer-pyproject.toml). TORCH_COMPILE_DISABLE=1 is REQUIRED — this box's
# gcc can't build triton/inductor's CUDA util (aarch64).
cp recipes/config.env.example config.env     # set SRC_REPO / SRC_DIR / OUT_DIR_STREAM
source config.env

python recipes/download.py                                         # resumable source download
TORCH_COMPILE_DISABLE=1 python recipes/streaming_quantize.py --check   # toolchain + fused-scale smoke test
TORCH_COMPILE_DISABLE=1 python recipes/streaming_quantize.py           # full run (~30s/shard)

# Serve / verify. flashinfer_cutlass REJECTS W4A16 NVFP4 — marlin MoE is mandatory.
vllm serve "$OUT_DIR_STREAM" \
  --quantization compressed-tensors --moe-backend marlin --trust-remote-code
```

## Publishing to the HF Hub

The Xet backend uploads are chunk-deduplicated and **resumable by re-running the same
command** — use `upload-large-folder` for 50–100 GB checkpoints, and don't delete-and-restart.

```bash
export HF_XET_HIGH_PERFORMANCE=1                # throughput knob for 50-100 GB
hf auth login                                  # needs a Write token
hf repo create <you>/<base>-NVFP4 --repo-type model
hf upload-large-folder <you>/<base>-NVFP4 --repo-type=model "$OUT_DIR_STREAM" --num-workers=16
```

- `--repo-type=model` on `upload-large-folder` is **mandatory** (forgetting it → wrong repo
  type, full re-upload).
- Model card: `base_model_relation: quantized`, `base_model:` listing the chain (original →
  REAP → our NVFP4), inherit the **base model's license** (don't invent one), and tag
  `nvfp4, fp4, w4a16, compressed-tensors, vllm, moe`.
- Keep the full `config.json`, the shard index, every shard, and **all tokenizer files** (a
  missing tokenizer file is the most common "loads locally, breaks on Hub" failure).
- If a worker wedges (all sockets in `CLOSE-WAIT`, 0 % CPU), kill and re-run — it resumes from
  the local cache; committed files are skipped.

## Models

{% assign group = site.configs | where_exp: "c", "c.tags contains 'Self-Quantized'" %}
<h2 id="self-quantized">Self-Quantized <span class="muted">({{ group | size }})</span></h2>
{% if group.size == 0 %}
*None tagged yet.*
{% else %}
{% include config-table.html configs=group %}
{% endif %}
