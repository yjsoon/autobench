#!/usr/bin/env python3
"""Streaming NVFP4 (W4A16) quantizer — bounded memory, no full-model load.

Handles BOTH source formats:
  * BF16 checkpoints (e.g. GLM-4.5-Air-REAP), and
  * block-FP8 checkpoints (DeepSeek-V3 style: `<w>.weight` float8_e4m3fn +
    `<w>.weight_scale_inv` [128,128] blocks — e.g. MiniMax-M2.5-REAP). FP8 weights
    are dequantized to BF16 ON THE FLY per shard, so NO giant BF16 intermediate is
    written to disk.

Off-the-shelf tools fail on these models vs 121 GB RAM (ModelOpt eagerly
materializes the whole model -> OOM; LLM Compressor's accelerate->compressed_tensors
offload converter is brittle). This processes ONE SHARD AT A TIME and, for each
Linear weight, computes the NVFP4 form with compressed_tensors' own primitives so
the on-disk format is byte-identical to what vLLM's `compressed-tensors` loader
expects. Peak RAM ~= one shard (~5 GB) + GPU working set. Scales to any size.

Output = compressed-tensors `nvfp4-pack-quantized`: per quantized Linear ->
weight_packed (uint8) / weight_scale (fp8_e4m3, block-16) / weight_global_scale
(fp32). lm_head, MoE routers (*.gate), embeddings and any MTP/nextn layer stay
dense (BF16) and are listed in the config `ignore`.

CRITICAL: vLLM FUSES parallel projections and forces ONE weight_global_scale per
fused group (q/k/v->qkv, gate_proj/up_proj or w1/w3 -> gate_up). A per-tensor
global scale => garbage output. So this is 2-pass: pass 1 collects per-group
min/max, pass 2 quantizes each fused group against a shared global scale.

Run:  uv run python streaming_quantize.py            (reads SRC_DIR / OUT_DIR_STREAM)
      uv run python streaming_quantize.py --check [--limit-shards N]
      uv run python streaming_quantize.py --limit-shards N   (smoke test)
"""
import glob
import json
import os
import re
import shutil
import sys
import time

# This box's gcc/triton can't build inductor's CUDA util (aarch64 quirk); the ops
# here are simple, so force eager. Must precede torch import side effects.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch  # noqa: E402
import torch._dynamo  # noqa: E402

torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True

from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from compressed_tensors.quantization import (  # noqa: E402
    QuantizationConfig,
    QuantizationStatus,
    preset_name_to_scheme,
)
from compressed_tensors.config import CompressionFormat  # noqa: E402
from compressed_tensors.quantization.utils.helpers import (  # noqa: E402
    compute_dynamic_scales_and_zp,
    generate_gparam,
)
from compressed_tensors.quantization.lifecycle.forward import quantize  # noqa: E402
from compressed_tensors.compressors.nvfp4.helpers import pack_fp4_to_uint8  # noqa: E402

SCHEME = preset_name_to_scheme("NVFP4A16", targets=["Linear"])
WARGS = SCHEME.weights
GROUP = WARGS.group_size  # 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Set in main() from config.json. Layers with index >= NUM_LAYERS are MTP/nextn
# heads -> kept dense.
NUM_LAYERS = None
HAS_MTP = False


def dequant_block_fp8(w_fp8: torch.Tensor, scale_inv: torch.Tensor, block: int = 128) -> torch.Tensor:
    """DeepSeek-style block-FP8 -> BF16: w_bf16[i,j] = w_fp8[i,j] * scale_inv[i//block, j//block]."""
    w = w_fp8.to(DEV, torch.float32)
    s = scale_inv.to(DEV, torch.float32)
    M, N = w.shape
    s = s.repeat_interleave(block, 0)[:M].repeat_interleave(block, 1)[:, :N]
    return (w * s).to(torch.bfloat16)


def read_weight_bf16(f, name: str, keyset: set) -> torch.Tensor:
    """Load a `.weight` as BF16, dequantizing block-FP8 if a `_scale_inv` sibling exists."""
    t = f.get_tensor(name)
    scale_name = name + "_scale_inv"  # "<base>.weight" + "_scale_inv"
    if scale_name in keyset:
        return dequant_block_fp8(t, f.get_tensor(scale_name))
    return t.to(torch.bfloat16) if t.dtype != torch.bfloat16 else t


def should_quantize(name: str, shape) -> bool:
    """Quantize 2-D Linear weights, except embeddings / lm_head / MoE routers / MTP."""
    if not name.endswith(".weight") or len(shape) != 2:
        return False
    if name == "lm_head.weight" or name.endswith("embed_tokens.weight"):
        return False
    if name.endswith(".gate.weight"):  # MoE router: *.mlp.gate or *.block_sparse_moe.gate
        return False
    if NUM_LAYERS is not None:
        m = re.search(r"\.layers\.(\d+)\.", name)
        if m and int(m.group(1)) >= NUM_LAYERS:  # MTP / nextn head
            return False
    out_f, in_f = shape
    return in_f % GROUP == 0 and in_f % 2 == 0


def fusion_group_key(base: str) -> str:
    """vLLM fuses parallel projections -> ONE global scale per group.
    q/k/v -> qkv; gate_proj/up_proj -> gate_up; w1/w3 (SwiGLU: w1=gate, w3=up) -> w13.
    Singletons (o_proj, down_proj/w2, MLA latent projs) keep a per-tensor scale.
    """
    parent, leaf = base.rsplit(".", 1)
    if leaf in ("q_proj", "k_proj", "v_proj") and parent.endswith(".self_attn"):
        return parent + "::QKV"
    if leaf in ("gate_proj", "up_proj"):
        return parent + "::GATEUP"
    if leaf in ("w1", "w3"):
        return parent + "::W13"
    return base


def build_ignore() -> list:
    ig = ["lm_head", "re:.*\\.gate$"]  # router: mlp.gate or block_sparse_moe.gate
    if HAS_MTP:
        ig.append(f"re:model\\.layers\\.{NUM_LAYERS}\\..*")
    return ig


@torch.no_grad()
def quantize_weight(w: torch.Tensor, gscale: torch.Tensor):
    """BF16 weight + shared global scale -> (weight_packed, weight_scale fp8, weight_global_scale)."""
    w = w.to(DEV)
    gscale = gscale.to(DEV)
    scale, zp = compute_dynamic_scales_and_zp(w, WARGS, module=None, global_scale=gscale)
    q = quantize(x=w, scale=scale, global_scale=gscale, zero_point=zp, args=WARGS)
    return pack_fp4_to_uint8(q).cpu(), scale.to(torch.float8_e4m3fn).cpu(), gscale.float().cpu()


@torch.no_grad()
def collect_stats(src: str, shards: list) -> dict:
    """Pass 1: per quantizable weight -> (min, max) of its BF16 (dequantized) values."""
    stats = {}
    for si, shard in enumerate(shards, 1):
        with safe_open(os.path.join(src, shard), framework="pt") as f:
            keyset = set(f.keys())
            for n in keyset:
                if not n.endswith(".weight"):
                    continue
                shape = f.get_slice(n).get_shape()
                if not should_quantize(n, shape):
                    continue
                w = read_weight_bf16(f, n, keyset)
                mn, mx = torch.aminmax(w.to(DEV))
                stats[n[: -len(".weight")]] = (float(mn), float(mx))
        print(f"  [pass1 {si}/{len(shards)}] {shard}: {len(stats)} weights scanned", flush=True)
    return stats


def group_global_scales(stats: dict) -> dict:
    from collections import defaultdict
    groups = defaultdict(list)
    for base in stats:
        groups[fusion_group_key(base)].append(base)
    out = {}
    for members in groups.values():
        gmin = min(stats[b][0] for b in members)
        gmax = max(stats[b][1] for b in members)
        gs = generate_gparam(torch.tensor(gmin), torch.tensor(gmax))  # fp32 [1]
        for b in members:
            out[b] = gs  # SAME object shared across the group
    return out


def build_quant_config() -> dict:
    fmt = CompressionFormat.nvfp4_pack_quantized.value
    cfg = QuantizationConfig(
        config_groups={"group_0": SCHEME},
        quantization_status=QuantizationStatus.COMPRESSED,
        format=fmt,
        ignore=build_ignore(),
    )
    d = cfg.model_dump()
    d["quant_method"] = "compressed-tensors"
    # compressed-tensors leaves the per-group scheme's format as null, but HF Hub's
    # config validator requires it to be a string. Mirror the top-level format here.
    # Single-format model (matches top-level) -> does NOT trigger mixed-precision parsing.
    d["config_groups"]["group_0"]["format"] = fmt
    return d


def detect_arch(src: str, shards: list):
    """Set NUM_LAYERS / HAS_MTP from config.json + the actual layer indices present."""
    global NUM_LAYERS, HAS_MTP
    cfg = json.load(open(os.path.join(src, "config.json")))
    NUM_LAYERS = cfg.get("num_hidden_layers")
    max_idx = -1
    for shard in shards:
        with safe_open(os.path.join(src, shard), framework="pt") as f:
            for n in f.keys():
                m = re.search(r"\.layers\.(\d+)\.", n)
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
    HAS_MTP = NUM_LAYERS is not None and max_idx >= NUM_LAYERS
    return cfg, max_idx


def main():
    check = "--check" in sys.argv
    limit = None
    if "--limit-shards" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit-shards") + 1])

    src = os.environ.get("SRC_DIR")
    out = os.environ.get("OUT_DIR_STREAM")
    if not src:
        sys.exit("Set SRC_DIR (source checkpoint).")

    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    weight_map = idx["weight_map"]
    shards = sorted(set(weight_map.values()))
    if limit:
        shards = shards[:limit]

    cfg, max_idx = detect_arch(src, shards)
    print("=" * 70)
    print("Streaming NVFP4 (W4A16) quantizer")
    print(f"  source : {src}")
    print(f"  output : {out}")
    print(f"  arch   : {cfg.get('model_type')} | num_layers={NUM_LAYERS} max_layer_idx={max_idx} "
          f"has_mtp={HAS_MTP} | src_fp8={'quantization_config' in cfg}")
    print(f"  scheme : NVFP4A16 block-{GROUP} | dense(ignore): {build_ignore()} | dev={DEV}")
    print("=" * 70)

    if check:
        stats = collect_stats(src, shards)
        gscales = group_global_scales(stats)
        # show a fused group sharing one scale
        for base in stats:
            grp = fusion_group_key(base)
            members = [b for b in stats if fusion_group_key(b) == grp]
            if len(members) > 1:
                gss = {b.rsplit(".", 2)[-2] + "." + b.rsplit(".", 1)[-1]: float(gscales[b]) for b in members}
                print(f"--check OK: fused group {grp} shares scale, members={gss} equal={len(set(gss.values()))==1}")
                break
        json.dumps(build_quant_config())
        print(f"--check OK: {len(stats)} quantizable weights; quant_config builds.")
        return

    if not out:
        sys.exit("Set OUT_DIR_STREAM (destination).")
    os.makedirs(out, exist_ok=True)

    print("\n[pass 1/2] scanning weights for fused-group global scales...", flush=True)
    t_start = time.time()
    stats = collect_stats(src, shards)
    gscales = group_global_scales(stats)
    ngroups = len(set(id(v) for v in gscales.values()))
    print(f"  {len(stats)} quantizable weights -> {ngroups} shared global scales "
          f"({time.time()-t_start:.0f}s)", flush=True)

    print("[pass 2/2] quantizing + writing shards (dequant FP8->BF16 on the fly)...", flush=True)
    new_weight_map = {}
    nq = nkept = 0
    for si, shard in enumerate(shards, 1):
        t0 = time.time()
        out_tensors = {}
        with safe_open(os.path.join(src, shard), framework="pt") as f:
            keyset = set(f.keys())
            for n in keyset:
                if n.endswith("_scale_inv"):
                    continue  # consumed when dequantizing its weight
                if n.endswith(".weight"):
                    shape = f.get_slice(n).get_shape()
                    if should_quantize(n, shape):
                        w = read_weight_bf16(f, n, keyset)
                        base = n[: -len(".weight")]
                        p, ws, gs = quantize_weight(w, gscales[base])
                        out_tensors[base + ".weight_packed"] = p
                        out_tensors[base + ".weight_scale"] = ws
                        out_tensors[base + ".weight_global_scale"] = gs
                        nq += 1
                    else:  # dense weight (dequant FP8->BF16 if needed)
                        out_tensors[n] = read_weight_bf16(f, n, keyset).contiguous()
                        nkept += 1
                else:  # biases, norms stored without .weight, e_score_correction_bias, etc.
                    out_tensors[n] = f.get_tensor(n).contiguous()
                    nkept += 1
        save_file(out_tensors, os.path.join(out, shard), metadata={"format": "pt"})
        for k in out_tensors:
            new_weight_map[k] = shard
        print(f"[{si}/{len(shards)}] {shard}: -> {len(out_tensors)} tensors "
              f"({time.time()-t0:.0f}s, total q={nq} kept={nkept})", flush=True)

    json.dump({"metadata": dict(idx.get("metadata", {})), "weight_map": new_weight_map},
              open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)
    cfg["quantization_config"] = build_quant_config()
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    for fp in glob.glob(os.path.join(src, "*")):
        b = os.path.basename(fp)
        if b.endswith(".safetensors") or b in ("model.safetensors.index.json", "config.json"):
            continue
        if os.path.isfile(fp):
            shutil.copy2(fp, os.path.join(out, b))

    print(f"\nDONE in {time.time()-t_start:.0f}s. quantized={nq} dense={nkept} -> {out}")
    print(f"  vllm serve {out} --quantization compressed-tensors --trust-remote-code --moe-backend marlin")


if __name__ == "__main__":
    main()
