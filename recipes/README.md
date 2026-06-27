# recipes/ — REAP + self-quantization tooling

The working tooling behind the **Self-REAP** and **Self-Quantized** tag pages, kept in
the repo so it is version-controlled and downloadable. The prose walkthroughs live on
the tag pages:

- **[/tags/self-quantized/](https://gauravmm.github.io/autobench/tags/self-quantized/)** —
  REAP/BF16/FP8 MoE checkpoint → NVFP4 W4A16 (compressed-tensors), the memory-wall
  problem, and the fused-scale correctness rule.
- **[/tags/self-reap/](https://gauravmm.github.io/autobench/tags/self-reap/)** — how REAP
  expert-pruning works and how to REAP a model yourself (and when the box can't).
- **[/tags/reap/](https://gauravmm.github.io/autobench/tags/reap/)** — all REAP-pruned
  models benchmarked here.

## Files

- `streaming_quantize.py` — shard-by-shard NVFP4 (W4A16) quantizer. Never loads the full
  model (peak RAM ≈ one shard). Handles BF16 **and** block-FP8 sources. Emits
  byte-compatible compressed-tensors `nvfp4-pack-quantized`. Two-pass so each vLLM-fused
  projection group shares one `weight_global_scale` (see the docstring + the Self-Quantized
  page for why this matters).
- `download.py` — resumable HF snapshot download of the source checkpoint.
- `config.env.example` — paths/settings; copy to `config.env` and `source` it.
- `quantizer-pyproject.toml` — the quantizer's uv env (llmcompressor 0.12 /
  compressed_tensors 0.17 / torch 2.12+cu130).

## Quick run

```bash
cp config.env.example config.env      # edit SRC_REPO / SRC_DIR / OUT_DIR_STREAM
source config.env
python download.py                     # resumable; HF_TOKEN from a gitignored .env
TORCH_COMPILE_DISABLE=1 python streaming_quantize.py --check   # toolchain smoke test
TORCH_COMPILE_DISABLE=1 python streaming_quantize.py           # full run
vllm serve "$OUT_DIR_STREAM" --quantization compressed-tensors --moe-backend marlin --trust-remote-code
```
