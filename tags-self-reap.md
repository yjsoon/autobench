---
layout: default
title: Tags · Self-REAP
permalink: /tags/self-reap/
---

# Self-REAP

Models we **REAP-pruned ourselves** (the `Self-REAP` tag). None yet — REAPing a model is
RAM-bound on the DGX Spark (see the feasibility note below), so so far we take Cerebras's
already-pruned checkpoints and only [quantize]({{ '/tags/self-quantized/' | relative_url }})
them. This page is the recipe for when we do prune one.
[← all tag kinds]({{ '/tags/' | relative_url }})

## What REAP does

**REAP = Router-weighted Expert Activation Pruning** (Cerebras —
[arXiv:2510.13999](https://arxiv.org/abs/2510.13999), ICLR 2026;
[github.com/CerebrasResearch/reap](https://github.com/CerebrasResearch/reap);
[blog](https://www.cerebras.ai/blog/reap)).

- **Saliency.** For each routed expert *j*, score
  `S_j = mean over calibration tokens where j is Top-K of [ router_gate_j(x) · ‖expert_output_j(x)‖₂ ]`.
  A pure forward-pass statistic — **no gradients**.
- **Uniform per-layer ratio.** Drop the lowest-`S` fraction (e.g. 25% or 50%) of experts
  in **every** MoE layer.
- **Drop, don't merge.** Pruned experts are deleted (weights not redistributed); the
  router logit row is removed and the surviving gates renormalised by `1/(1−g_j)`. Cerebras
  shows merging causes "functional subspace collapse" (a static combination can't imitate a
  dynamic router); pruning's error bound is strictly smaller whenever the router actually
  mixes experts.
- **Output = same arch, fewer experts.** Only `n_routed_experts` shrinks in `config.json`,
  so the result is a drop-in for vanilla vLLM and a clean BF16 source for downstream
  FP8/NVFP4 — exactly the input the [Self-Quantized]({{ '/tags/self-quantized/' | relative_url }})
  pipeline expects.

## How to REAP a model

Standalone tooling (it is **not** in llm-compressor or ModelOpt):

```bash
git clone https://github.com/CerebrasResearch/reap && cd reap
# pinned: torch 2.7.1 / transformers 4.55 / vllm 0.10 (vLLM only used for eval)
pip install -e .

# Single-GPU, block-wise observer: model on CPU, one decoder block to GPU at a
# time, replaying cached hidden states. Defaults: 1024 samples x 2048 tokens.
bash experiments/pruning-layerwise-cli.sh
```

**Calibration data is mandatory and must be in-domain.** Generic C4 calibration collapses
coding accuracy to ~0%; Cerebras calibrates on evol-codealpaca / Mixture-of-Thoughts /
xlam-function-calling / SWE-smith. Match the calibration set to the model's intended use.

## Feasibility on the DGX Spark (GB10, 128 GB unified) — the blocker is RAM, not tooling

The layer-wise observer does CPU↔GPU offload but **no disk streaming — the full model must
fit in CPU RAM, which on the GB10 is the same 128 GB unified pool.** So:

- **~100–110B MoE** (GLM-Air class): borderline-feasible **only if kept FP8** (~106 GB,
  tight). **BF16 at that size (~212 GB) does not fit**, which kills any dequant-then-REAP plan.
- **200B+** (MiniMax-M2, Qwen3-235B): **not feasible** on one box.

Quality: a 25% prune is ≈ near-lossless; 50% keeps generative/coding strong (~1–2 pt drops)
but knowledge/multiple-choice degrades sharply (the collapse signal). **For big models,
take Cerebras's already-pruned checkpoints and just quantize them** — which is why
everything under [REAP]({{ '/tags/reap/' | relative_url }}) here is Self-Quantized, not
Self-REAP.

## Models

{% assign group = site.configs | where_exp: "c", "c.tags contains 'Self-REAP'" %}
<h2 id="self-reap">Self-REAP <span class="muted">({{ group | size }})</span></h2>
{% if group.size == 0 %}
*None yet — see the recipe above.*
{% else %}
{% include config-table.html configs=group %}
{% endif %}
