---
layout: default
title: Tags · REAP
permalink: /tags/reap/
---

# REAP-pruned models

Configurations whose base checkpoint was pruned with **REAP** (Router-weighted Expert
Activation Pruning) — the `REAP` tag. [← all tag kinds]({{ '/tags/' | relative_url }})

**REAP** (Cerebras, [arXiv:2510.13999](https://arxiv.org/abs/2510.13999), code:
[CerebrasResearch/reap](https://github.com/CerebrasResearch/reap)) is a one-shot,
forward-pass-only compression for MoE models: it scores each routed expert by
router-gate weight × output magnitude over calibration tokens and **drops** (does not
merge) the lowest-saliency experts in every layer, renormalising the surviving router
gates. The result is the **same HF architecture with fewer experts** — a drop-in for
vanilla vLLM and a clean BF16 source for downstream FP8/NVFP4 quantization.

Two flavours appear on this site:

- **[Self-REAP]({{ '/tags/self-reap/' | relative_url }})** — models *we* REAP-pruned (none
  yet; that page documents how to do it).
- **[Self-Quantized]({{ '/tags/self-quantized/' | relative_url }})** — models we quantized
  ourselves, which here means taking Cerebras's already-REAP'd checkpoints to NVFP4.

{% assign group = site.configs | where_exp: "c", "c.tags contains 'REAP'" %}
<h2 id="reap">REAP <span class="muted">({{ group | size }})</span></h2>
{% if group.size == 0 %}
*None tagged yet.*
{% else %}
{% include config-table.html configs=group %}
{% endif %}
