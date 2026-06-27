---
layout: default
title: Tags
permalink: /tags/
---

# Browse by tag

Every configuration is tagged in seven kinds, plus three provenance tags for models we compressed
ourselves. Each has its own page, where every tag value lists its configurations in a table like the
homepage. Pick a kind:

- **[Model]({{ '/tags/model/' | relative_url }})** — a per-model slug (e.g. `gemma-4-31b`). Every run
  of one model shares it, so it groups all of a model's engines, quants, and speculative variants.
- **[Lab]({{ '/tags/lab/' | relative_url }})** — the lab that released the model (e.g. NVIDIA, OpenAI).
- **[Family]({{ '/tags/family/' | relative_url }})** — model family (e.g. Nemotron, Llama, Gemma).
- **[Quant / precision]({{ '/tags/quant/' | relative_url }})** — weight format (e.g. NVFP4, FP8, Q4_K_M).
- **[Size bucket]({{ '/tags/size/' | relative_url }})** — by total params (`≤4B`, `5-15B`, `16-40B`,
  `41-130B`, `130B+`).
- **[Concurrency]({{ '/tags/concurrency/' | relative_url }})** — parallel serves per run (e.g.
  `conc-32`); the benchmark's fixed-load axis.
- **[Spark recipe]({{ '/tags/spark-recipe/' | relative_url }})** — models with native DGX Spark support.
- **[REAP]({{ '/tags/reap/' | relative_url }})** — base checkpoint pruned with REAP (router-weighted
  expert-activation pruning).
- **[Self-REAP]({{ '/tags/self-reap/' | relative_url }})** — models we REAP-pruned ourselves; the page
  carries the how-to (none pruned yet).
- **[Self-Quantized]({{ '/tags/self-quantized/' | relative_url }})** — models we quantized ourselves to
  NVFP4; the page carries the streaming quantizer + full recipe.

<p class="tag-cloud">
  <a class="tag" href="{{ '/tags/model/' | relative_url }}">Model</a>
  <a class="tag" href="{{ '/tags/lab/' | relative_url }}">Lab</a>
  <a class="tag" href="{{ '/tags/family/' | relative_url }}">Family</a>
  <a class="tag" href="{{ '/tags/quant/' | relative_url }}">Quant</a>
  <a class="tag" href="{{ '/tags/size/' | relative_url }}">Size</a>
  <a class="tag" href="{{ '/tags/concurrency/' | relative_url }}">Concurrency</a>
  <a class="tag" href="{{ '/tags/spark-recipe/' | relative_url }}">Spark recipe</a>
  <a class="tag" href="{{ '/tags/reap/' | relative_url }}">REAP</a>
  <a class="tag" href="{{ '/tags/self-reap/' | relative_url }}">Self-REAP</a>
  <a class="tag" href="{{ '/tags/self-quantized/' | relative_url }}">Self-Quantized</a>
</p>
