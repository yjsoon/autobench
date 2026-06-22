---
layout: default
title: Tags
permalink: /tags/
---

# Browse by tag

Every configuration is tagged in six kinds. Each kind has its own page, where every tag value lists
its configurations in a table like the homepage. Pick a kind:

- **[Model]({{ '/tags/model/' | relative_url }})** — a per-model slug (e.g. `gemma-4-31b`). Every run
  of one model shares it, so it groups all of a model's engines, quants, and speculative variants.
- **[Lab]({{ '/tags/lab/' | relative_url }})** — the lab that released the model (e.g. NVIDIA, OpenAI).
- **[Family]({{ '/tags/family/' | relative_url }})** — model family (e.g. Nemotron, Llama, Gemma).
- **[Quant / precision]({{ '/tags/quant/' | relative_url }})** — weight format (e.g. NVFP4, FP8, Q4_K_M).
- **[Size bucket]({{ '/tags/size/' | relative_url }})** — by total params (`≤4B`, `5-15B`, `16-40B`,
  `41-130B`, `130B+`).
- **[Spark recipe]({{ '/tags/spark-recipe/' | relative_url }})** — models with native DGX Spark support.

<p class="tag-cloud">
  <a class="tag" href="{{ '/tags/model/' | relative_url }}">Model</a>
  <a class="tag" href="{{ '/tags/lab/' | relative_url }}">Lab</a>
  <a class="tag" href="{{ '/tags/family/' | relative_url }}">Family</a>
  <a class="tag" href="{{ '/tags/quant/' | relative_url }}">Quant</a>
  <a class="tag" href="{{ '/tags/size/' | relative_url }}">Size</a>
  <a class="tag" href="{{ '/tags/spark-recipe/' | relative_url }}">Spark recipe</a>
</p>
