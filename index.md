---
layout: default
title: About
---

# DGX Spark Autobench

Benchmarks of open-weight LLMs running on a single **NVIDIA DGX Spark** (GB10
Grace-Blackwell, 121 GB unified memory, ARM64, CUDA 13).

The goal: take a list of models and, **one at a time**, download each, run it across
several engine / quantization / context configurations, and record the attributes that
actually matter for local inference — above all the **full run command, context window,
and tok/s (prefill + decode)**, plus peak memory.

## How it's organized

- **Each page in the listing below is one configuration** — a specific model × engine ×
  quantization. Pages are tagged in six kinds — model, lab, model family, quant, size bucket, and a
  `Spark recipe` flag — each with its own page, so you can [browse by tag]({{ '/tags/' | relative_url }}).
- Engines are run as NVIDIA NGC / official containers: **llama.cpp**, **vLLM**, **SGLang**,
  **TensorRT-LLM** (and **NIM** where available). The engine shown on a *pending* row is a
  starting guess — the actual server is chosen per model at benchmark time (whichever is fastest
  for that model on the Spark), so a model often ends up with **several configs that compound
  across engines, quants, and speculative-decoding variants**.
- Completed runs sort to the top by completion time; pending configurations follow, with
  blocked (needs-review) ones last.

## Configurations

{%- comment -%}
  Order: done first (newest completion on top) → pending → blocked (needs review, last).
  completed_at format "YYYY-MM-DD HH:MM +TZ" sorts chronologically as a string.
{%- endcomment -%}
{% if site.configs.size == 0 %}
*No configurations recorded yet.*
{% else %}
{% include config-table.html configs=site.configs %}
{% endif %}
