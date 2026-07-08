---
layout: default
title: About
---

# Strix Halo Autobench

Benchmarks of open-weight LLMs running on an **AMD Strix Halo** mini-PC — a GMKtec EVO X2
(Ryzen AI Max+ 395, Radeon 8060S iGPU, 96 GiB unified VRAM, x86_64, Vulkan/ROCm) — forked
from [gauravmm/autobench](https://github.com/gauravmm/autobench), whose **NVIDIA DGX Spark**
(GB10 Grace-Blackwell, 121 GB unified memory, CUDA 13) results are kept here as the
comparison baseline.

The goal: take a list of models and, **one at a time**, download each, run it across
several engine / quantization / context configurations, and record the attributes that
actually matter for local inference — above all the **full run command, context window,
and tok/s (prefill + decode)**, plus peak memory.

## How it's organized

- **The listing shows this machine's runs by default.** Switch to *Compare vs DGX Spark*
  above to interleave the upstream Spark numbers — rows are then colour-coded
  <span class="machine-badge mb-strix">Strix Halo</span> vs
  <span class="machine-badge mb-spark">DGX Spark</span>. Where the same model/quant/workload
  exists on both machines, the config pages cross-link the comparison.
- **Each page in the listing below is one configuration** — a specific model × engine ×
  quantization. Pages are tagged by model, lab, model family, quant, size bucket, machine, and a
  `Spark recipe` flag — each with its own page, so you can [browse by tag]({{ '/tags/' | relative_url }}).
- Engines run as official containers: **llama.cpp** (Vulkan) and **SGLang** / **vLLM** (ROCm)
  on the Strix Halo box; the Spark baseline used NVIDIA NGC builds of the same engines plus
  **TensorRT-LLM**/**NIM**. A model often ends up with **several configs that compound across
  engines, quants, and speculative-decoding variants**.
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
