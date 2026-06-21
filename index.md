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
  quantization. Pages are tagged by model family, company, engine, and quant, so you can
  [browse by tag]({{ '/tags/' | relative_url }}).
- Three inference engines are benchmarked, all as NVIDIA NGC containers:
  **llama.cpp**, **vLLM**, and **TensorRT-LLM**.
- Completed runs sort to the top by completion time; pending configurations follow.

## Configurations

{%- comment -%}
  Order: done first (newest completion on top) → pending → blocked (needs review, last).
  completed_at format "YYYY-MM-DD HH:MM +TZ" sorts chronologically as a string.
{%- endcomment -%}
{% assign done = site.configs | where_exp: "c", "c.completed_at" | sort: "completed_at" | reverse %}
{% assign blocked = site.configs | where_exp: "c", "c.status == 'blocked'" | sort: "title" %}
{% assign pending = site.configs | where_exp: "c", "c.completed_at == nil" | where_exp: "c", "c.status != 'blocked'" | sort: "title" %}
{% assign configs = done | concat: pending | concat: blocked %}
{% if configs.size == 0 %}
*No configurations recorded yet.*
{% else %}
<table class="listing">
  <thead>
    <tr><th>Configuration</th><th>Engine</th><th>Quant</th><th>Ctx</th><th>Decode tok/s</th><th>Status</th><th>Completed</th></tr>
  </thead>
  <tbody>
  {% for c in configs %}
    <tr>
      <td><a href="{{ c.url | relative_url }}">{{ c.title }}</a></td>
      <td>{{ c.engine | default: "—" }}</td>
      <td>{{ c.quant | default: "—" }}</td>
      <td>{{ c.context | default: "—" }}</td>
      <td>{{ c.decode_toks | default: "—" }}</td>
      <td><span class="status status-{{ c.status | default: 'pending' }}">{{ c.status | default: "pending" }}</span></td>
      <td>{{ c.completed_at | default: c.measured_on | default: "—" }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}
