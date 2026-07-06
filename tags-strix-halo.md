---
layout: default
title: Tags · Strix Halo
permalink: /tags/strix-halo/
---

# Strix Halo runs

Configurations measured on **this machine** — a GMKtec EVO X2 (AMD Ryzen AI Max+ 395 "Strix Halo",
Radeon 8060S iGPU, 96 GiB UMA VRAM) — the `strix-halo` tag. Everything *without* this tag is the
upstream **NVIDIA DGX Spark** baseline. [← all tag kinds]({{ '/tags/' | relative_url }})

{% assign group = site.configs | where_exp: "c", "c.tags contains 'strix-halo'" %}
<h2 id="strix-halo">Strix Halo <span class="muted">({{ group | size }})</span></h2>
{% if group.size == 0 %}
*None tagged yet.*
{% else %}
{% include config-table.html configs=group %}
{% endif %}
