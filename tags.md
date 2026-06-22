---
layout: default
title: Tags
permalink: /tags/
---

# Browse by tag

Every configuration is tagged in six categories: **model** (e.g. `gemma-4-31b`) — every run of one
model shares this tag, so it groups all of a model's engines, quants, and speculative variants
together; **lab** (e.g. NVIDIA, OpenAI), **model family** (e.g. Nemotron, Llama),
**quant / precision** (e.g. NVFP4, Q4_K_M), **size bucket** by total params (`≤4B`, `5-15B`, `16-40B`,
`41-130B`, `130B+`), and **`Spark recipe`** — models with native DGX Spark recipe/support.

{%- comment -%} Collect every tag across all configs (pure Liquid, no plugins). {%- endcomment -%}
{% assign rawtags = "" %}
{% for c in site.configs %}
  {% assign ctags = c.tags | join: "||" %}
  {% assign rawtags = rawtags | append: "||" | append: ctags %}
{% endfor %}
{% assign alltags = rawtags | split: "||" | uniq | sort_natural %}

{% if alltags.size == 0 %}
*No tags yet.*
{% else %}
<p class="tag-cloud">
{% for tag in alltags %}{% if tag != "" %}<a class="tag" href="#{{ tag | slugify }}">{{ tag }}</a>{% endif %}{% endfor %}
</p>

{% for tag in alltags %}{% if tag != "" %}
<h2 id="{{ tag | slugify }}">{{ tag }}</h2>
<ul>
{% for c in site.configs %}{% if c.tags contains tag %}
  <li><a href="{{ c.url | relative_url }}">{{ c.title }}</a></li>
{% endif %}{% endfor %}
</ul>
{% endif %}{% endfor %}
{% endif %}
