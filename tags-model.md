---
layout: default
title: Tags · Model
permalink: /tags/model/
---

# Browse by model

Every run of a model shares one **model** tag (the first tag on each config), so each section below
collects *all* of that model's configurations — across engines, quants, and speculative-decoding
variants — in one table. [← all tag kinds]({{ '/tags/' | relative_url }})

{%- comment -%} Distinct model slugs = first tag of each config (pure Liquid, no plugins). {%- endcomment -%}
{% assign rawm = "" %}
{% for c in site.configs %}{% assign m = c.tags | first %}{% assign rawm = rawm | append: "||" | append: m %}{% endfor %}
{% assign models = rawm | split: "||" | uniq | sort_natural %}

<p class="tag-cloud">
{% for v in models %}{% if v != "" %}<a class="tag" href="#{{ v | slugify }}">{{ v }}</a>{% endif %}{% endfor %}
</p>

{% for v in models %}{% if v != "" %}
{% assign group = site.configs | where_exp: "c", "c.tags contains v" %}
<h2 id="{{ v | slugify }}">{{ v }} <span class="muted">({{ group | size }})</span></h2>
{% include config-table.html configs=group %}
{% endif %}{% endfor %}
