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

{%- comment -%} Split models: multi-config get their own section; one-offs (1 config) fold into a trailing "Other". {%- endcomment -%}
{% assign others = "" | split: "" %}{% comment %} empty array {% endcomment %}
{% assign multi = "" %}
{% for v in models %}{% if v != "" %}
{% assign group = site.configs | where_exp: "c", "c.tags contains v" %}
{% if group.size == 1 %}{% assign others = others | concat: group %}
{% else %}{% assign multi = multi | append: "||" | append: v %}{% endif %}
{% endif %}{% endfor %}
{% assign multi = multi | split: "||" %}

<p class="tag-cloud">
{% for v in multi %}{% if v != "" %}<a class="tag" href="#{{ v | slugify }}">{{ v }}</a>{% endif %}{% endfor %}
{% if others.size > 0 %}<a class="tag" href="#other">Other</a>{% endif %}
</p>

{% for v in multi %}{% if v != "" %}
{% assign group = site.configs | where_exp: "c", "c.tags contains v" %}
<h2 id="{{ v | slugify }}">{{ v }} <span class="muted">({{ group | size }})</span></h2>
{% include config-table.html configs=group %}
{% endif %}{% endfor %}

{% if others.size > 0 %}
<h2 id="other">Other <span class="muted">({{ others.size }})</span></h2>
{% include config-table.html configs=others %}
{% endif %}
