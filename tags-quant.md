---
layout: default
title: Tags · Quant
permalink: /tags/quant/
---

# Browse by quant / precision

Configurations grouped by **quant / precision** format. [← all tag kinds]({{ '/tags/' | relative_url }})

{% assign quants = site.configs | map: "quant" | compact | uniq | sort_natural %}

<p class="tag-cloud">
{% for v in quants %}<a class="tag" href="#{{ v | slugify }}">{{ v }}</a>{% endfor %}
</p>

{% for v in quants %}
{% assign group = site.configs | where: "quant", v %}
<h2 id="{{ v | slugify }}">{{ v }} <span class="muted">({{ group | size }})</span></h2>
{% include config-table.html configs=group %}
{% endfor %}
