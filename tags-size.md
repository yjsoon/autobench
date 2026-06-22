---
layout: default
title: Tags · Size
permalink: /tags/size/
---

# Browse by size bucket

Configurations grouped by **size bucket** (total parameters). [← all tag kinds]({{ '/tags/' | relative_url }})

{%- comment -%} Fixed bucket order, smallest → largest (these are tag values, not a frontmatter field). {%- endcomment -%}
{% assign sizes = "≤4B,5-15B,16-40B,41-130B,130B+" | split: "," %}

<p class="tag-cloud">
{% for v in sizes %}<a class="tag" href="#{{ v | slugify }}">{{ v }}</a>{% endfor %}
</p>

{% for v in sizes %}
{% assign group = site.configs | where_exp: "c", "c.tags contains v" %}
<h2 id="{{ v | slugify }}">{{ v }} <span class="muted">({{ group | size }})</span></h2>
{% if group.size == 0 %}
*None yet.*
{% else %}
{% include config-table.html configs=group %}
{% endif %}
{% endfor %}
