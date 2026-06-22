---
layout: default
title: Tags · Spark recipe
permalink: /tags/spark-recipe/
---

# Spark recipe

Models with a **native DGX Spark recipe / support** (`Spark recipe` tag).
[← all tag kinds]({{ '/tags/' | relative_url }})

{% assign group = site.configs | where_exp: "c", "c.tags contains 'Spark recipe'" %}
<h2 id="spark-recipe">Spark recipe <span class="muted">({{ group | size }})</span></h2>
{% if group.size == 0 %}
*None tagged yet.*
{% else %}
{% include config-table.html configs=group %}
{% endif %}
