---
layout: default
title: Tags · Lab
permalink: /tags/lab/
---

# Browse by lab

Configurations grouped by the **lab** that released the model. [← all tag kinds]({{ '/tags/' | relative_url }})

{% assign labs = site.configs | map: "company" | compact | uniq | sort_natural %}

<p class="tag-cloud">
{% for v in labs %}<a class="tag" href="#{{ v | slugify }}">{{ v }}</a>{% endfor %}
</p>

{% for v in labs %}
{% assign group = site.configs | where: "company", v %}
<h2 id="{{ v | slugify }}">{{ v }} <span class="muted">({{ group | size }})</span></h2>
{% include config-table.html configs=group %}
{% endfor %}
