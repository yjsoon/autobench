---
layout: default
title: Tags · Family
permalink: /tags/family/
---

# Browse by model family

Configurations grouped by **model family**. [← all tag kinds]({{ '/tags/' | relative_url }})

{% assign families = site.configs | map: "family" | compact | uniq | sort_natural %}

<p class="tag-cloud">
{% for v in families %}<a class="tag" href="#{{ v | slugify }}">{{ v }}</a>{% endfor %}
</p>

{% for v in families %}
{% assign group = site.configs | where: "family", v %}
<h2 id="{{ v | slugify }}">{{ v }} <span class="muted">({{ group | size }})</span></h2>
{% include config-table.html configs=group %}
{% endfor %}
