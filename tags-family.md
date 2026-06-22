---
layout: default
title: Tags · Family
permalink: /tags/family/
---

# Browse by model family

Configurations grouped by **model family**. [← all tag kinds]({{ '/tags/' | relative_url }})

{% assign families = site.configs | map: "family" | compact | uniq | sort_natural %}

{%- comment -%} Split families: those with >3 configs get their own section; small ones (≤3) fold into a trailing "Other". {%- endcomment -%}
{% assign others = "" | split: "" %}{% comment %} empty array {% endcomment %}
{% assign multi = "" %}
{% for v in families %}
{% assign group = site.configs | where: "family", v %}
{% if group.size <= 3 %}{% assign others = others | concat: group %}
{% else %}{% assign multi = multi | append: "||" | append: v %}{% endif %}
{% endfor %}
{% assign multi = multi | split: "||" %}

<p class="tag-cloud">
{% for v in multi %}{% if v != "" %}<a class="tag" href="#{{ v | slugify }}">{{ v }}</a>{% endif %}{% endfor %}
{% if others.size > 0 %}<a class="tag" href="#other">Other</a>{% endif %}
</p>

{% for v in multi %}{% if v != "" %}
{% assign group = site.configs | where: "family", v %}
<h2 id="{{ v | slugify }}">{{ v }} <span class="muted">({{ group | size }})</span></h2>
{% include config-table.html configs=group %}
{% endif %}{% endfor %}

{% if others.size > 0 %}
<h2 id="other">Other <span class="muted">({{ others.size }})</span></h2>
{% include config-table.html configs=others %}
{% endif %}
