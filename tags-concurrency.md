---
layout: default
title: Tags · Concurrency
permalink: /tags/concurrency/
---

# Browse by concurrency

Configurations grouped by the **number of parallel serves** (`--parallel` / `--max-num-seqs` /
client concurrency) used for the run. The whole benchmark is run at a fixed concurrency for
cross-config comparability — this page makes that explicit and groups any future low-concurrency
(e.g. `conc-1`) variants separately. [← all tag kinds]({{ '/tags/' | relative_url }})

{% assign concs = site.configs | map: "concurrency" | compact | uniq | sort %}

<p class="tag-cloud">
{% for v in concs %}<a class="tag" href="#conc-{{ v }}">conc-{{ v }}</a>{% endfor %}
</p>

{% for v in concs %}
{% assign group = site.configs | where: "concurrency", v %}
<h2 id="conc-{{ v }}">conc-{{ v }} <span class="muted">({{ group | size }})</span></h2>
{% include config-table.html configs=group %}
{% endfor %}
