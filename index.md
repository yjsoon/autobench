---
layout: default
title: Configurations
---

# Benchmark configurations

One page per model × engine × quant configuration run on the DGX Spark.
Each records the **full run command, context window, and tok/s**.

{% assign configs = site.configs | sort: "title" %}
{% if configs.size == 0 %}
*No configurations recorded yet.*
{% else %}
<table class="listing">
  <thead>
    <tr><th>Configuration</th><th>Engine</th><th>Quant</th><th>Ctx</th><th>Decode tok/s</th><th>Status</th></tr>
  </thead>
  <tbody>
  {% for c in configs %}
    <tr>
      <td><a href="{{ c.url | relative_url }}">{{ c.title }}</a></td>
      <td>{{ c.engine | default: "—" }}</td>
      <td>{{ c.quant | default: "—" }}</td>
      <td>{{ c.context | default: "—" }}</td>
      <td>{{ c.decode_toks | default: "—" }}</td>
      <td><span class="status status-{{ c.status | default: 'pending' }}">{{ c.status | default: "pending" }}</span></td>
    </tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}
