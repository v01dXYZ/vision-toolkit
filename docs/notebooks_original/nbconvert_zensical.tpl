{% extends "markdown/index" %}

{% block input %}
```
{%- if "magics_language" in cell.metadata -%}
{{ cell.metadata.magics_language }}
{%- elif "name" in nb.metadata.get("language_info", {}) -%}
{{ nb.metadata.language_info.name }}
{%- endif %}
{{ cell.source if not cell.source.startswith("%%") else cell.source[cell.source.find("\n")+1:]}}
```
{% endblock input %}

{% block output_group %}
<div class="result" markdown>
{{ super() }}
</div>
{% endblock output_group %}

{% block data_text %}
``` {{ nb.metadata.language_info.name }}
{{ output.data['text/plain'] }}
```
{% endblock data_text %}

{% block stream %}
{% endblock stream %}