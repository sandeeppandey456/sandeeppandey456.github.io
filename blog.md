---
layout: page
title: AI Blog
permalink: /blog/
---

<link rel="stylesheet" href="/assets/css/custom.css">

## Technical Articles

<p style="color: var(--text-secondary);">Here I share insights on LLM alignment, tensor architectures, speech processing, and AI research.</p>

<div class="section">

### Topics I Write About

<div>
<span class="badge">LLM Alignment</span>
<span class="badge">RLHF</span>
<span class="badge">Speech Emotion Recognition</span>
<span class="badge">Tensor Neural Networks</span>
<span class="badge">RAG Systems</span>
<span class="badge">Conversational AI</span>
<span class="badge">Multi-Agent Frameworks</span>
<span class="badge">Indic NLP</span>
</div>

</div>

---

## Latest Posts

{% for post in site.posts %}
<div class="card">
<div class="card-header">
<h3 class="card-title"><a href="{{ post.url }}">{{ post.title }}</a></h3>
<span class="card-date">{{ post.date | date: "%b %d, %Y" }}</span>
</div>
<div class="card-body">
{{ post.excerpt }}
</div>
</div>
{% endfor %}

{% if site.posts.size == 0 %}
<div class="card">
<div class="card-body">
<p>Coming soon! Stay tuned for technical deep-dives on AI research and implementation.</p>
</div>
</div>
{% endif %}
