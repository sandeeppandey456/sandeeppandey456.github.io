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
<div class="card-footer">
<div class="card-engagement">
<span class="engagement-indicator" title="Like this post">
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
<path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
</svg>
</span>
<span class="engagement-indicator" title="Join the discussion">
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
</svg>
</span>
</div>
<a href="{{ post.url }}" class="read-more">Read More →</a>
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
