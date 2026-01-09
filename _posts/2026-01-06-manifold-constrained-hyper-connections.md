---
layout: post
title: "From Residual Connections to Manifold-Constrained Hyper-Connections"
date: 2026-01-06
categories: [deep-learning, transformers, research]
tags: [ResNet, Hyper-Connections, mHC, LLM, Sinkhorn-Knopp]
---

<link rel="stylesheet" href="/assets/css/custom.css">

<!-- MathJax for LaTeX rendering -->
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  },
  svg: {
    fontCache: 'global'
  }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<!-- Mermaid for diagrams -->
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
mermaid.initialize({
  startOnLoad: true,
  theme: 'base',
  themeVariables: {
    primaryColor: '#e0f2fe',
    primaryTextColor: '#1e293b',
    primaryBorderColor: '#0891b2',
    lineColor: '#64748b',
    secondaryColor: '#f8fafc',
    tertiaryColor: '#ffffff',
    background: '#ffffff',
    mainBkg: '#f1f5f9',
    nodeBorder: '#0891b2',
    clusterBkg: '#f8fafc',
    fontSize: '14px'
  },
  flowchart: {
    curve: 'basis',
    padding: 20,
    nodeSpacing: 50,
    rankSpacing: 50
  }
});
</script>

## A Simple, Intuitive, and Visual Guide

Deep neural networks owe much of their success to a deceptively simple idea: the <strong>residual connection</strong>. Since the introduction of ResNets in 2016 [1], residual connections have become a foundational design element across modern architectures—most notably Transformers [2] and large language models (LLMs).

Recently, <strong>Hyper-Connections (HC)</strong> [3] proposed a bold extension to residuals by widening the residual stream and increasing connection complexity—unlocking new representational power. However, this power comes with a hidden cost: <strong>training instability at scale</strong>.

<strong>Manifold-Constrained Hyper-Connections (mHC)</strong> [4] solve this problem elegantly using geometric constraints from optimal transport theory.

In this post, we'll build intuition step-by-step for:
- Why residual connections work
- What goes wrong in Hyper-Connections  
- How mHC fix the problem
- Why the Sinkhorn–Knopp projection is the key ingredient

---

## 1. Why Residual Connections Changed Everything

A standard neural network layer replaces its input:

<div class="math-block">

$$x_{l+1} = F(x_l)$$

</div>

Residual connections modify this to:

<div class="math-block">

$$x_{l+1} = x_l + F(x_l)$$

</div>

<pre class="mermaid">
graph LR
    A["x_l (Input)"] --> F["F(·) Transform"]
    A --> ADD(("+"))
    F --> ADD
    ADD --> B["x_l+1 (Output)"]
</pre>

### Why does this help?

You can think of a residual layer as keeping what you already know and applying a small correction.

This simple change enables:
- Stable training at great depth
- Protection against vanishing gradients
- <strong>Identity fallback</strong>: if F(x) = 0, the network behaves like the identity function

Stacking many residual layers yields:

<div class="math-block">

$$x_L = x_0 + \sum_{i=0}^{L-1} F(x_i)$$

</div>

The original signal flows unchanged through all deeper layers. This property—<strong>identity mapping</strong>—is the secret behind ResNet's stability.

---

## 2. Hyper-Connections: Making Residuals Wider

Hyper-Connections (HC) [3] generalize residual connections by introducing <strong>multiple parallel residual streams</strong>.

| Standard ResNet | Hyper-Connections |
|----------------|-------------------|
| Single stream | Multiple streams (n) |
| Fixed identity path | Learnable mixing matrices |
| Limited expressiveness | Rich multi-path interactions |

<pre class="mermaid">
graph TB
    subgraph Input["Input Streams"]
        s1["Stream 1"]
        s2["Stream 2"]
        s3["Stream 3"]
    end
    
    H1["H_pre"]
    F["F(·) Core"]
    H2["H_post"]
    HR["H_res"]
    MX["⊕ Mix"]
    
    Input --> H1 --> F --> H2 --> MX
    Input --> HR --> MX
    
    subgraph Output["Output Streams"]
        o1["Stream 1'"]
        o2["Stream 2'"]
        o3["Stream 3'"]
    end
    
    MX --> Output
</pre>

Each layer mixes these streams using learnable matrices:

<div class="math-block">

$$x_{l+1} = H^{res}_l \cdot x_l + (H^{post}_l)^\top \cdot F(H^{pre}_l \cdot x_l)$$

</div>

The idea is powerful: multiple pathways, richer interaction, no extra FLOPs in the core function F.

<strong>But something subtle breaks.</strong>

---

## 3. The Hidden Instability in Hyper-Connections

In ResNet, the identity path has weight exactly = 1.

In HC, the identity path becomes a <strong>matrix product</strong>:

<div class="math-block">

$$x_L = \left(\prod_{i=0}^{L-1} H^{res}_i\right) \cdot x_0 + \cdots$$

</div>

<pre class="mermaid">
graph LR
    x0["x_0"] --> H1["H_res_1"]
    H1 --> H2["H_res_2"]
    H2 --> H3["H_res_3"]
    H3 --> dots["···"]
    dots --> HL["H_res_L"]
    HL --> xL["x_L"]
</pre>

<div class="highlight-box">

<strong>The Problem:</strong> The H_res matrices are unconstrained.

<p>Across many layers, matrix products can:</p>
<ul>
<li><strong>Amplify signals</strong> exponentially</li>
<li><strong>Attenuate signals</strong> to zero</li>
<li><strong>Destroy mean preservation</strong></li>
</ul>

<p>There is no guarantee that the product of all H_res matrices behaves like identity!</p>

</div>

---

## 4. A 1D Toy Simulation

Consider a 1D signal x, with a small damping function:

<div class="math-block">

$$F(x) = -0.05x$$

</div>

| Architecture | Update Rule | Behavior |
|:------------|:-----------|:---------|
| ResNet | x = 1.0 × x + F(x) | ✅ Stable decay |
| HC | x = 1.08 × x + F(x) | ⚠️ Slow explosion |
| mHC | x = 1.0 × x + F(x) | ✅ Stable decay |

<strong>After 50 layers:</strong>
- ResNet & mHC: signal ≈ 0.08 (controlled decay)
- HC: signal ≈ 6.0+ (explosion!)

Gradients behave similarly:
- HC gradients grow like (1.03)^L — exponential explosion
- ResNet/mHC gradients remain controlled

This becomes catastrophic at LLM scale with L > 100 layers.

---

## 5. The mHC Solution: Doubly Stochastic Matrices

mHC asks: How do we allow rich mixing while preserving identity behavior?

<div class="highlight-box">

<strong>Answer: Constrain H_res to be a doubly stochastic matrix</strong>

<p>A matrix H is doubly stochastic if:</p>
<ul>
<li>All entries ≥ 0</li>
<li>Each row sums to 1</li>
<li>Each column sums to 1</li>
</ul>

</div>

<pre class="mermaid">
graph TB
    subgraph Properties["Doubly Stochastic Properties"]
        P1["Σ_j H_ij = 1 (rows)"]
        P2["Σ_i H_ij = 1 (columns)"]
        P3["H_ij ≥ 0 (non-negative)"]
    end
    
    subgraph Benefits["Key Benefits"]
        B1["Mean preserved"]
        B2["Norms bounded"]
        B3["Products stable"]
    end
    
    Properties --> Benefits
</pre>

### Why This Works

If H is doubly stochastic, then the mean is preserved:

<div class="math-block">

$$\text{mean}(Hx) = \text{mean}(x)$$

</div>

<strong>Critical property:</strong> The product of doubly stochastic matrices is also doubly stochastic!

This guarantees <strong>identity preservation across arbitrarily deep networks</strong>.

---

## 6. Sinkhorn–Knopp Algorithm

How do we enforce the doubly stochastic constraint during training?

The <strong>Sinkhorn–Knopp algorithm</strong> [5] projects any non-negative matrix onto the set of doubly stochastic matrices through alternating row and column normalization.

<pre class="mermaid">
graph TB
    A["Unconstrained Matrix A"] --> S1["Step 1: Normalize rows"]
    S1 --> S2["Step 2: Normalize columns"]
    S2 --> C{"Converged?"}
    C -->|No| S1
    C -->|Yes| B["Doubly Stochastic B ✓"]
</pre>

### Example

<strong>Before Sinkhorn projection:</strong>

```
A = | 1.37  1.79  1.51 |    Row sums: 4.67, 4.04, 5.73
    | 1.36  1.06  1.62 |    Col sums: 3.83, 5.08, 5.53
    | 1.09  2.23  2.41 |
```

<strong>After Sinkhorn projection:</strong>

```
B = | 0.355  0.366  0.279 |    Row sums: 1.0, 1.0, 1.0 ✓
    | 0.406  0.250  0.344 |    Col sums: 1.0, 1.0, 1.0 ✓
    | 0.239  0.385  0.376 |
```

---

## 7. Identity Preservation in Action

<strong>Input vector:</strong> x = [1, 2, 3], mean = 2

<pre class="mermaid">
graph LR
    subgraph HC["Unconstrained HC"]
        I1["[1, 2, 3] mean=2"] --> A1["A × x"] --> O1["[9.5, 8.3, 12.8] mean=10.2 ⚠️"]
    end
</pre>

<pre class="mermaid">
graph LR
    subgraph MHC["mHC with Sinkhorn"]
        I2["[1, 2, 3] mean=2"] --> B1["B × x"] --> O2["[1.9, 1.9, 2.1] mean=2.0 ✓"]
    end
</pre>

| Method | Output | Mean | Status |
|--------|--------|------|--------|
| Unconstrained HC | [9.5, 8.3, 12.8] | 10.2 | ⚠️ 5× explosion |
| mHC (Sinkhorn) | [1.9, 1.9, 2.1] | 2.0 | ✓ Preserved |

---

## 8. Physical Intuition: Conservation Law

Think of each stream as a <strong>water pipe</strong> and matrix weights as <strong>valves</strong>.

<pre class="mermaid">
graph LR
    P1["Pipe 1"] --> V["Valve Matrix H"]
    P2["Pipe 2"] --> V
    P3["Pipe 3"] --> V
    V --> O1["Out 1"]
    V --> O2["Out 2"]
    V --> O3["Out 3"]
</pre>

The Sinkhorn constraint ensures:
- No water (information) is created
- No water (information) is destroyed
- Only redistributed across streams

mHC introduces a <strong>conservation law for information flow</strong> in neural networks.

---

## 9. Scaling in Practice

mHC also includes practical optimizations [4]:

| Optimization | Benefit |
|-------------|---------|
| Kernel fusion | Reduced memory transfers |
| Mixed-precision (BF16) | 2× memory efficiency |
| Selective recomputation | Lower memory footprint |
| Communication overlap | Better multi-GPU scaling |

In practice:
- Expansion rate n = 4
- Only approximately 6.7% training slowdown
- Massive stability gains at scale

---

## 10. Summary

<pre class="mermaid">
graph LR
    R["2016: ResNet"] --> HC["2024: Hyper-Connections"] --> MHC["2025: mHC"]
</pre>

| Architecture | Expressiveness | Stability |
|-------------|---------------|-----------|
| ResNet | Limited | ✓ Stable |
| Hyper-Connections | High | ⚠️ Unstable |
| mHC | High | ✓ Stable |

<div class="highlight-box">

<strong>Key Takeaways:</strong>

<ol>
<li>Residual connections enable deep training via identity preservation</li>
<li>Hyper-Connections add expressiveness but break stability guarantees</li>
<li>mHC restores stability by constraining mixing to doubly stochastic matrices</li>
<li>Sinkhorn–Knopp projection efficiently enforces this constraint during training</li>
</ol>

</div>

> Sinkhorn–Knopp turns arbitrary mixing into structured averaging, preserving identity mappings across extreme depth.

This is why mHC scales where vanilla HC does not—and why it is a promising building block for the next generation of large-scale models.

---

## References

[1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). <strong>Deep Residual Learning for Image Recognition</strong>. CVPR 2016. <a href="https://arxiv.org/abs/1512.03385">arXiv:1512.03385</a>

[2] Vaswani, A., et al. (2017). <strong>Attention Is All You Need</strong>. NeurIPS 2017. <a href="https://arxiv.org/abs/1706.03762">arXiv:1706.03762</a>

[3] Zhu, Y., et al. (2024). <strong>Hyper-Connections</strong>. arXiv preprint. <a href="https://arxiv.org/abs/2409.19606">arXiv:2409.19606</a>

[4] Yang, D., et al. (2024). <strong>Manifold-Constrained Hyper-Connections</strong>. arXiv preprint. <a href="https://arxiv.org/abs/2512.24880">arXiv:2512.24880</a>

[5] Sinkhorn, R., & Knopp, P. (1967). <strong>Concerning nonnegative matrices and doubly stochastic matrices</strong>. Pacific Journal of Mathematics, 21(2), 343-348.

[6] Birkhoff, G. (1946). <strong>Three observations on linear algebra</strong>. Univ. Nac. Tucumán Rev. Ser. A, 5, 147-151.

---

<em>If you found this post useful, connect with me on <a href="https://www.linkedin.com/in/sandeep-pandey-43790921/">LinkedIn</a> or check out my publications on <a href="https://scholar.google.com/citations?user=NveAdp8AAAAJ&hl=en">Google Scholar</a>.</em>
