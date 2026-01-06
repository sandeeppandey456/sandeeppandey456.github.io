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
  theme: 'dark',
  themeVariables: {
    primaryColor: '#06b6d4',
    primaryTextColor: '#f8fafc',
    primaryBorderColor: '#8b5cf6',
    lineColor: '#94a3b8',
    secondaryColor: '#1e293b',
    tertiaryColor: '#0f172a',
    background: '#0a0f1a',
    mainBkg: '#1e293b',
    nodeBorder: '#8b5cf6',
    clusterBkg: '#1e293b',
    fontSize: '16px'
  },
  flowchart: {
    curve: 'basis',
    padding: 20,
    nodeSpacing: 50,
    rankSpacing: 60
  }
});
</script>

<style>
.mermaid {
  background: linear-gradient(135deg, rgba(6, 182, 212, 0.05), rgba(139, 92, 246, 0.05));
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 12px;
  padding: 24px;
  margin: 24px 0;
  overflow-x: auto;
}
.mermaid svg {
  max-width: 100%;
  height: auto;
}
.math-block {
  background: rgba(17, 24, 39, 0.8);
  border-left: 3px solid #06b6d4;
  padding: 16px 20px;
  margin: 20px 0;
  border-radius: 0 8px 8px 0;
  overflow-x: auto;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 20px 0;
  background: rgba(17, 24, 39, 0.6);
  border-radius: 8px;
  overflow: hidden;
}
th, td {
  padding: 12px 16px;
  text-align: left;
  border-bottom: 1px solid rgba(148, 163, 184, 0.1);
}
th {
  background: rgba(6, 182, 212, 0.2);
  color: #06b6d4;
  font-weight: 600;
}
blockquote {
  border-left: 4px solid #8b5cf6;
  background: rgba(139, 92, 246, 0.1);
  padding: 16px 20px;
  margin: 20px 0;
  border-radius: 0 8px 8px 0;
  font-style: italic;
}
.highlight-box {
  background: linear-gradient(135deg, rgba(6, 182, 212, 0.1), rgba(139, 92, 246, 0.1));
  border: 1px solid rgba(6, 182, 212, 0.3);
  border-radius: 12px;
  padding: 20px;
  margin: 20px 0;
}
</style>

## A Simple, Intuitive, and Visual Guide

Deep neural networks owe much of their success to a deceptively simple idea: **the residual connection**. Since the introduction of ResNets in 2016 [1], residual connections have become a foundational design element across modern architectures—most notably Transformers [2] and large language models (LLMs).

Recently, **Hyper-Connections (HC)** [3] proposed a bold extension to residuals by widening the residual stream and increasing connection complexity—unlocking new representational power. However, this power comes with a hidden cost: **training instability at scale**.

**Manifold-Constrained Hyper-Connections (mHC)** [4] solve this problem elegantly using geometric constraints from optimal transport theory.

In this post, we'll build intuition step-by-step for:
- Why residual connections work
- What goes wrong in Hyper-Connections  
- How mHC fix the problem
- Why **Sinkhorn–Knopp projection** is the key ingredient

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

<div class="mermaid">
graph LR
    subgraph Standard["🔴 Standard Layer"]
        A1["x<sub>l</sub>"] --> F1["F(·)"] --> B1["x<sub>l+1</sub>"]
    end
</div>

<div class="mermaid">
graph LR
    subgraph Residual["🟢 Residual Connection"]
        A2["x<sub>l</sub>"] --> F2["F(·)"]
        A2 --> add["+"]
        F2 --> add
        add --> B2["x<sub>l+1</sub>"]
    end
</div>

### Why does this help?

You can think of a residual layer as:

> "Keep what you already know, and apply a small correction."

This simple change enables:
- ✅ Stable training at great depth
- ✅ Protection against vanishing gradients
- ✅ **Identity fallback**: if $F(x) = 0$, the network behaves like the identity function

Stacking many residual layers yields:

<div class="math-block">

$$x_L = x_0 + \sum_{i=0}^{L-1} F(x_i)$$

</div>

The original signal $x_0$ flows **unchanged** through all deeper layers. This property—**identity mapping**—is the secret behind ResNet's stability.

---

## 2. Hyper-Connections: Making Residuals Wider

Hyper-Connections (HC) [3] generalize residual connections by introducing **multiple parallel residual streams**.

| Standard ResNet | Hyper-Connections |
|----------------|-------------------|
| Single stream: $x \in \mathbb{R}^C$ | Multiple streams: $x \in \mathbb{R}^{n \times C}$ |
| Fixed identity path | Learnable mixing matrices |
| Limited expressiveness | Rich multi-path interactions |

<div class="mermaid">
graph TB
    subgraph HC["Hyper-Connection Layer"]
        direction TB
        subgraph IN["📥 Input (n streams)"]
            s1["Stream 1"]
            s2["Stream 2"]
            s3["Stream 3"]
        end
        
        H1["H<sup>pre</sup>"]
        F["F(·)"]
        H2["H<sup>post</sup>"]
        HR["H<sup>res</sup>"]
        MX[("⊕ Mix")]
        
        IN --> H1 --> F --> H2 --> MX
        IN --> HR --> MX
        
        subgraph OUT["📤 Output (n streams)"]
            o1["Stream 1'"]
            o2["Stream 2'"]
            o3["Stream 3'"]
        end
        
        MX --> OUT
    end
</div>

Each layer **mixes** these streams using learnable matrices:

<div class="math-block">

$$x_{l+1} = H^{\text{res}}_l \cdot x_l + (H^{\text{post}}_l)^\top \cdot F(H^{\text{pre}}_l \cdot x_l)$$

</div>

The idea is powerful: multiple pathways, richer interaction, no extra FLOPs in the core function $F$.

**But something subtle breaks.**

---

## 3. The Hidden Instability in Hyper-Connections

In ResNet, the identity path has weight exactly = 1.

In HC, the identity path becomes a **matrix product**:

<div class="math-block">

$$x_L = \left(\prod_{i=0}^{L-1} H^{\text{res}}_i\right) \cdot x_0 + \cdots$$

</div>

<div class="mermaid">
graph LR
    x0["x<sub>0</sub>"] --> H1["H<sup>res</sup><sub>1</sub>"]
    H1 --> H2["H<sup>res</sup><sub>2</sub>"]
    H2 --> H3["H<sup>res</sup><sub>3</sub>"]
    H3 --> dots["···"]
    dots --> HL["H<sup>res</sup><sub>L</sub>"]
    HL --> xL["x<sub>L</sub>"]
    
    style x0 fill:#06b6d4,stroke:#06b6d4,color:#000
    style xL fill:#dc2626,stroke:#dc2626,color:#fff
</div>

<div class="highlight-box">

**🚨 The Problem:** $H^{\text{res}}$ matrices are **unconstrained**

Across many layers, matrix products can:
- 📈 **Amplify signals** exponentially
- 📉 **Attenuate signals** to zero
- 💥 **Destroy mean preservation**

There is **no guarantee** that $\prod_i H^{\text{res}}_i$ behaves like identity!

</div>

---

## 4. A 1D Toy Simulation

Consider a 1D signal $x$, with a small damping:

<div class="math-block">

$$F(x) = -0.05x$$

</div>

| Architecture | Update Rule | Behavior |
|:------------|:-----------|:---------|
| **ResNet** | $x_{l+1} = 1.0 \cdot x_l + F(x_l)$ | ✅ Stable decay |
| **HC** | $x_{l+1} = 1.08 \cdot x_l + F(x_l)$ | 🚨 Slow explosion |
| **mHC** | $x_{l+1} = 1.0 \cdot x_l + F(x_l)$ | ✅ Stable decay |

<div class="mermaid">
graph LR
    subgraph Comparison["Signal After 50 Layers"]
        R["🟢 ResNet<br/>x ≈ 0.08"]
        M["🟢 mHC<br/>x ≈ 0.08"]
        H["🔴 HC<br/>x ≈ 6.0+ 💥"]
    end
</div>

**Gradients behave similarly:**
- HC gradients grow like $(1.03)^L$ — **exponential explosion**
- ResNet/mHC gradients remain **controlled**

This becomes catastrophic at LLM scale with $L > 100$ layers.

---

## 5. The mHC Solution: Doubly Stochastic Matrices

mHC asks: *"How do we allow rich mixing while preserving identity behavior?"*

<div class="highlight-box">

**Answer: Constrain $H^{\text{res}}$ to be a doubly stochastic matrix**

A matrix $H$ is **doubly stochastic** if:
- ✅ All entries $\geq 0$
- ✅ Each row sums to 1
- ✅ Each column sums to 1

</div>

<div class="mermaid">
graph TB
    subgraph DS["Doubly Stochastic Matrix Properties"]
        P1["∑<sub>j</sub> H<sub>ij</sub> = 1<br/>(rows sum to 1)"]
        P2["∑<sub>i</sub> H<sub>ij</sub> = 1<br/>(columns sum to 1)"]
        P3["H<sub>ij</sub> ≥ 0<br/>(non-negative)"]
    end
    
    subgraph Benefits["✨ Benefits"]
        B1["Mean preserved"]
        B2["Norms bounded"]
        B3["Products stable"]
    end
    
    DS --> Benefits
</div>

### Why This Works

If $H$ is doubly stochastic:

<div class="math-block">

$$\text{mean}(Hx) = \text{mean}(x)$$

</div>

**Critical property:** The product of doubly stochastic matrices is also doubly stochastic!

<div class="math-block">

$$H_1 \cdot H_2 \cdot \cdots \cdot H_L \text{ is doubly stochastic if each } H_i \text{ is}$$

</div>

This guarantees **identity preservation across arbitrarily deep networks**.

---

## 6. Sinkhorn–Knopp Algorithm

How do we enforce the doubly stochastic constraint during training?

**Sinkhorn–Knopp algorithm** [5] projects any non-negative matrix onto the set of doubly stochastic matrices.

<div class="mermaid">
graph TB
    A["📊 Unconstrained Matrix A"] --> S1["1️⃣ Normalize rows"]
    S1 --> S2["2️⃣ Normalize columns"]
    S2 --> C{Converged?}
    C -->|No| S1
    C -->|Yes| B["✅ Doubly Stochastic B"]
    
    style A fill:#dc2626,stroke:#dc2626,color:#fff
    style B fill:#22c55e,stroke:#22c55e,color:#fff
</div>

### Example

**Before Sinkhorn projection:**

```
A = | 1.37  1.79  1.51 |    Row sums: 4.67, 4.04, 5.73
    | 1.36  1.06  1.62 |    Col sums: 3.83, 5.08, 5.53
    | 1.09  2.23  2.41 |
```

**After Sinkhorn projection:**

```
B = | 0.355  0.366  0.279 |    Row sums: 1.0, 1.0, 1.0 ✓
    | 0.406  0.250  0.344 |    Col sums: 1.0, 1.0, 1.0 ✓
    | 0.239  0.385  0.376 |
```

---

## 7. Identity Preservation in Action

**Input vector:** $x = [1, 2, 3]$, mean = 2

<div class="mermaid">
graph LR
    subgraph UC["🔴 Unconstrained HC"]
        I1["[1, 2, 3]<br/>mean = 2"] --> A["A × x"] --> O1["[9.5, 8.3, 12.8]<br/>mean = 10.2 💥"]
    end
</div>

<div class="mermaid">
graph LR
    subgraph MHC["🟢 mHC (Sinkhorn)"]
        I2["[1, 2, 3]<br/>mean = 2"] --> B["B × x"] --> O2["[1.9, 1.9, 2.1]<br/>mean = 2.0 ✓"]
    end
</div>

| Method | Output | Mean | Status |
|--------|--------|------|--------|
| Unconstrained HC | $[9.5, 8.3, 12.8]$ | 10.2 | 🚨 5× explosion |
| mHC (Sinkhorn) | $[1.9, 1.9, 2.1]$ | 2.0 | ✅ Preserved |

---

## 8. Physical Intuition: Conservation Law

<div class="mermaid">
graph LR
    subgraph Water["💧 Water Pipe Analogy"]
        P1["Pipe 1"] --> V["🔧 Valves<br/>(Matrix H)"]
        P2["Pipe 2"] --> V
        P3["Pipe 3"] --> V
        V --> O1["Out 1"]
        V --> O2["Out 2"]
        V --> O3["Out 3"]
    end
</div>

Think of each stream as a **water pipe** and matrix weights as **valves**.

**Sinkhorn constraint ensures:**
- 💧 No water is created
- 💧 No water is destroyed
- 💧 Only redistributed

mHC introduces a **conservation law for information flow** in neural networks.

---

## 9. Scaling in Practice

mHC also includes **practical optimizations** [4]:

| Optimization | Benefit |
|-------------|---------|
| Kernel fusion | Reduced memory transfers |
| Mixed-precision (BF16) | 2× memory efficiency |
| Selective recomputation | Lower memory footprint |
| Communication overlap | Better multi-GPU scaling |

**In practice:**
- Expansion rate $n = 4$
- Only **~6.7% training slowdown**
- Massive stability gains at scale

---

## 10. Summary

<div class="mermaid">
graph LR
    subgraph Evolution["Evolution of Residual Architectures"]
        R["2016<br/>ResNet"] --> HC["2024<br/>Hyper-Connections"] --> MHC["2025<br/>mHC"]
    end
    
    R2["✅ Stable<br/>❌ Limited"] --> HC2["✅ Expressive<br/>❌ Unstable"]
    HC2 --> MHC2["✅ Expressive<br/>✅ Stable"]
    
    style R fill:#3b82f6,stroke:#3b82f6,color:#fff
    style HC fill:#f59e0b,stroke:#f59e0b,color:#000
    style MHC fill:#22c55e,stroke:#22c55e,color:#fff
</div>

<div class="highlight-box">

**Key Takeaways:**

1. **Residual connections** enable deep training via identity preservation
2. **Hyper-Connections** add expressiveness but break stability guarantees
3. **mHC** restores stability by constraining mixing to doubly stochastic matrices
4. **Sinkhorn–Knopp** projection efficiently enforces this constraint during training

</div>

> **"Sinkhorn–Knopp turns arbitrary mixing into structured averaging, preserving identity mappings across extreme depth."**

This is why mHC scales where vanilla HC does not—and why it is a promising building block for the next generation of large-scale models.

---

## References

[1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep Residual Learning for Image Recognition**. *CVPR 2016*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

[2] Vaswani, A., et al. (2017). **Attention Is All You Need**. *NeurIPS 2017*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

[3] Zhu, Y., et al. (2024). **Hyper-Connections**. *arXiv preprint*. [arXiv:2409.19606](https://arxiv.org/abs/2409.19606)

[4] Yang, D., et al. (2025). **Manifold-Constrained Hyper-Connections for Stable LLM Training**. *arXiv preprint*. [arXiv:2506.08095](https://arxiv.org/abs/2506.08095)

[5] Sinkhorn, R., & Knopp, P. (1967). **Concerning nonnegative matrices and doubly stochastic matrices**. *Pacific Journal of Mathematics*, 21(2), 343-348.

[6] Birkhoff, G. (1946). **Three observations on linear algebra**. *Univ. Nac. Tucumán Rev. Ser. A*, 5, 147-151.

---

*If you found this post useful, connect with me on [LinkedIn](https://www.linkedin.com/in/sandeep-pandey-43790921/) or check out my publications on [Google Scholar](https://scholar.google.com/citations?user=NveAdp8AAAAJ&hl=en).*
