---
layout: post
title: "From Residual Connections to Manifold-Constrained Hyper-Connections"
date: 2026-01-06
categories: [deep-learning, transformers, research]
tags: [ResNet, Hyper-Connections, mHC, LLM, Sinkhorn-Knopp]
---

<link rel="stylesheet" href="/assets/css/custom.css">
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>mermaid.initialize({startOnLoad:true, theme:'dark'});</script>

# From Residual Connections to Manifold-Constrained Hyper-Connections
### A Simple, Intuitive, and Visual Guide

Deep neural networks owe much of their success to a deceptively simple idea: **the residual connection**. Since the introduction of ResNets in 2016, residual connections have become a foundational design element across modern architectures—most notably Transformers and large language models (LLMs).

Recently, **Hyper-Connections (HC)** proposed a bold extension to residuals by widening the residual stream and increasing connection complexity—unlocking new representational power. However, this power comes with a hidden cost: **training instability at scale**.

In this post, we'll build intuition step-by-step for:
* Why residual connections work
* What goes wrong in Hyper-Connections
* How **Manifold-Constrained Hyper-Connections (mHC)** fix the problem
* Why **Sinkhorn–Knopp projection** is the key ingredient
* What all of this looks like in **simple toy simulations**

No prior knowledge of optimal transport or advanced matrix theory is required.

---

## 1. Why Residual Connections Changed Everything

A standard neural network layer replaces its input:

$$x_{l+1} = F(x_l)$$

Residual connections modify this to:

$$x_{l+1} = x_l + F(x_l)$$

<div class="mermaid">
flowchart LR
    subgraph Standard["Standard Layer"]
        A1[x_l] --> F1[F] --> B1[x_l+1]
    end
    subgraph Residual["Residual Connection"]
        A2[x_l] --> F2[F]
        A2 --> add((+))
        F2 --> add
        add --> B2[x_l+1]
    end
</div>

### Why does this help?

You can think of a residual layer as:

> "Keep what you already know, and apply a small correction."

This simple change enables:
* Stable training at great depth
* Protection against vanishing gradients
* An **identity fallback**: if F(x)=0, the network behaves like the identity function

Stacking many residual layers yields:

$$x_L = x_l + \sum_{i=l}^{L-1} F(x_i)$$

The original signal x_l flows **unchanged** through all deeper layers. This property—**identity mapping**—is the secret behind ResNet's stability.

<div class="mermaid">
flowchart LR
    x0[x_0] --> add1((+))
    x0 --> F1[F_1]
    F1 --> add1
    add1 --> add2((+))
    add1 --> F2[F_2]
    F2 --> add2
    add2 --> add3((+))
    add2 --> F3[F_3]
    F3 --> add3
    add3 --> xL[x_L]
    
    style x0 fill:#06b6d4
    style xL fill:#8b5cf6
</div>

---

## 2. Hyper-Connections: Making Residuals Wider

Hyper-Connections (HC) generalize residual connections by introducing **multiple parallel residual streams**.

Instead of:
* One stream x ∈ ℝ^C

We now have:
* n streams → x ∈ ℝ^(n×C)

<div class="mermaid">
flowchart TB
    subgraph HC["Hyper-Connections (n=3 streams)"]
        direction TB
        subgraph Input["Input Streams"]
            s1[Stream 1]
            s2[Stream 2]
            s3[Stream 3]
        end
        
        H1[H_pre]
        F[F - Core Function]
        H2[H_post]
        H3[H_res]
        
        Input --> H1
        H1 --> F
        F --> H2
        H3 --> MIX((Mix))
        H2 --> MIX
        
        subgraph Output["Output Streams"]
            o1[Stream 1']
            o2[Stream 2']
            o3[Stream 3']
        end
        MIX --> Output
    end
</div>

Each layer **mixes** these streams using a learnable matrix:

$$x_{l+1} = H^{\text{res}}_l x_l + H^{\text{post}}_l{}^\top F(H^{\text{pre}}_l x_l)$$

The idea is powerful:
* Multiple pathways
* Richer interaction
* No extra FLOPs in the core function F

But something subtle breaks.

---

## 3. The Hidden Instability in Hyper-Connections

In ResNet, the identity path is literal: weight = 1.

In HC, the identity path becomes:

$$x_{l+1} = H^{\text{res}}_l x_l$$

<div class="mermaid">
flowchart LR
    subgraph Problem["The Instability Problem"]
        direction LR
        x0[x_0] --> H1[H_res^1] --> H2[H_res^2] --> H3[H_res^3] --> dots[...] --> HL[H_res^L] --> xL[x_L]
    end
    
    subgraph Result["After L Layers"]
        product["x_L = (∏ H_res) × x_0"]
    end
    
    style Problem fill:#1a1a2e
    style Result fill:#dc2626,color:#fff
</div>

Here's the problem:
* H_res is **unconstrained**
* Across many layers, matrix products can:
  * Amplify signals
  * Attenuate signals
  * Destroy mean preservation

After many layers:

$$x_L = \left(\prod_i H^{\text{res}}_i\right) x_l + \cdots$$

There is **no guarantee** that this product behaves like identity.

---

## 4. A 1D Toy Simulation (Why This Matters)

Consider a 1D signal x, with a small stabilizing residual:

$$F(x) = -0.05x$$

### Three cases

| Architecture | Update rule | Behavior |
|:------------|:-----------|:---------|
| **ResNet** | x_{l+1} = x_l + F(x_l) | ✅ Stable decay |
| **HC** | x_{l+1} = 1.08 · x_l + F(x_l) | 🚨 Slow explosion |
| **mHC** | x_{l+1} = 1.0 · x_l + F(x_l) | ✅ Stable decay |

<div class="mermaid">
xychart-beta
    title "Signal Magnitude Over Depth"
    x-axis "Layer Depth" [0, 10, 20, 30, 40, 50]
    y-axis "Signal Magnitude" 0 --> 5
    line "ResNet" [1, 0.6, 0.35, 0.2, 0.12, 0.08]
    line "mHC" [1, 0.6, 0.35, 0.2, 0.12, 0.08]
    line "HC (unstable)" [1, 1.5, 2.2, 3.3, 4.5, 6.0]
</div>

### What happens?
* **ResNet & mHC**: signal decays smoothly and predictably
* **HC**: signal *slowly explodes* despite damping

Gradients behave similarly:
* HC gradients grow like (1.03)^L
* ResNet/mHC gradients remain controlled

This is not a hypothetical issue—it becomes catastrophic at LLM scale.

---

## 5. What Is mHC Trying to Enforce?

mHC asks a simple question:

> "How do we allow rich mixing across streams while preserving identity behavior?"

The answer:

### **Constrain residual mixing to be a convex combination**

Mathematically, this means constraining:

$$H^{\text{res}} \in \text{Birkhoff polytope}$$

<div class="mermaid">
flowchart TB
    subgraph Constraint["Doubly Stochastic Constraint"]
        direction TB
        R1["Row sums = 1"] 
        R2["Column sums = 1"]
        R3["All entries ≥ 0"]
    end
    
    subgraph Result["Result"]
        M["H is a<br/>Doubly Stochastic<br/>Matrix"]
    end
    
    Constraint --> M
    
    style M fill:#06b6d4
</div>

In plain terms:
* Rows sum to 1
* Columns sum to 1
* All entries are non-negative

These are called **doubly stochastic matrices**.

---

## 6. Why Doubly Stochastic Matrices Are Special

Take a matrix H with:

$$\sum_j H_{ij} = 1,\quad \sum_i H_{ij} = 1$$

Then:
* Each output is a **weighted average** of inputs
* The **mean of features is preserved**
* Norms are naturally regularized

Most importantly:

$$\text{Product of doubly stochastic matrices is also doubly stochastic}$$

<div class="mermaid">
flowchart LR
    DS1["DS Matrix 1"] --> MUL((×)) --> DS2["DS Matrix 2"] --> MUL2((×)) --> DSn["DS Matrix n"] --> Result["Still DS!"]
    
    style DS1 fill:#06b6d4
    style DS2 fill:#8b5cf6
    style DSn fill:#06b6d4
    style Result fill:#22c55e
</div>

So identity preservation holds **across arbitrarily deep networks**.

---

## 7. Sinkhorn–Knopp: Turning Chaos into Conservation

How do we enforce this constraint in practice?

Enter the **Sinkhorn–Knopp algorithm**.

### Algorithm intuition

<div class="mermaid">
flowchart TB
    A[Unconstrained Matrix A] --> S1[Normalize Rows]
    S1 --> S2[Normalize Columns]
    S2 --> Check{Converged?}
    Check -->|No| S1
    Check -->|Yes| B[Doubly Stochastic B]
    
    style A fill:#dc2626
    style B fill:#22c55e
</div>

1. Normalize rows to sum to 1
2. Normalize columns to sum to 1
3. Repeat until convergence

### Visual example

We start with an unconstrained matrix A:

```
A = | 1.372  1.788  1.507 |
    | 1.362  1.059  1.615 |
    | 1.094  2.229  2.409 |
```

Row/column sums are wildly uneven.

After Sinkhorn projection:

```
B = | 0.355  0.366  0.279 |
    | 0.406  0.250  0.344 |
    | 0.239  0.385  0.376 |
```

✅ All rows sum to 1  
✅ All columns sum to 1

---

## 8. Identity Preservation in Action

Let the input be:

$$x = [1, 2, 3] \quad \Rightarrow \quad \text{mean} = 2$$

### Unconstrained HC

$$Ax = [9.47,\ 8.33,\ 12.78] \quad \Rightarrow \quad \text{mean} = 10.19$$

🚨 Signal explosion

### mHC (Sinkhorn constrained)

$$Bx = [1.92,\ 1.94,\ 2.14] \quad \Rightarrow \quad \text{mean} = 2$$

✅ Mean preserved  
✅ Energy conserved  
✅ Stable propagation

<div class="mermaid">
flowchart LR
    subgraph Before["Unconstrained HC"]
        I1["[1, 2, 3]<br/>mean=2"] --> A1["A × x"] --> O1["[9.47, 8.33, 12.78]<br/>mean=10.19 🚨"]
    end
    
    subgraph After["mHC (Sinkhorn)"]
        I2["[1, 2, 3]<br/>mean=2"] --> A2["B × x"] --> O2["[1.92, 1.94, 2.14]<br/>mean=2 ✅"]
    end
    
    style O1 fill:#dc2626
    style O2 fill:#22c55e
</div>

This is the **identity mapping property restored**, now in multi-stream form.

---

## 9. Physical Intuition: Information Conservation

An analogy that works well:

<div class="mermaid">
flowchart LR
    subgraph Conservation["Water/Information Conservation"]
        direction TB
        P1[Pipe 1] --> V[Valve Matrix]
        P2[Pipe 2] --> V
        P3[Pipe 3] --> V
        V --> O1[Output 1]
        V --> O2[Output 2]
        V --> O3[Output 3]
    end
    
    Note["Sinkhorn ensures:<br/>• No water created<br/>• No water destroyed<br/>• Only redistributed"]
    
    style Note fill:#1a1a2e,stroke:#06b6d4
</div>

* Each stream = pipe of water
* Matrix weights = valves
* Sinkhorn ensures:
  * No water is created
  * No water is destroyed
  * Only redistributed

mHC introduces a **conservation law for information**.

---

## 10. Scaling in Practice

mHC also addresses hardware concerns:
* Kernel fusion
* Mixed-precision kernels
* Selective recomputation
* Communication-compute overlap

In practice:
* Expansion rate n=4
* Only ~**6.7% training slowdown**
* Massive stability gains at scale

---

## Summary: The Evolution of Residual Connections

<div class="mermaid">
timeline
    title Evolution of Residual Architectures
    2016 : ResNet
         : Identity shortcut
         : Single stream
    2024 : Hyper-Connections
         : Multiple streams
         : Unconstrained mixing
         : Instability at scale
    2025 : mHC
         : Multiple streams
         : Sinkhorn constraint
         : Identity preservation
         : Stable at scale
</div>

---

## Final Takeaway

**Hyper-Connections make residuals more powerful, but fragile.**

**Manifold-Constrained Hyper-Connections restore stability by enforcing conservation.**

> **Sinkhorn–Knopp turns arbitrary mixing into structured averaging, preserving identity mappings across extreme depth.**

This is why mHC scales where vanilla HC does not—and why it is a promising building block for the next generation of large-scale models.

---

*If you found this post useful, feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/sandeep-pandey-43790921/) or check out my other publications on [Google Scholar](https://scholar.google.com/citations?user=NveAdp8AAAAJ&hl=en).*
