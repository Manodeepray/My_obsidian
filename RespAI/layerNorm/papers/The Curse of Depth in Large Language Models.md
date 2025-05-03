
github : https://github.com/lmsdss/LayerNorm-Scaling

## Layer Normalization in LLMs:

brief :
	layer norm normalizes the  inputs across the features for each datasamples instead of normalizing along the batch (Batch normalization)

Formula:

For a given input vector 
$$x=[x1,x2,...,xd] 
$$

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

$$
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i \\
\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
$$

- μ\mu: Mean of the input vector
    
- σ2\sigma^2: Variance of the input vector
    
- γ,β\gamma, \beta: Learnable scale and shift parameters
    
- ϵ\epsilon: Small constant for numerical stability
    

---

### used in Transformers

Layer normalization is used at **multiple points** in the Transformer architecture:

#### 1. **Before or after residual connections** (depends on the variant like Pre-LN or Post-LN):

```text
Post-LN (original Transformer):
x = x + Sublayer(x)
x = LayerNorm(x)

Pre-LN (used in newer models like GPT-2, GPT-3):
x = x + Sublayer(LayerNorm(x))
```

#### 2. **After multi-head attention and feed-forward layers**

---

### 🚀 **Why is it important?**

1. **Stabilizes training**: Keeps the activations from exploding or vanishing
    
2. **Improves convergence**: Makes gradients behave nicely during backpropagation
    
3. **Enables deep networks**: Without normalization, stacking many layers would be hard
    

---

### 🧪 How it's different from BatchNorm?

| Aspect           | LayerNorm                   | BatchNorm                     |
| ---------------- | --------------------------- | ----------------------------- |
| Normalization    | Across features per example | Across batch for each feature |
| Sequence Models  | ✅ Good for Transformers     | ❌ Not ideal                   |
| Input Dependency | Independent of batch size   | Depends on batch statistics   |

---

**Curse of Depth,** 
a concept that highlights, explains, and addresses the recent observation in modern Large Language Models (LLMs) where ==nearly half of the layers are less effective than expected==

underlying reason for the ineffectiveness of deep layers in LLMs is the ==widespread usage of Pre-Layer Normalization (Pre-LN).== 	

While Pre-LN stabilizes the training of Transformer LLMs, its output variance exponentially grows with the model depth, which undesirably causes the derivative of the deep Transformer blocks to be an identity matrix, and therefore barely contributes to the training

---

**LayerNorm Scaling**, 
which scales the *variance of output* of the layer normalization ==inversely by the square root of its depth==

---

**The Curse of Depth.**
The Curse of Depth refers to the observed phenomenon where ==deeper layers in modern large language models (LLMs) contribute significantly less to learning== and representation compared to earlier layers. These **deeper layers** often exhibit ==remarkable robustness to pruning and perturbations==, implying they fail to perform meaningful transformations. This behavior prevents these layers from effectively contributing to training and representation learning, resulting in resource inefficiency.

---
**Root Cause of CoD.**
We theoretically and empirically identify the root cause of CoD as the use of ==PreLayer Normalization (Pre-LN)== (Baevski and Auli, 2019; Dai et al., 2019), which normalizes layer inputs before applying the main computations, such as attention or feedforward operations, rather than after. 

Specifically, while stabilizing training, we observe that the output variance of Pre-LN accumulates significantly with layer depth (see Appendix C), causing the derivatives of deep Pre-LN layers to approach an identity matrix. This behavior prevents these layers from introducing meaningful transformations, leading to diminished representation learning

---
**Mitigating CoD through LayerNorm Scaling**


LayerNorm Scaling, which scales the output of Layer Normalization by the square root of the depth 1/√L .


![[Pasted image 20250408105514.png]]



![[Pasted image 20250408111440.png]]