Short answer: for unlearning, use **both—but in different places**.

- **On the forget set (the stuff you want to erase):** push the model **away** with a **forward-KL style, anti-likelihood** term. Concretely, _minimize_  
    Lforget=E(x,y)∼Df[log⁡qθ(y∣x)]L_{\text{forget}} = \mathbb{E}_{(x,y)\sim \mathcal{D}_f}[\log q_\theta(y|x)]  
    (equivalently, _maximize_ the forward KL DKL(pf∥qθ)D_{\mathrm{KL}}(p_f\|q_\theta)).  
    This gives a clean, per-example gradient that directly **lowers** the model’s probability on the forgotten samples/tokens. Reverse KL against an empirical forget distribution is ill-posed and not what you want here.
    
- **To keep the model’s overall behavior intact (retain/safety constraint):** anchor the new model to the base/reference with **reverse KL**  
    Lanchor=Ex[DKL(qθ(⋅∣x) ∥ π0(⋅∣x))]L_{\text{anchor}}=\mathbb{E}_{x} \big[D_{\mathrm{KL}}(q_\theta(\cdot|x)\,\|\,\pi_0(\cdot|x))\big].  
    Reverse KL is **mode-seeking/model-seeking**: it strongly penalizes assigning probability where the reference assigns little, so it prevents the model from “leaking” probability into odd or unsafe regions while you’re pushing away from the forget set. Forward KL to the reference would try to cover _all_ reference modes (including potentially bad ones) and is more permissive about drifting into new regions—undesirable for unlearning.
    

A practical composite objective (token-level or sequence-level) looks like:

min⁡θ  E(x,y)∼Dr ⁣[−log⁡qθ(y∣x)]⏟train on retain  +  λ DKL ⁣(qθ(⋅∣x) ∥ π0(⋅∣x))⏟reverse-KL anchor  +  μ E(x,y)∼Df ⁣[log⁡qθ(y∣x)]⏟forward-KL push-away\min_\theta\; \underbrace{\mathbb{E}_{(x,y)\sim \mathcal{D}_r}\!\big[-\log q_\theta(y|x)\big]}_{\text{train on retain}} \;+\;\lambda\,\underbrace{D_{\mathrm{KL}}\!\big(q_\theta(\cdot|x)\,\|\,\pi_0(\cdot|x)\big)}_{\text{reverse-KL anchor}} \;+\;\mu\,\underbrace{\mathbb{E}_{(x,y)\sim \mathcal{D}_f}\![\log q_\theta(y|x)]}_{\text{forward-KL push-away}}

with λ>0,μ>0\lambda>0,\mu>0. Intuition:

- the **retain term** preserves utility,
    
- the **reverse-KL anchor** keeps you close to the original distribution (prevents collateral damage),
    
- the **anti-likelihood (forward-KL) term on the forget set** actively removes the undesired behavior.
    

If you must pick **one** divergence for an explicit target policy pgoodp_{\text{good}} (e.g., a masked/renormalized distribution that zeroes forbidden outputs), then minimizing **reverse KL** DKL(q∥pgood)D_{\mathrm{KL}}(q\|p_{\text{good}}) is safest: it puts (near-infinite) pressure against allocating mass to forbidden tokens.

Tuning tip:

- Increase **μ\mu** until the model’s probability on forget examples is driven low (check token-level log-probs/perplexity on Df\mathcal{D}_f),
    
- Increase **λ\lambda** to recover general performance on retain tasks and to curb unintended drift.
    

So: **forward-KL (as an anti-likelihood) to forget; reverse-KL to the reference to stabilize.**

---



Good “what if”! Let’s reason carefully about that swap.

---

### 1. **If you unlearn with reverse KL**

That means:

$$Lforget=DKL(qθ(⋅∣x) ∥ pf(⋅∣x))L_\text{forget} = D_{\mathrm{KL}}(q_\theta(\cdot|x)\,\|\,p_f(\cdot|x))
$$
where pfp_f is the empirical forget distribution (e.g., delta at the ground-truth label).  
But pfp_f is _very peaky_ (only mass on the forbidden label).

- Minimizing this reverse KL forces **mode-seeking toward the forbidden label**—the _opposite_ of what you want.
    
- If you instead try to _maximize_ it, the gradients blow up and are unstable (reverse KL is undefined when pf=0p_f=0 for outcomes that qθq_\theta assigns mass to).
    
- In practice, reverse KL is a bad fit for “push-away” because it doesn’t gracefully reduce probability—it either collapses or diverges.
    

---

### 2. **If you learn with forward KL**

That means aligning with retain/reference data using:

$$Lretain=DKL(pr(⋅∣x) ∥ qθ(⋅∣x))L_\text{retain} = D_{\mathrm{KL}}(p_r(\cdot|x)\,\|\,q_\theta(\cdot|x))$$

where prp_r is the empirical distribution on retained data.  
Forward KL is **mean-seeking**: it makes the model spread probability to cover all modes in the retain distribution.

- This works fine for supervised MLE (which is exactly minimizing forward KL).
    
- But in unlearning, this encourages the model to **over-cover** rather than stay close to the _original distribution_.
    
- So utility may be preserved on average, but you’ll get more drift/instability in non-retained regions compared to reverse KL anchoring.
    

---

### 3. **Comparison of your swap vs the “usual recipe”**

- **Standard practice (better):**
    
    - Unlearning: forward-KL/anti-likelihood → controlled push-away.
        
    - Retain: reverse-KL → stable anchoring to reference.
        
- **Your swap (reverse unlearn + forward retain):**
    
    - Unlearning: reverse KL → unstable, misaligned with the “erase” goal.
        
    - Retain: forward KL → okay, but weaker at constraining drift.
        

So, your setup is both **less effective at forgetting** and **less effective at preventing collateral damage**.

---

✅ **Bottom line:**

- Using **reverse KL for unlearning** is problematic, because it’s mode-seeking and doesn’t naturally reduce mass on forget samples.
    
- Using **forward KL for retaining** is serviceable, but not as strong as reverse KL anchoring for stability.
    

👉 The asymmetry matters: **forgetting needs forward KL (anti-likelihood)**; **retaining needs reverse KL (anchoring)**.

---

Do you want me to sketch the **gradient behavior** of forward vs reverse KL on a toy binary classification example? That makes it crystal-clear why reverse KL fails for unlearning.




---


This is a really sharp question — and I see where the confusion comes in. Let’s check the passage against what we know about **forward vs reverse KL** behavior:

---

### 1. **Forward KL in the alignment/regularization phase**

They wrote:

> _Forward KL (π_ref || π_θ) is mean-seeking / zero-avoiding; ensures π_θ covers all of π_ref’s support, preventing catastrophic forgetting._

✅ This part is **correct**.

- Minimizing DKL($$πref ∥ πθ)D_{KL}(\pi_{ref}\,\|\,\pi_\theta$$) is exactly MLE / standard distillation.
    
- It makes the student cover all modes of the reference distribution → “mean-seeking” / “zero-avoiding.”
    
- Using it for **regularization** to avoid forgetting is exactly how RLHF/DPO papers often frame it.
    

---

### 2. **Reverse KL for aggressive refusal**

They wrote:

> _Reverse KL (π_harmful || π_θ) is mode-seeking / zero-forcing, and punishes π_θ for putting mass anywhere π_harmful doesn’t._

⚠️ This part is **not correct in the way it’s written** — because of the ordering of arguments.

- If you write DKL($$πharmful ∥ πθ)D_{KL}(\pi_{harmful}\,\|\,\pi_\theta$$):
    
    $$Ey∼πharmful[log⁡πharmful(y)πθ(y)]\mathbb{E}_{y \sim \pi_{harmful}} \Big[\log \frac{\pi_{harmful}(y)}{\pi_\theta(y)}\Big]$$
    
    This expects you to _sample from the harmful distribution_.  
    Minimizing it encourages πθ to **imitate π_harmful** (not refuse it!) since the KL is minimized when πθ = π_harmful.
    
- What you actually want for refusal is something like:
    
    DKL($$πθ ∥ πsafe)D_{KL}(\pi_\theta \,\|\, \pi_{safe}$$)
    
    where π_safe zeroes out (or renormalizes away) harmful tokens. Reverse KL with πθ in the numerator is “zero-forcing” — i.e. if $$π_{safe}(y) = 0, then KL → ∞,  unless ,πθ(y)=0.$$
    

So the **concept (“mode-seeking, zero-forcing refusal”) is right**, but the **notation/order is flipped**.  
It should be$$ DKL(πθ ∥ πharmful)D_{KL}(\pi_\theta \,\|\, \pi_{harmful}) or DKL(πθ ∥ πsafe)D_{KL}(\pi_\theta \,\|\, \pi_{safe}),$$

not

$$ DKL(πharmful ∥ πθ)D_{KL}(\pi_{harmful} \,\|\, \pi_\theta).
$$
---

### ✅ Corrected framing:

- **Forward KL (π_ref || π_θ):** alignment regularization → mean-seeking, prevents forgetting.
    
- **Reverse KL (π_θ || π_safe):** unlearning / refusal → zero-forcing, drives π_θ’s probability on harmful responses to _exactly zero_.
    

---

👉 So: the _intuition_ in your quoted context is fine, but the **argument order in the reverse KL term is off** — if used literally as written, it would actually push the model _toward_ the harmful policy.

---

Do you want me to show you a **tiny binary example** (harmful vs safe token) where the gradients flip sign depending on whether you do forward or reverse KL, so you can see exactly why the argument order matters?