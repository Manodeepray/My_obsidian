


# lora
## Why Is Unlearning Hard with LoRA?

### 1. **LoRA only _adds_ delta weights**

- The base model is frozen.
    
- The LoRA adapters **modify the behavior**, but don’t overwrite the original knowledge.
    
- So when you delete or reverse them — you’re just **removing the influence**, not the knowledge.
    

> ⚠️ Base model still “remembers” everything.

---

### 2. **Unlearning ≠ Subtracting LoRA**

If LoRA made the model learn something (say "X is true"), deleting it won't **revert** the knowledge — it just removes the _bias_ LoRA added.

If the base model already **kind of knows X**, you're stuck.

---

### 3. **Catastrophic forgetting doesn't help**

You can't fine-tune a new LoRA adapter to "unlearn" a thing (e.g., private data, bias, false info) without **risking hurting general performance** — especially with a small rank.

LoRA isn't good at:

- Targeted forgetting
    
- Undoing specific facts
    
- Preserving performance while forgetting just one slice
    

---

## 📉 How Bad Is It?

### ❌ Bad for:

- GDPR/right-to-be-forgotten use cases
    
- Safety (e.g., forgetting jailbreak prompts or hate speech)
    
- Deleting proprietary info after fine-tuning
    

### ✅ Less bad for:

- Domain adapters (e.g., science vs. finance)
    
- Switching between tasks — if you’re not "unlearning" per se, but replacing behaviors
    

---

## 🛠️ Is There Hope?

Kind of. Here’s what researchers are exploring:

- **"Unlearning via negative training"**: training LoRA adapters to make the model _disprefer_ certain outputs
    
- **"Rewriting memory with LoRA"**: forcibly overriding factual knowledge (e.g., ROME-like methods with LoRA)
    
- **LoRA + model editing**: try to surgically edit specific neurons or attention heads with auxiliary losses
    

But none of these are plug-and-play today.

---

## 🚨 Bottom Line:

|Question|Answer|
|---|---|
|Can I remove knowledge with LoRA?|❌ No, not fully.|
|Is it dangerous to assume unlearning via LoRA works?|✅ Yes, it could lead to misleading model behavior.|
|Is research ongoing?|🔬 Yes, but it's early days.|
|Safer approach?|Retrain from scratch, or use full model editing techniques.|

---
## 🔹 1. FLoRA (Fisher-weighted LoRA)

**Paper**: _"Towards Robust and Cost‑Efficient Knowledge Unlearning for LLMs"_ (Aug 2024) [reddit.com+13arxiv.org+13scribd.com+13](https://arxiv.org/html/2408.06621v1?utm_source=chatgpt.com)  
**Authors**: Zhao et al.

### 🛠️ Method:

- Initializes LoRA adapters using **Fisher information**, giving more weight to parameters actually responsible for the target (to‑be‑unlearned) knowledge.
    
- Trains with a **hinge/unlearning loss (IHL)** and Fisher-weighted initial adapter (FLoRA) to push the model away from specific knowledge.
    

### 🚀 Observations:

- Vanilla gradient descent + LoRA **fails** to unlearn reliably.
    
- Combining **IHL + Fisher init (FLoRA)** improves unlearning success and better **preserves general capabilities**, but still **decreases performance**.
    
- Higher LoRA rank helps, but at **significant cost to model utility** [reddit.com+8arxiv.org+8arxiv.org+8](https://arxiv.org/html/2408.06621v1?utm_source=chatgpt.com).
    

### 📝 Author’s Take:

- Fisher initialization speeds up unlearning and reduces collateral forgetting.
    
- Yet, even FLoRA suffers meaningful degradation compared to full fine-tune or editing.
    
- Unlearning with parameter-efficient adapters is “promising but imperfect.”
    

---

## 🔹 2. MEND (Model Editing Networks for Editing)

**Paper**: _"MEND: Using meta-learning to revise model behavior"_ (Meng et al.) detailed in evaluations [arxiv.org+2arxiv.org+2ar5iv.labs.arxiv.org+2](https://arxiv.org/html/2401.07453?utm_source=chatgpt.com)[arxiv.org+3reddit.com+3arxiv.org+3](https://www.reddit.com/r/MachineLearning/comments/1fwqgfx?utm_source=chatgpt.com)[reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/15l8oyv?utm_source=chatgpt.com)

### 🛠️ Method:

- A **meta-learned hypernetwork** predicts weight updates needed to rewrite a fact in a frozen LLM.
    
- Applies targeted edits with minimal parameter changes.
    

### 🚫 Observations:

- Works for **single edits**, but **fails** in sequential edits—accuracy drops with more edits [reddit.com+13arxiv.org+13medium.com+13](https://arxiv.org/html/2401.07453?utm_source=chatgpt.com).
    
- Compared to ROME, MEND shows **weaker locality** and less consistent results, especially for large batches [arxiv.org+8github.com+8arxiv.org+8](https://github.com/pritamgouda11/Editing-Techniques-Across-LLM-Architectures/blob/main/README.md?utm_source=chatgpt.com).
    

### 📝 Author’s Take:

- Effective for controlled single edits, but scalability and locality issues limit broader usability.
    

---

## 🔹 3. ROME (Rank-One Model Editing)

**Paper**: _Meng et al. (2022)_; analyzed extensively in sequential editing study

### 🛠️ Method:

- Identifies the **key-value associative memory** inside MLP layers.
    
- Computes a **rank-1 update** to insert a new (key, value) fact into the model.
    

### 🚧 Observations:

- Near-100% success rate for **initial edits**, but **neighborhood accuracy declines** after ~100–1000 edits—performance bleed across facts [arxiv.org+5ar5iv.labs.arxiv.org+5researchgate.net+5](https://ar5iv.labs.arxiv.org/html/2402.11122?utm_source=chatgpt.com)[reddit.com+4researchgate.net+4arxiv.org+4](https://www.researchgate.net/publication/381704714_How_Well_Can_Knowledge_Edit_Methods_Edit_Perplexing_Knowledge?utm_source=chatgpt.com)[reddit.com+2arxiv.org+2ar5iv.labs.arxiv.org+2](https://arxiv.org/html/2401.07453?utm_source=chatgpt.com).
    
- Outperforms MEND in **locality** and **generalization** for single edits [scribd.com+4github.com+4arxiv.org+4](https://github.com/pritamgouda11/Editing-Techniques-Across-LLM-Architectures/blob/main/README.md?utm_source=chatgpt.com).
    

### 📝 Author’s Take:

- Precise for small-scale edits.
    
- Sequential application causes **catastrophic degradation due to interference**.
    

---

## 🔹 4. Editing-as-Unlearning (WISE, AlphaEdit)

**Paper**: _"Editing as Unlearning"_ (May 2025)

### 🛠️ Method:

- Treats unlearning as editing the model to **refuse answering** specific prompts.
    
- Applies editing methods like **WISE** and **AlphaEdit**, plus techniques like self-improvement and query merging.
    

### 🚀 Observations:

- WISE and AlphaEdit—ordinary editing methods—are **strong baselines** for unlearning.
    
- Achieve **human-aligned refusal behaviors** for pretrained facts without extensive retraining.
    

### 📝 Author’s Take:

- Propose **editing-focused unlearning recipes** as a practical path forward.
    
- Encourage community adoption of editing methods for unlearning tasks.
    

---

## 🔹 5. PruneLoRA (Prune → LoRA Adapt → Unlearn)

**Paper**: _"LoRA Unlearns More and Retains More"_ (Nov 2024) student work [arxiv.org+1arxiv.org+1](https://arxiv.org/abs/2505.19855?utm_source=chatgpt.com)

### 🛠️ Method:

1. **Prune** the base model to reduce capacity.
    
2. **Apply LoRA adapters** on the pruned subnetwork.
    
3. **Train to unlearn**, leveraging smaller parameter space.
    

### 🚀 Observations:

- More efficient than full-model approaches.
    
- Better retention of retained classes than fine-tuning or pruning alone.
    

### 📝 Author’s Take:

- Pruning before adapter training helps balance forgetting unwanted classes and preserving performance.
    
- Still a **compromise**, not perfect – results vary by architecture and dataset.
    

---

## 🧩 Summary Table

|Method|Approach|Unlearning Success|Collateral Damage|Scale Issues|
|---|---|---|---|---|
|**FLoRA**|LoRA + Fisher init + hinge loss|✅ Improved|⚠️ Moderate|Yes, costly|
|**MEND**|Meta-learned weight edits|🔸 Good (singles)|❌ Bad (sequential)|Yes|
|**ROME**|Rank‑1 insertion|✅ Excellent (initial)|⚠️ Bleed-over|Yes|
|**WISE, Alpha**|Editing-based refusal|✅ Effective|🚫 Low|Limited data|
|**PruneLoRA**|Prune → LoRA adapt|✅ Efficient|⚠️ Balanced|Varies|

---

### 🧠 Final Takeaways:

- **LoRA alone is not enough** to truly unlearn; base model knowledge remains intact.
    
- **FLoRA** offers the best LoRA-based path so far—Fisher-informed adapters + tailored loss—but still sacrifices model capability.
    
- **Editing methods (ROME, AlphaEdit, WISE)** can effectively remove specific knowledge with **higher fidelity and less side effects**.
    
- **Hybrid strategies** (e.g., pruning + LoRA, editing + self-improvement) show promise but remain experimental.







