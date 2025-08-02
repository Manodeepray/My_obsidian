https://arxiv.org/pdf/2311.00915

# INTRODUCTION 
**1. The Problem: LLMs Struggle with Dialects**

- **Dialectal Diversity:** People speak in many different ways due to their background (race, culture, region, age, etc.).
    
- **LLMs Everywhere:** Large Language Models (LLMs) are becoming common tools in our daily lives.
    
- **The Issue:** These LLMs are **not invariant to dialectal differences**. This means they don't perform as well when dealing with English dialects that are very different from Standard American English (SAE).
    
- **Impact:** This performance gap creates fairness concerns (racial, ethnic, socio-economic) for groups whose dialects are **under-represented** in the data used to train these LLMs.
    
- **Serious Consequences:** This can lead to harmful outcomes, like denial of healthcare or racial bias in tools designed to detect hate speech.
    

**2. Past Approaches and Their Limitations:**

- **Previous Methods:** Historically, efforts to make LLMs better with dialects focused on:
    
    - **More Data:** Getting more dialect-specific data, either manually or using less strict forms of supervision.
        
    - **Synthetic Data:** Creating artificial dialect data (like with "Multi-VALUE").
        
- **Shared Problem:** A major drawback of these methods is that they **assume you have dialectal data for every specific task** you want the LLM to do.
    
- **Unrealistic Expectation:** In reality, it's very hard to find people to annotate data in all the different dialects.
    
- **Recent Shift:** Newer work (like "TADA") has tried to reduce the need for _task-specific_ dialect data by training **task-agnostic adapters** that help bridge the gaps between dialects.
    
- **Lingering Need:** Even with these advancements, there's still a need for methods that are **resource-efficient** and can adapt to new dialects without constantly needing new data.
    

**3. Our Solution: HyperLoRA**

- **Introducing HyperLoRA:** We propose a new, efficient method called **HyperLoRA**. Its goal is to adapt LLMs to new dialects _without needing more dialect annotations_ (specific data for each dialect).
    
- **Leveraging Expert Knowledge:** Instead of data, we use **existing expert knowledge about dialects**. Experts can identify a speaker's dialect more easily and cheaply than gathering tons of annotated data.
    
- **Inspiration from Multilingualism:** This idea is similar to how "typological features" (characteristics of language types) have been successfully used to bridge data gaps in multilingual settings.
    
- **How HyperLoRA Works (Technical Details):**
    
    - **Hypernetworks:** We use a **hypernetwork** (a neural network that generates the weights for another network). These are good at generalization.
        
    - **Dialect-Specific LoRA Adapters:** The hypernetwork creates **dialect-specific LoRA adapters**. Remember, LoRA adapters are small, efficient additions that help fine-tune LLMs.
        
    - **Typological Features:** The hypernetwork uses these expert-provided **typological features** as input to generate the LoRA adapters.
        
    - **Minimizing Interference:** By having the hypernetwork handle the complexity of dialect differences and generate specific LoRA adapters, we reduce "cross-dialectal interference" in the main LLM.
        
    - **Training Objective:** The hypernetwork is trained using parallel corpora (texts in different dialects/languages) to optimize a **morphosyntactic alignment objective**. This means it learns to align how words and sentences are structured across dialects in the LLM's "representation space," making it adaptable regardless of the specific task.
        
- **Benefits:**
    
    - This alignment objective is new, well-founded, and easy to calculate.
        
    - Using expert knowledge effectively can be equivalent to having 250 annotations per dialect.
        
    - We also created a new **metric** to measure how well dialect features are covered, helping us understand the limits of using hypernetworks for **zero-shot generalization** (adapting to new, unseen dialects without specific training data for them).

# hyper network

---

### 🔹 **Background**

1. **LoRA (Low-Rank Adaptation)**:
    
    - Proposed by Hu et al. (2021).
        
    - A fine-tuning method where the main model parameters stay frozen.
        
    - Only a **low-rank decomposition** of attention matrices (e.g., `W_q`, `W_v`) is updated: `W ≈ W + D × U`.
        
2. **Hypernetworks**:
    
    - Introduced by Ha et al. (2016).
        
    - A neural network (`g`) that generates weights for another network.
        

---

### 🔹 **Novelty of This Work**

3. Instead of learning LoRA weights (`D`, `U`) directly, the authors **learn a hypernetwork** that **generates** the LoRA weights.
    
4. This is the **first known method** to generate LoRA adapters via hypernetworks **for domain adaptation**, particularly in multi-dialect settings.
    

---

### 🔹 **Hypernetwork Architecture Setup**

5. For each layer `k`:
    
    - `D_k^q`, `U_k^q`: LoRA matrices for the **query**.
        
    - `D_k^v`, `U_k^v`: LoRA matrices for the **value**.
        
6. **Input to the hypernetwork `g`**:
    
    - A **concatenation** of:
        
        - Dialect feature vector `d ∈ [0,1]^#features`.
            
        - Positional embedding `i_k^{q,v} ∈ {0, ..., 2 × #blocks}`.
            
        - So, input `x = concat(d, i_k^{q,v})`.
            
7. **Separate hypernetworks** are used for:
    
    - `D_k^q` and `U_k^q`
        
    - `D_k^v` and `U_k^v`
        

---

### 🔹 **Hypernetwork Function**

8. Each hypernetwork is parameterized by:
    
    - `W_d`: Down-projection weight.
        
    - `W_u`: Up-projection weight.
        
9. For `D_{q,v}` (and similarly `U_{q,v}`), the computation is:
    
    ```text
    x = concat(d, i_k^{q,v})           (1)
    D_k^{q,v}, U_k^{q,v} = g(x), g'(x) (2)
    D_{q,v} = MM(ReLU(MM(x, W_d)), W_u) (3)
    ```
    
    - `MM`: Matrix multiplication.
        
    - A two-layer MLP is used in the hypernetwork.
        

---

### 🔹 **Training and Visualization**

10. The **training pipeline** of HyperLoRA:
    
    - Is illustrated in **Figure 2** and **Algorithm 1** in the paper.
        
    - Involves end-to-end training of the hypernetwork to generate effective LoRA weights based on the input dialect and position.
        

---

Let me know if you want me to explain **Figure 2 or Algorithm 1** next.#


# Dialect-Specific Low-Rank Adaptation AND Morphosyntactic Alignment

---

### ✅ **3.3 Dialect-Specific Low-Rank Adaptation**

1. **Background on Adapter Modules**:
    
    - Prior work (e.g., Lialin et al., Pfeiffer et al.) used **bottleneck adapter modules** after the multi-head attention layer in transformers for **cross-lingual adaptation**.
        
2. **Focus of This Work**:
    
    - Instead of post-attention adapters, this work **targets adaptation _within_ the attention mechanism itself**.
        
3. **Key Hypothesis**:
    
    - Dialect-specific variations, especially **morphosyntactic differences** (e.g., grammar and word order), can be better captured by **modifying self-attention**.
        
    - Reason: Self-attention is highly **sensitive to syntactic structure**, making it ideal for handling these subtle variations.
        
4. **Gap in Literature**:
    
    - There's a lack of systematic evaluation of **PEFT (Parameter-Efficient Fine-Tuning)** modules specifically for **dialectal adaptation**.
        
    - The authors leave a more comprehensive comparison of different PEFT modules for future work.
        

---

### ✅ **3.4 Morphosyntactic Alignment**

5. **Challenge in English Dialects**:
    
    - Unlike major languages, there is **no large corpus of sentence-aligned dialectal bitexts** for English dialects.
        
    - Parallel corpora (like those used in machine translation) are scarce for dialects.
        
6. **Solution Using MultiVALUE**:
    
    - They use the **rule-based translation system from MultiVALUE** to generate **synthetic sentence-aligned dialect data**.
        
    - Although synthetic, MultiVALUE's performance is **predictive of real-world adaptation quality**.
        
7. **Limitation**:
    
    - MultiVALUE only aligns data at the **sentence level**, while the actual dialectal variation is often at the **token/morphosyntactic level**.
        
8. **Token-Level Alignment Strategy**:
    
    - To address this, they use **unsupervised token-level alignment methods** from prior works.
        
    - The goal is to measure how individual **tokens shift** between dialectal and standard English representations.
        
9. **Metric Used: Earth Mover's Distance (Wasserstein Distance)**:
    
    - Denoted as `W(P_DIAL, P_SAE)`:
        
        - `P_DIAL`: Distribution of last-layer embeddings for dialectal sentences.
            
        - `P_SAE`: Distribution of last-layer embeddings for Standard American English (SAE).
            
    - Measures the **minimum cost to transform one distribution into another**.
        
10. **Practical Computation via Sinkhorn Divergence**:
    
    - Exact Wasserstein distance is expensive, so they use **Sinkhorn divergence**:
        
        Sε(α,β)=Wε(α,β)−12Wε(α,α)−12Wε(β,β)S_ε(α, β) = W_ε(α, β) - \frac{1}{2}W_ε(α, α) - \frac{1}{2}W_ε(β, β)
    - Interpolates between Wasserstein distance and **Maximum Mean Discrepancy (MMD)**.
        
11. **Definition of Wε (Sinkhorn distance)**:
    
    Wε(α,β)=min⁡π∈Π(α,β)∫X×Yc(x,y)dπ(x,y)+ε⋅KL(π∥α⊗β)W_ε(α, β) = \min_{\pi ∈ Π(α, β)} \int_{X × Y} c(x, y) dπ(x, y) + ε \cdot KL(π \| α ⊗ β)
    - `π`: Coupling (transport plan) between distributions `α` and `β`.
        
    - `c(x, y)`: Cost function (they use **squared error**).
        
    - `ε`: Entropy regularization parameter (set to **0.05**).
        
    - `KL`: Kullback-Leibler divergence between the coupling and product of marginals.
        
12. **Implementation**:
    
    - Sinkhorn divergence is computed using the **solver from Feydy et al. (2018)**.
        
    - Inputs: Last-layer representations of both dialect and SAE.
        

---

### 📌 Summary

- Section 3.3 proposes adapting attention weights directly for dialect-specific tuning using LoRA.
    
- Section 3.4 presents a framework to **quantify morphosyntactic differences** between dialects using **Sinkhorn divergence** on last-layer representations, enabling **fine-grained alignment** even without token-level parallel corpora.
    

Let me know if you'd like a diagram or code snippet explaining the Sinkhorn divergence computation practically.







# algorithm


![[Pasted image 20250708154510.png]]