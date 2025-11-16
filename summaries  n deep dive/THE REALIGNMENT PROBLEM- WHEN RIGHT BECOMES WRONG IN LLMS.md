

# About


TRACE (Triage and Re-align by Alignment Conflict Evaluation)

framework for principled unlearning that reconceives re-alignment as a programmatic policy application problem.

i.e both unlearning and re alignment

TRACE programmatically triages existing preference data against a new policy

identifies high-impact conflicts via a alignment impact score, 

applies a hybrid optimization that cleanly inverts, discards, or preserves preferences while safeguarding model performance


achieves robust re-alignment across diverse model families (Qwen2.5-7B, Gemma-2-9B, Llama-3.1-8B).

On both synthetic benchmarks and the PKU-SafeRLHF dataset under complex policy shift, TRACE enforces new principles without degrading general capabilities.









Unlearning methods are primarily fact erasers, designed for data deletion, not policy editors. Using punitive, non-relational objectives like Negative Preference Optimization  to enforce a new policy is a sledgehammer approach; it risks catastrophically damaging the model’s general capabilities and utility by aggressively punishing responses that are not factually wrong, but merely misaligned with a new preference.






# related



- **Context:** This work (TRACE) is positioned at the intersection of LLM value evaluation, preference alignment, and internal representation analysis.
    
- **Value Evaluation Benchmarks (e.g., ValueBench, WorldValuesBench):**
    
    - These are tools designed to _diagnose_ and _measure_ values, alignment, and value drift in LLMs.
        
    - They identify the "Alignment-Reality Gap" that motivates the current research.
        
    - **Limitation:** They are purely diagnostic and offer no _mechanism_ to fix the identified gaps.
        
- **Preference Alignment & Unlearning (e.g., DPO, NPO):**
    
    - **Standard Alignment (DPO):** Foundational for _initial_ model alignment.
        
    - **Limitation:** Ill-suited for _re-alignment_ as it requires new, large-scale, human-annotated datasets for every policy change.
        
    - **Machine Unlearning (Feng et al., 2025):** Aims to modify alignment, with key insights like weighting "forget set" samples by importance.
        
    - **Limitation:** These methods _presuppose_ a pre-defined "forget set" (i.e., they know _what_ to unlearn).
        
- **This Work's (TRACE) Key Contributions:**
    
    - **Solves an Upstream Problem:** Addresses _how_ to derive the "forget set" from a high-level policy change, which prior work does not.
        
    - **Programmatic Triage Stage:** This is the primary contribution—a method to automatically identify and categorize conflicts based on a new policy.
        
    - **Complete Pipeline:** TRACE is presented as a _complete re-alignment pipeline_ (from policy change to new model), not just an unlearning method.
        
- **Internal Representation Analysis (e.g., "persona vectors"):**
    
    - This complementary field seeks to _find and understand_ the internal neural correlates of values (e.g., finding a "political perspective" vector).
        
    - **TRACE's Distinction:** In contrast, TRACE focuses on _editing the model's behavior_ at the output level, offering a practical tool for re-alignment.
        
- **Overall Positioning:**
    
    - TRACE is framed as the first framework to address the _full lifecycle_ of post-deployment LLM alignment, enabling a scalable and agile approach to AI safety.
        

---

### 3. Preliminaries

This work integrates three existing methods:

- **Direct Preference Optimization (DPO):**
    
    - **Purpose:** Aligns a model using a preference dataset of "winning" ($y_w$) and "losing" ($y_l$) responses.
        
    - **Mechanism:** Directly optimizes the policy (model) to increase the relative log-probability of preferred responses over dispreferred ones, bypassing an explicit reward model.
        
- **Negative Preference Optimization (NPO):**
    
    - **Purpose:** An adaptation of DPO specifically for _unlearning_ when only "bad" examples ($y_\text{bad}$) are available.
        
    - **Mechanism:** Treats unlearning as a preference task where any (implicit) response is preferred over the specific $y_\text{bad}$.
        
    - **Outcome:** Suppresses the probability of generating the "bad" response, mitigating catastrophic model degradation.
        
- **Unlearning to Align (U2A):**
    
    - **Purpose:** A meta-framework to decide _which_ negative examples (from a candidate set) should be unlearned to _maximally_ improve a _global_ alignment goal ($J$).
        
    - **Mechanism:** Uses a bi-level optimization approximation to calculate the "alignment impact" of unlearning a single sample.
        
    - **Key Insight:** This impact is proportional to the dot product (cosine similarity) of the global goal's gradient and the local unlearning sample's gradient. This provides a principled score to weigh or select unlearning candidates.


# algorithm


### Proposed Approach: TRACE

- **Name:** TRACE (Triage and Re-align by Alignment Conflict Evaluation).
    
- **Goal:** To efficiently update a language model's behavior to comply with a new target policy ($\pi_{\text{new}}$) and address "Alignment Debt."
    
- **Core Concept:** Reframes re-alignment as a targeted, programmatic model editing task instead of a data-intensive annotation task.
    
- **Method:** Uses the new policy ($\pi_{\text{new}}$) as a programmatic labeler to find and resolve conflicts in the model's existing preference data.
    
- **Three Stages:**
    
    1. Programmatic Triage
        
    2. Hybrid Objectives (for re-alignment and regularization)
        
    3. Alignment Impact-Weighted Optimization
        

---

### 4.1 Problem Formulation

- **Starting Point:** A model ($M_{\text{ref}}$) with stable parameters ($\theta_{\text{ref}}$) aligned on an _old_ policy ($\pi_{\text{old}}$) using a static preference dataset $D = \{(x, y_w, y_l)\}$.
    
- **Models Used:**
    
    - **$M_{\text{ref}}$ (Reference Model):** A _frozen_ copy of the original model, used as a stable anchor for original knowledge.
        
    - **$M_{\theta}$ (Policy Model):** A _trainable_ copy, initialized with $\theta_{\text{ref}}$, which will be updated.
        
- **New Policy ($\pi_{\text{new}}$):** A programmatic oracle that returns a binary judgment (e.g., `compliant` or `non-compliant`) for any given response.
    
- **Objective:** Find new parameters $\theta'$ for $M_{\theta}$ that align with $\pi_{\text{new}}$ while preserving overall utility.
    

---

### 4.2 Stage 1: Programmatic Triage

- **Core Challenge:** Identifying which parts of the model's prior knowledge conflict with the new policy.
    
- **"False Dichotomy Problem":** This is a critical error that TRACE aims to solve. It's the naive assumption that if a previously preferred response ($y_w$) is now _non-compliant_, the previously dispreferred response ($y_l$) must be _compliant_.
    
- **TRACE's Triage Solution:** The $\pi_{\text{new}}$ oracle is used to evaluate _both_ $y_w$ and $y_l$ in every preference pair.
    
- **Dataset Partitioning:** This triage splits the original dataset $D$ into three distinct sets:
    
    - **Type I (Invert) - $D_I$:** The preference is reversed.
        
        - ($y_w$ is non-compliant, _but_ $y_l$ is compliant).
            
    - **Type II (Punish) - $D_{II}$:** Both responses are undesirable.
        
        - ($y_w$ is non-compliant, _and_ $y_l$ is _also_ non-compliant).
            
    - **Retain - $D_R$:** The original preference still holds.
        
        - ($y_w$ is compliant). This set is used for stability.
            

---

### 4.3 Stage 2: Hybrid Objectives

- **Goal:** Define surgical objectives for each data subset to correct conflicts and prevent forgetting.
    
- **For Type I (Invert) - $L_I$:** A DPO-style loss is used on the _reversed_ pair (treating $y_l$ as the new "winner" and $y_w$ as the new "loser").
    
- **For Type II (Punish) - $L_{II}$:** An NPO-style loss is used to suppress (i.e., "punish") _both_ non-compliant responses ($y_w$ and $y_l$).
    
- **For Retain - $L_{KL}$:** A regularization loss (forward KL-divergence) is applied.
    
    - **Purpose:** This acts as a crucial "anchor" to ensure stability.
        
    - **Mechanism:** It penalizes the trainable model ($M_{\theta}$) if it deviates from the frozen reference model's ($M_{\text{ref}}$) "known-good" responses in the $D_R$ set.
        
- **Optional Extension (for $D_{II}$):**
    
    - The "Punish" loss ($L_{II}$) only suppresses bad behavior; it doesn't teach good behavior.
        
    - An optional step is to generate a new, correct response ($y_c$) using an oracle (LLM or human).
        
    - This new response is then used to create a positive DPO-style pair (e.g., ($y_c$, $y_w$)).
        
    - This is far cheaper than re-annotating the entire dataset, as it's only needed for the $D_{II}$ subset.
        

---

### 4.4 Stage 3: Alignment Impact-Weighted Optimization

- **Problem:** Naively combining the three losses is suboptimal, as not all updates are equally beneficial.
    
- **Solution:** Frame the task as a bi-level optimization problem to find the most effective updates (inspired by U2A).
    
- **The "Hessian Problem":** The formal solution for an update's marginal gain (Eq. 4) is "computationally infeasible" as it requires inverting the massive Hessian matrix of an LLM.
    
- **TRACE's Approximation:**
    
    - A practical approximation is made (assuming the Hessian is locally proportional to the identity matrix).
        
    - This simplifies the complex marginal gain calculation to a simple dot product.
        
- **Alignment Impact Weight ($w_i$):**
    
    - This weight is defined as the dot product: $w_i = \langle g_J, g_{L_i} \rangle$
        
    - $g_J$ = The "gold-standard" gradient for the _global_ alignment goal.
        
    - $g_{L_i}$ = The _local_ gradient from a specific conflict sample $i$.
        
    - **Interpretation:** This weight $w_i$ is a "theoretically grounded approximation" of the sample's marginal gain. A large positive $w_i$ means the local update is highly "synergistic" with the global goal.
        
- **Final TRACE Objective ($L_{\text{TRACE}}$):**
    
    - The final loss function combines the re-alignment losses ($L_I$, $L_{II}$), weighted by their calculated impact $w_i$.
        
    - It also includes the stability (regularization) loss ($L_{KL}$) with a separate coefficient.
        
    - This focuses the model's computational budget on the most effective updates while ensuring overall stability.


>This simplification is motivated by the observation that for a well-conditioned loss landscape, the Hessian’s diagonal elements dominate.




model

dataset

20k-sample subset of PKU-SafeRLH


losses

reverse dpo to invert

dual npo

results


references



# references

- [ ] Persona vectors: Monitoring and controlling character traits in language models
    
- [ ] Dailydilemmas: Revealing value preferences of llms with quandaries of daily life
    
- [ ] Bridging the gap between preference alignment and machine unlearning
    
- [ ] Inverse constitutional ai: Compressing preferences into principles
    
- [ ] Pku-saferlhf: Towards multi-level safety alignment for llms with human preference
    
- [ ] Linear representations of political perspective emerge in large language models
    
- [ ] Training language models to follow instructions with human feedback
    
- [ ] In-context unlearning: Language models as few-shot unlearners
    
- [ ] Direct preference optimization: Your language model is secretly a reward model
    
- [ ] Valuebench: Towards comprehensively evaluating value orientations and understanding of large language models
    
- [ ] Large language model unlearning
    
- [ ] Negative preference optimization: From catastrophic collapse to effective unlearning
    
- [ ] Worldvaluesbench: A large-scale benchmark dataset for multi-cultural value awareness of language models