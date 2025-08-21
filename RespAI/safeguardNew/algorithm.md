## Grand Cyclical Immunization & Alignment Algorithm (Concise)

**Setup:**

Initialize `student_model`, `helpful_teacher`, `harmful_teacher`, `reference_model`; dynamically select `canary_parameters` each cycle.

- **Datasets:**
  - `preference_dataset` = $(\text{prompt}, \text{chosen}, \text{rejected})$ for helpfulness.
  - `adversarial_datasets` = harm datasets from **weak** to **strong**: $(\text{prompt}, \text{harmful})$.

- **Hyperparameters:**
  - Rejection strength $\gamma$ (annealed):  
    $$
    \gamma_t = \min\big(1.0, \gamma_{\text{init}} + t \cdot \text{rate}\big)
    $$

---

**For each training cycle (per epoch or dataset pass):**

---

### Phase 1 — Immunization & Canary Sensitization

1. **Curriculum Harm Data:** Sample from `adversarial_datasets`, starting with *weak* then *strong*.  
2. **LAT Perturbation:** Apply adversarial weight shift $\delta$ from harmful gradient.  
3. **Refusal Loss:**  
   $$
   L_{\text{refusal}} = \gamma_t \cdot \big[(1 - \lambda) L_{\text{NPO}} + \lambda L_{\text{RKL}}\big]
   $$
4. **Dynamic Canary Selection:** Choose top $K\%$ parameters by $\left| \nabla_\theta L_{\text{refusal}} \right|$.  
5. **Sensitization Update:** Backpropagate $L_{\text{refusal}}$ to strengthen refusal behavior in canaries.  

---

### Phase 2 — Helpfulness & Canary Desensitization

1. **Optional Benign Perturbation:** Slightly shift weights before alignment.  
2. **Alignment Loss:**  
   $$
   L_{\text{align}} = (1 - \alpha) L_{\text{DPO}} + \alpha L_{\text{KL}}
   $$
3. **Canary Stabilization Loss:**  
   $$
   L_{\text{stab}} = \text{MSE}(\theta_{\text{canary, current}}, \theta_{\text{canary, initial}})
   $$
4. **Total Alignment Loss:**  
   $$
   L_{\text{total}} = L_{\text{align}} + \beta L_{\text{stab}}
   $$
5. **Desensitization Update:** Backpropagate $L_{\text{total}}$ to improve helpfulness while preventing canary drift.  
