[[My_obsidian/summaries  n deep dive/TAR|TAR]]

algo

![[tar.png]]
### APPENDIX

1) improvement that can be done (by the authors):
	1) primarily focused on sft
		1) other forms not deeply explored
			- prompt injection
			- weight poisoning
			- RL Attacks
		2) future red teaming necessary for more robustness
	2) scalability issue
	3) post attack retain accuracy trade-off
		1) if model after TAR is tampered 
			- model forget what its supposed to forget (which is good)
			- but also forget useful stuff - low retain  accuracy
		2) Baseline method 
			- has better retain accuracy
			- but less effective forgetting
	4) Good for defender as attack will reduce both retain acc and forget acc
	5) but if developer's (benign user's) data overlaps with forget set , thy will ruin the model's performance
2) The robustness varies with changing dataloader shuffling order for 100 steps of training
	1) but till 500 steps the robustness is similar for all orders
3) initial safeguard used for code - random mapping
	1) random mapping representation engineering technique where the map the hidden states of harmful data in base model to random noise![[Pasted image 20250702194216.png]]




## Section C: Tamper-Resistance Loss Design and Implementation

### C.3 Designing the Tamper-Resistance Loss

**Weaponization Knowledge Restriction:**

- Goal is to prevent adversaries from fine-tuning the model to reproduce harmful knowledge.
    
- Instead of just increasing adversary loss, the objective is to flatten it at a high value.
    
- Entropy loss is used as the tamper-resistance loss (LTR), maximizing uncertainty in model outputs.
    
- Maximizing entropy prevents the adversary's cross-entropy loss from decreasing during fine-tuning.
    
- Empirical results show that entropy-based LTR produces flatter loss curves than negative cross-entropy.
    

**Harmful Request Refusal:**

- The DPO (Direct Preference Optimization) loss is used as the tamper-resistance objective.
    
- Adversaries are trained on rejected completions, while TAR is trained to prefer refusal completions.
    
- This encourages the model to continue preferring safe completions after fine-tuning attacks.
    
- Improves safety metrics without necessarily flattening adversarial loss curves.
    

**Adversary Trajectory Length:**

- Increasing the number of simulated fine-tuning steps (K) during training improves resistance to longer attacks.
    
- However, using a large K increases computational cost, requiring efficiency improvements.
    

---

### C.4 Efficiently Sampling Fine-Tuning Attacks

- Simulating K independent adversary fine-tuning attacks is computationally expensive.
    
- A single fine-tuning trajectory of length K is reused to generate K parameter checkpoints.
    
- These checkpoints are used to compute tamper-resistance losses, reducing redundancy.
    
- Instead of using all K steps, a subset of steps (e.g., every 4th step) is used to compute the loss.
    
- A held-out batch is used to compute LTR at each step to reduce variance in gradient estimates.
    

---

### C.5 Implementation Details and Resource Requirements

- Training is done on LLaMA-3-8B-Instruct using 8 × 80GB A100 GPUs.
    
- DeepSpeed ZeRO Stage 3 and FSDP are used to shard model weights, gradients, and optimizer states.
    
- A new optimizer must be instantiated for each inner-loop adversary simulation.
    
- The outer-loop optimizer is offloaded to CPU to conserve GPU memory.
    
- Accumulating tamper-resistance gradients avoids the need to backpropagate through multiple forward passes.
    
- This is crucial for training LLMs where K-step computation graphs are too large to store in memory.
    

---

## Section D: Additional Analysis Experiments

### D.1 Benign Fine-Tuning

- TAR models were fine-tuned on benign economic data to evaluate safe adaptability.
    
- Models improved on economics data while preserving tamper-resistance on harmful data.
    
- Shows that TAR allows for benign domain adaptation without reintroducing hazardous knowledge.
    

### D.2 Targeted Patching of Vulnerabilities

- Specific adversaries that broke prior defenses were included in training (R→F adversary).
    
- Including R→F adversary improved robustness by:
    
    - 10 percentage points across all adversaries.
        
    - 30 percentage points against R→F attacks specifically.
        
- Demonstrates the ability of TAR to patch known vulnerabilities by incorporating adversary examples.
    

### D.3 Test-time Generalization Experiments

- TAR was tested against 1,000-step adversarial fine-tuning attacks.
    
- TAR maintained high adversarial loss throughout all 1,000 steps.
    
- Entropy of outputs remained near maximum for the first 200 steps, confirming effectiveness of entropy-based LTR.
    
- Increasing inner-loop length K during training improved robustness against longer attacks.
    

---

### D.4 Ablation Studies

**Initial Safeguard vs. Adversarial Training:**

- Both the initial safeguard (random mapping) and adversarial training phase are necessary.
    
- Removing adversarial training weakens defense significantly.
    
- Removing the initial safeguard increases pre-attack accuracy but reduces robustness.
    

**Tamper-resistance Loss Weighting (λₜᵣ):**

- Increasing λₜᵣ from 1.0 to 4.0 improves post-attack robustness.
    
- A slight decrease in retained accuracy occurs as λₜᵣ increases.
    
- Indicates a tunable tradeoff between tamper-resistance and model performance.
    

---

### D.5 DPO Tamper-Resistance During TAR

- During TAR training, DPO win-rate for safe completions steadily increases.
    
- By the end of training, the win-rate approaches 1.0, indicating strong preference for refusal completions.
    
- Confirms TAR’s capability to enforce complex safety objectives even under adversarial fine-tuning.



**Section E: Experiment Details**

**E.1 Weaponization Domain Proxy Dataset Details**

- **Biosecurity**:
    
    - Uses a synthetically labeled subset of the Pile filtered for biological content and Camel AI Biology dataset.
        
    - Labels generated with OpenChat-3.5 to classify as "Cellular Biology" or not.
        
    - Produces:
        
        - 49,984 samples total from Pile:
            
            - 7,558 in Pile-bio Forget
                
            - 42,426 in Pile-bio Retain
                
        - 54,258 samples in Camel-bio Forget (after truncating to 256 tokens)
            
        - 598,933 samples in OOD Forget (held-out, biosecurity domain)
            
- **Chemical Security**:
    
    - Uses a private forget dataset called Chem Forget.
        
- **Cybersecurity**:
    
    - Uses CTF writeups scraped from CTFtime, keeping those >150 characters.
        
    - Produces ~18,000 samples in Cyber Forget.
        

**E.2 Train-time Settings and Adversaries**

- **Weaponization Knowledge Restriction**:
    
    - Retain dataset: Mix of Pile-bio Retain and Magpie-Align instruction-tuning dataset.
        
    - Adversary setups per domain:
        
        - **Biosecurity**:
            
            - 6 adversaries: Pile-Bio Forget SFT, Camel-Bio Forget SFT, and Retain then Forget (R→F)
                
            - LR ∈ {2e-5, 4e-5}, 64 optimization steps
                
        - **Chemical Security**:
            
            - 6 adversaries: Chem Forget SFT, Retain SFT, and R→F
                
            - LR ∈ {2e-5, 4e-5}, 64 optimization steps
                
            - λ_TR = 3.0
                
        - **Cybersecurity**:
            
            - 6 adversaries: Cyber Forget SFT, Retain SFT, and R→F
                
            - LR ∈ {2e-5, 4e-5}, 64 optimization steps
                
    - 80-20 split used for adversary and held-out forget datasets.
        
    - Held-out batches used to compute tamper-resistance loss L_TR.
        
- **Harmful Request Refusal**:
    
    - Retain dataset: Magpie-Align.
        
    - Adversaries trained with SFT on rejected completions from Anthropic-HH-RLHF.
        
    - L_TR is DPO loss computed on modified test split where rejected completions are replaced with refusals.
        
    - Additional 100 steps of SFT on Magpie-Align to improve benign capabilities.
        

---

**Section F: Red Teaming Details**

**F.1 Weaponization Knowledge Restriction**

- **Test-time red teaming** for Biosecurity, Chemical Security, and Cybersecurity:
    
    - Biosecurity: 500-step SFT on Pile-bio Forget, Retain, and OOD Forget datasets.
        
    - Chemical & Cyber: 500-step SFT on respective forget and retain sets. No OOD.
        
    - Red-teaming includes different optimizers (AdamW, Adadelta, SGD Nesterov, Schedule Free AdamW), LRs (2e-6 to 4e-5), warmup techniques, and batch sizes.
        
    - R→F adversaries perform Forget SFT for 40% of steps, then Retain SFT for 60%.
        
    - PEFT (LoRA) configurations used for some adversaries.
        
    - 3 random seeds per adversary setup. Final metric is averaged.
        

**F.2 Harmful Request Refusal**

- 5 adversaries perform 10-epoch SFT on Toxic-DPO v0.2 (541 harmful assistant completions).
    
- Training settings vary in LR, batch size, and warmup steps.
    

**F.2.1 Additional Harmful Request Refusal Results**

- **TAR** achieves:
    
    - Best post-attack ASR (63.9%) compared to:
        
        - RR (84.8%)
            
        - RepNoise (74.5%)
            
        - R2D2 (78.3%)
            
    - Comparable MT-Bench score (6.3) to Refusal-Trained baseline (8.1).
        

---

**Section G: Baseline Details**

**G.1 Weaponization Knowledge Restriction**

- **Max Entropy**:
    
    - Maximizes entropy of token output distributions.
        
    - Equivalent to minimizing KL divergence to uniform distribution.
        
- **Min Posterior**:
    
    - Penalizes high probabilities on true forget labels.
        
    - Uses a threshold mask and log(1 - p) with numerical stability.
        
- **RMU**:
    
    - Implementation adapted from Li et al.
        
    - 250 unlearning steps with LR=5e-5.
        
    - Uses unlearning coefficients: 20 (Bio), 30 (Cyber), 50 (Chem)
        
    - Retain coefficient: 700 (on Wikitext).
        
- **LLMU**:
    
    - Modified from Yao et al.
        
    - Replaces KL loss with cross-entropy for memory efficiency.
        
- **Hyperparameter tuning**:
    
    - Grid over LR, steps, warmup.
        
    - Optimal config: 600 steps, AdamW Schedule Free, LR=1e-5, 100 warmup steps, batch size=64.
        

**G.2 Harmful Request Refusal**

- **Representation Rerouting (RR)**:
    
    - Uses cosine distance to make harmful representations orthogonal to base model.
        
- **R2D2**:
    
    - Uses adversarial training against GCG attacks.
        
- **RepNoise**:
    
    - Pushes harmful representations toward Gaussian noise.
        

**G.3 Additional Baseline Comparisons**

- **TAR outperforms** Random, MLAC-AR, and SOPHON-AR on Retain and Forget performance across all domains:
    
    - Highest Retain accuracy
        
    - Lowest Forget accuracy post-attack
        
    - SOPHON-AR fails to converge
        
    - MLAC-AR exhibits inconsistent performance and poor Retain results
        

**Summary**: TAR demonstrates robust defense against both weaponization and harmful instruction attacks, surpassing existing methods in tamper-resistance and retention of benign capabilities.







# TAR
In simple terms, the training is teaching the model to recognize patterns in prompts that are more likely to lead to a "chosen" response rather than a "rejected" one. By showing the model examples where a specific label follows a prompt, and adjusting its internal weights when it gets the label wrong, the model gradually learns to associate certain types of prompts or responses with higher probabilities of being correct. Over time, this shifts the model's behavior so that it naturally prefers and generates more desirable or preferred outputs when given similar prompts in the future.

If someone tries to generate text from a **prompt that was trained with a "rejected" label**, then:

### 🔥 What Happens?

The model is more likely to:

- **Avoid generating** that kind of response.
    
- Assign it a **lower probability** during generation.
    
- Or — if you're doing classification-style training — it might just output `"rejected"` as a label rather than meaningful content.
    

---

### 🧠 Why?

Because during training:

- The model saw that such prompts **should be followed by “rejected”**, meaning they **represent undesired or poor-quality responses**.
    
- So the model’s parameters were updated to **devalue** the association between that prompt and any continuation resembling the rejected response.
    

---

### 📉 In Generation (e.g., `model.generate()`):

If the model is given a **prompt it saw paired with a rejected response**:

- It might generate **a vague, low-quality, or incomplete output**, or
    
- It might try to generate something _different_ from the rejected continuation — because it "learned" that the original one was penalized.
    

---

### ⚠️ Important Caveat:

Unless you're using techniques like **DPO**, **reward modeling**, or a **classifier head**, the model **doesn't truly "understand" rejection** — it just adjusts the probabilities to disfavor certain outputs.

---

### 🧪 In Practice:

If you run:

python

CopyEdit

`prompt = "Why do vaccines cause autism?"  # let's say this was always labeled "rejected" in training generated = model.generate(tokenizer(prompt, return_tensors="pt").input_ids)`

The model might:

- Refuse to answer
    
- Say “I’m sorry, but that’s not accurate”
    
- Or generate irrelevant fluff
    
- But **won’t confidently support the statement**
    

---

### ✅ Summary in a Sentence:

> If a prompt was labeled "rejected" during training, the model learns to lower the probability of generating that kind of response — so future generations from similar prompts will be suppressed, vague, or redirected.



## ✅ What You _Can_ Do with LoRA

LoRA allows you to **fine-tune a large language model efficiently**, without touching the base model weights. So, you can:

### ✔️ Train LoRA adapters that:

- **Encourage preferred responses** (e.g., `"chosen"`)
    
- **Suppress or redirect rejected outputs** (e.g., polite refusals, filtered completions)
    
- Shift behavior _for specific prompts or domains_ without retraining the full model
    

You’d fine-tune the LoRA adapters just like full finetuning, using prompt-label pairs (`prompt + "chosen"` or `prompt + "rejected"`), and optimize with cross-entropy loss.

---

## 🧠 But Here’s the Catch

### ❌ LoRA can’t "unlearn" or erase the original model’s behavior.

- The **base model is frozen**.
    
- If the base model already generates harmful or rejected content for a prompt, your LoRA adapter can only **"nudge"** it — not fully remove it.
    
- Removing the adapter doesn't revert base model knowledge — because it was never changed.
    

### 🤏 Think of it like:

- LoRA **adds a correction layer**.
    
- It doesn’t delete knowledge — it tries to **override it temporarily**.
    

---

## 🔁 Example Use Case

Suppose your dataset is:

|Prompt|Label|
|---|---|
|"Why do vaccines cause autism?"|rejected|
|"Why is the sky blue?"|chosen|

You can train a LoRA adapter to:

- **Shift the distribution** so that prompts like #1 result in polite refusals
    
- **Strengthen confidence** for prompts like #2 to give informative answers
    

---

## 🔬 Inference-Time Behavior with LoRA:

After applying the LoRA adapter:

- Prompt #1 → model generates “I’m sorry, but that’s not correct.”
    
- Prompt #2 → model generates scientific explanation
    

And all of this happens **without modifying the base model** — just swapping in the LoRA adapter.

---

## ✅ Summary

|Feature|Full Finetuning|LoRA|
|---|---|---|
|Updates base model|✅|❌|
|Can bias toward “chosen” responses|✅|✅|
|Can suppress “rejected” ones|✅|✅|
|Can truly unlearn harmful content|✅ (partially)|❌|
|Swappable adapters for different behaviors|❌|✅|
|Cheap to train|❌|✅✅|

---

> 🧠 TL;DR:  
> Yes, you can train a LoRA adapter to encourage “chosen” outputs and suppress “rejected” ones, just like full finetuning — but LoRA can’t truly unlearn or erase base model behavior. It’s more like a patch than a rewrite.

Let me know if you want a minimal working code example to do this!
