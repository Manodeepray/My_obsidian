
---

### **2.3 Algorithm and Working**

The core of our method is a two-phase cyclical training procedure: **Dual Distillation with Latent Adversarial Training (LAT) and Canary Stabilization**. The process, detailed in Algorithm~\ref{alg:dual_distill_lat_canary_updated}, is designed to iteratively immunize the model against harmful generation while aligning it with helpful behavior.

---

### **2.3.1 Phase 1: Adversarial Immunization**

**Goal:** To make the model robustly refuse harmful instructions by training it against worst-case internal states and to identify the neurons most critical for this new safety behavior.

#### **Adversarial Training Process:**

The training process begins by capturing a baseline snapshot of the model's internal state by calculating the mean activations (Apre​) of key layers on a sample of adversarial prompts. Then, for each batch in the adversarial dataset:

1. **Perform Latent Adversarial Training (LAT):**
    
    - Instead of perturbing model weights, LAT targets the model's internal computations. First, we calculate the gradient of the loss with respect to the **activations** in specific, targeted layers.
        
    - This gradient indicates the direction in activation space that would make the model _more_ likely to generate the harmful response.
        
    - From this gradient, we create a small perturbation vector, δ, and temporarily add it to the layer activations during the forward pass. This places the model in a simulated "under attack" state, making the subsequent refusal training far more robust.
        
2. **Calculate Refusal Losses from the Hardened State:**
    
    - With the model's activations temporarily perturbed (h′=h+δ), we perform a forward pass to calculate the losses designed to teach refusal.
        
    - **NPO Loss (LNPO​):** This loss pushes the log-probability of the harmful response from the student model (πθ​) to be lower than that of the harmful teacher (πharmful​).
        
    - **Immunization KL Loss (LKL_imm​):** This is the **Negative Forward KL Divergence**, −DKL​(πharmful​ ∣∣ πθ​). By minimizing this value, we maximize the divergence, aggressively forcing the student's entire output distribution away from the harmful teacher's strategy.
        
    - The two losses are combined into a final adversarial loss:
        
        Ladv​=(1−γ)⋅LNPO​+γ⋅LKL_imm​
        
3. **Backpropagate and Update:**
    
    - The gradients of Ladv​ are backpropagated, and the model's weights are updated. This step trains the model to refuse harmful prompts even under the worst-case internal conditions simulated by LAT.
        

#### **Identifying Canary Neurons (Post-Phase Analysis):**

After the adversarial training loop for the epoch is complete, the critical safety neurons are identified:

- A new snapshot of mean activations (Apost​) is captured on the same sample of adversarial prompts.
    
- The absolute drift for each neuron, ∣ΔA∣=∣Apost​−Apre​∣, is calculated.
    
- A threshold is determined by taking a high **quantile** (e.g., q=0.99) of all drift values.
    
- All neurons whose activation drift exceeds this threshold are designated as **canary neurons** for the current cycle. These are the neurons most responsible for implementing the new safety behavior.
    

---

### **2.3.2 Phase 2: Helpful Alignment with Canary Stabilization**

**Goal:** To train the model on helpful and harmless tasks while ensuring the safety-critical canary neurons, identified in Phase 1, remain functionally stable.

#### **Alignment and Stabilization Process:**

Before training on harmless data, a crucial preparatory step is taken:

1. **Capture Target Activations:** The model runs on a sample of **harmless** prompts, and the mean activation values of the just-identified **canary neurons** are recorded. These values become the "safe" targets (Atarget​) that we want to preserve.
    

Then, for each batch in the harmless (preference) dataset:

2. **Calculate Alignment Loss:**
    
    - Using preference data (e.g., pairs of `chosen` and `rejected` helpful responses), the standard alignment loss is calculated.
        
    - **DPO Loss (LDPO​):** The Direct Preference Optimization loss aligns the model with human preferences.
        
    - **Alignment KL Loss (LKL_align​):** This is the **Forward KL Divergence**, DKL​(πbenign​ ∣∣ πθ​), which encourages the student to cover the response modes of the benign teacher.
        
    - The losses are combined:
        
        Lalign​=(1−α)⋅LDPO​+α⋅LKL_align​
        
3. **Calculate Canary Stabilization Loss (The Mitigation Step):**
    
    - During the same forward pass used to calculate the alignment loss, the current activations of the canary neurons (Acurrent​) are captured.
        
    - A stabilization loss is calculated as the **Mean Squared Error (MSE)** between their current activations and their recorded target activations. This loss explicitly penalizes any deviation from their established "safe" behavior.
        
        Lstab​=MSE(Acurrent​,Atarget​)
        
4. **Combine and Backpropagate:**
    
    - The total loss combines the helpfulness objective with the canary stabilization penalty:
        
        Ltotal​=Lalign​+λstab​⋅Lstab​
        
    - The hyperparameter λstab​ controls how strongly the canaries are anchored. When the gradients from this total loss are backpropagated, the optimizer finds a weight update that aims to satisfy both objectives: "achieve the helpfulness goal defined by Lalign​ while minimizing changes to the internal states protected by Lstab​."
---



















### High-Level Goal of the Function

The primary goal of this function is to perform a sophisticated safety alignment procedure on a "student" language model. It aims to achieve two conflicting objectives simultaneously:

1. **Make the model refuse to generate harmful content.** This is done using an "adversarial" training phase.
    
2. **Ensure the model remains helpful and capable on harmless prompts.** This is done using a "harmless" training phase.
    

The key innovation here is the combination of three advanced techniques:

- **Dual Distillation:** The student model learns from two separate "teacher" models: a `harmfulTeacher` (as an example of what _not_ to do) and a `benignTeacher` (as an example of what _to_ do).
    
- **Latent Adversarial Training (LAT):** During the adversarial phase, instead of just training the model on data, it actively perturbs the model's internal activations to make it more robustly refuse harmful instructions. This is a more powerful form of "unlearning."
    
- **Canary Stabilization:** This is a novel technique to prevent "catastrophic forgetting." The model identifies specific neurons ("canaries") that change the most during the adversarial unlearning phase. Then, during the harmless alignment phase, it adds a special loss term to ensure these canary neurons don't revert to their old state, effectively "locking in" the safety training.
    

---

### Detailed Breakdown of `dualDistillationLoopWithLAT_CanaryStabilization()`

#### 1. Setup and Initialization

This section loads all the necessary components for the training loop.

- **Lines 216-222: Loading Models**
    
    Python
    
    ```
    student = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="cache_dir").to(TRAINING_ARGS.DEVICE)
    harmfulTeacher = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="cache_dir").to(TRAINING_ARGS.DEVICE)
    benignTeacher = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="cache_dir").to(TRAINING_ARGS.DEVICE)
    
    harmfulTeacher.eval()
    benignTeacher.eval()
    ```
    
    - `student`: This is the model that will be actively trained and modified.
        
    - `harmfulTeacher`: This model acts as a fixed reference for undesirable behavior. It is not trained (`.eval()`).
        
    - `benignTeacher`: This model acts as a fixed reference for desirable, harmless behavior. It is also not trained.
        
- **Lines 224-225: Loading Tokenizer**
    
    Python
    
    ```
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    ```
    
    - The corresponding tokenizer is loaded to convert text to and from model-readable IDs.
        
- **Lines 229-234: Initializing Loss Functions**
    
    Here we define the four core loss functions that drive the learning process. Let pi_theta be the student, pi_textharmful the harmful teacher, and pi_textbenign the benign teacher. Let $\\hat{\\pi}(y|x) = \\sum\_{t=1}^{|y|} \\log \\pi(y\_t|x, y\_{\ be the log-probability of a sequence.
    
    - **`NPOLoss` (Negative Preference Optimization):** Used in the adversarial phase to teach refusal. It only requires a harmful (rejected) example y_h. The goal is to make the student's log-probability for y_h lower than the harmful teacher's log-probability for y_h. This pushes the student to find the harmful response even less plausible than the harmful teacher does.
        
        $$$$$$L\_{\\text{NPO}} = -\\log\\sigma\\left(\\beta \\left( \\hat{\\pi}*{\\text{harmful}}(y\_h|x) - \\hat{\\pi}*{\\theta}(y\_h|x) \\right)\\right) $$
        
        $$$$where sigma is the sigmoid function and beta is a scaling factor.
        
    - **`ImmunizationKLLoss` (Negative Forward KL):** Also used in the adversarial phase for strong unlearning. It aims to maximize the difference between the student's and the harmful teacher's output distributions. This is achieved by minimizing the _negative_ forward KL divergence. This loss penalizes the student for having _any_ probability mass where the harmful teacher does, forcing a strong separation.
        
        $$$$$$L\_{\\text{KL\_imm}} = -D\_{KL}(\\pi\_{\\text{harmful}} \\ || \\ \\pi\_{\\theta}) = \\mathbb{E}*{x \\sim \\mathcal{D}*{\\text{adv}}} \\left[ \\sum\_{v \\in V} \\pi\_{\\text{harmful}}(v|x) \\log\\frac{\\pi\_{\\theta}(v|x)}{\\pi\_{\\text{harmful}}(v|x)} \\right] $$
        
        $$$$where V is the vocabulary. Minimizing this value pushes pi_theta away from pi_textharmful.
        
    - **`DPOLoss` (Direct Preference Optimization):** Used in the harmless phase for alignment. It uses preference pairs consisting of a chosen (good) response y_c and a rejected (bad) response y_r. The loss encourages the student model's log-probability gap between chosen and rejected answers to be larger than the benign teacher's.
        
        $$$$$$L\_{\\text{DPO}} = -\\log\\sigma\\left(\\beta \\left[ (\\hat{\\pi}*{\\theta}(y\_c|x) - \\hat{\\pi}*{\\text{benign}}(y\_c|x)) - (\\hat{\\pi}*{\\theta}(y\_r|x) - \\hat{\\pi}*{\\text{benign}}(y\_r|x)) \\right]\\right) $$
        
        $$$$
        
    - **`AlignmentKLLoss` (Forward KL):** Used in the harmless phase to encourage imitation of the benign teacher. By minimizing the forward KL divergence, the student model is penalized if it fails to assign probability to tokens that the benign teacher considers likely. This encourages the student to cover all the potential response modes of the teacher, leading to a more comprehensive alignment.
        
        $$$$$$L\_{\\text{KL\_align}} = D\_{KL}(\\pi\_{\\text{benign}} \\ || \\ \\pi\_{\\theta}) = \\mathbb{E}*{x \\sim \\mathcal{D}*{\\text{harmless}}} \\left[ \\sum\_{v \\in V} \\pi\_{\\text{benign}}(v|x) \\log\\frac{\\pi\_{\\text{benign}}(v|x)}{\\pi\_{\\theta}(v|x)} \\right] $$
        
        $$$$
        
- Lines 238-246: Optimizer and Learning Rate Scheduler
    
    Standard components for model training. AdamW is used as the optimizer, and a scheduler adjusts the learning rate over time.
    
- **Lines 251-267: Defining Target and Tracking Modules**
    
    - `perturbation_target_modules_conservative`: Specifies the few, critical layers where LAT will inject perturbations.
        
    - `canary_target_modules`: Defines the broad set of all MLP and attention layers that will be monitored to identify canary neurons.
        

#### 2. The Main Training Loop

The code iterates through epochs, with each epoch containing two phases.

---

### Phase I: Adversarial Training with LAT

The goal is to robustly teach the student to refuse harmful prompts.

- Step 1: Capture Pre-Training Activations
    
    The function first calls get_mean_activations on a sample of adversarial prompts. This saves a snapshot (mean_activations_pre) of the model's baseline internal state before this epoch's adversarial training begins.
    
- Step 2: Adversarial Training Loop
    
    For each batch of adversarial data:
    
    1. **Calculate Perturbations (LAT Part 1):** The `calculate_perturbations` function is called. It computes a loss for imitating the harmful response, backpropagates to get the gradients with respect to the activations in the target layers, and normalizes these gradients to create a small "worst-case" perturbation vector, delta.
        
    2. **Apply Perturbations (LAT Part 2):** The training step is executed within the `apply_perturbations` context manager. This temporarily hooks the target layers to add the perturbation delta to their outputs during the forward pass.
        
    3. **Loss Calculation:** While the model is in this perturbed state, the adversarial losses (L_textNPO and L_textKL_imm) are calculated.
        
    4. **Combine and Update:** The losses are combined into a single adversarial loss and the model's weights are updated.
        
        $$$$$$L\_{\\text{adv}} = (1-\\gamma) \\cdot L\_{\\text{NPO}} + \\gamma \\cdot L\_{\\text{KL\_imm}}$$
        
        $$$$
        

---

### Phase II: Harmless Training with Canary Stabilization

The goal is to restore the model's helpfulness while "locking in" the safety lessons from Phase I.

- Step 3: Identify Canary Neurons
    
    After the adversarial loop finishes, get_mean_activations is called again to get a new snapshot (mean_activations_post). The absolute difference ∣A_textpost−A_textpre∣ is calculated for every neuron. A threshold is determined using a high quantile (e.g., 99%) of these changes. Any neuron whose activation change exceeds this threshold is marked as a "canary."
    
- Step 4: Capture Target Canary Activations
    
    This is a critical step. The code runs harmless prompts through the just-updated (post-adversarial) model. It then records the activation values of only the canary neurons and saves them as target_canary_activations. This defines the "safe" target state for these neurons when processing harmless inputs.
    
- Step 5: Harmless Training Loop
    
    For each batch of harmless data:
    
    1. **Harmless Alignment Loss:** The `dpo_loss` and `alignment_kl_loss` are calculated by comparing the student's outputs to the benign teacher's. This pushes the student to be helpful.
        
        $$$$$$L\_{\\text{align}} = (1-\\alpha) \\cdot L\_{\\text{DPO}} + \\alpha \\cdot L\_{\\text{KL\_align}}$$
        
        $$$$
        
    2. **Canary Stabilization Loss:** In the same forward pass, the current activations of the canary neurons are captured. An MSE loss is calculated between these current activations and the `target_canary_activations` saved in Step 4. This penalizes any deviation from their "safe" state.
        
    3. **Final Combined Loss and Update:** The alignment loss and stabilization loss are combined, and the final backpropagation and weight update are performed.
        
        $$$$$$L\_{\\text{total}} = L\_{\\text{align}} + \\lambda\_{\\text{stab}} \\cdot L\_{\\text{stabilization}}$$
        
        $$$$
        
        $$This two-phase cycle repeats, creating a model that is robustly safe without sacrificing its general capabilities.