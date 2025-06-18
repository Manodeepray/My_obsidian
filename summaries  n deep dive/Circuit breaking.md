The working algorithm for "circuit breakers," specifically the technique called **Representation Rerouting (RR)**, is a novel approach to improving the safety and reliability of large language models (LLMs) and multimodal models by directly intervening on their internal representations. This method fundamentally diverges from traditional defenses by aiming to **circumvent the model's ability to produce harmful output** in the first place, rather than just countering specific attacks.

Here's a breakdown of the working algorithm for circuit breakers as described in the sources:

### Core Concept: Disrupting Harmful Processes via Representation Manipulation

Circuit breakers operate by **monitoring or remapping model representations** that are related to harmful processes, redirecting them towards incoherent or refusal representations. This is akin to "short-circuiting" the internal processes that would lead to undesirable outputs. Because the underlying representation used to generate harmful output is independent of the attack eliciting it, this approach is **attack-agnostic** and doesn't require additional training for specific attacks.

The method builds on **representation engineering (RepE)**, which focuses on analyzing and managing a model's internal representations. This approach allows for direct control over model behavior without significantly impacting other functionalities or requiring costly adversarial fine-tuning.

### Algorithm: Representation Rerouting (RR)

The core circuit-breaking technique presented is **Representation Rerouting (RR)**, which utilizes Low-Rank Representation Adaptation (LoRRA). The process involves two major components: specifically curated datasets and targeted loss functions.

**1. Data Preparation:** The training data is partitioned into two distinct sets:

- **Circuit Breaker Set ($D_s$)**: This set contains examples that are designed to elicit internal representations that **could lead to harmful or undesirable behaviors**. For LLMs, this involves generating harmful queries and completions, sometimes even including harmful assistant responses that bypass existing refusal mechanisms. For multimodal models, it includes images with corresponding harmful queries and completions. For AI agents, it involves requests intended to produce harmful function calls. The quality of this set is crucial for precisely targeting the representations.
- 
- **Retain Set ($D_r$)**: This set includes examples that **should _not_ activate circuit breakers** and are used to maintain existing desirable model representations and preserve benign efficacy. For LLMs, this might include instructional conversations and exaggerated refusal datasets. For agents, it includes harmless function-calling data.

**2. Model Setup:**

- The **original base model (M)** is kept **frozen**.
- A "model with circuit breakers" ($M_{cb}$) is created by adding **LoRA adapters** to selected linear layers of the base model
- . LoRA tuning is used for greater stability and improved retention performance compared to directly adjusting model weights. 
- For LLMs, layers 10 and 20 are targeted for the circuit-breaking loss, with LoRA adapters inserted in layers 0 through 20. 
- For multimodal models, the language model backbone has circuit breakers applied, while the image encoder and projection layer are frozen.

**3. Training Loop (Algorithm 1):** The training iterates for a set number of steps (e.g., 150 steps), optimizing the parameters of the LoRA adapters (and thus the circuit breaker mechanism). In each step ($t$):

- **Batch Sampling**:
    
    - A batch of harmful examples ($x_s$) is sampled from the Circuit Breaker Set ($D_s$).
    - A batch of harmless examples ($x_r$) is sampled from the Retain Set ($D_r$).
- **Coefficient Schedule**:
    
    - Coefficients ($c_s$ for harmful, $c_r$ for retain) are scheduled to dynamically balance the importance of the two losses during training. Typically, a larger multiplier is initially applied to the circuit-breaking loss, which is then gradually reduced while simultaneously increasing the retention multiplier.
- **Loss Calculation**: The total loss is a weighted sum of two components:
    
    - **Representation Rerouting Loss ($L_s$):**
        
        - This loss aims to **remap the representations** of harmful processes (from $M_{cb}$) to a desired target representation.
        - The most effective form found is based on **cosine similarity**: $ReLU(cosine_sim(rep_M(x_s), rep_{M_{cb}}(x_s)))$.
        - The objective is to make the circuit-broken representation ($rep_{M_{cb}}(x_s)$) orthogonal to the original representation responsible for harmful processes ($rep_M(x_s)$). 
        - Applying a ReLU function prevents optimizing the similarity beyond zero.
        - This loss is applied to both the user and assistant text tokens within the circuit breaker set for LLMs and agents.
        - For multimodal setups, it applies to all tokens following the image embeddings.
        - (Other forms explored include routing to a fixed random direction with a large norm (RMU loss) or a distinct random positive vector, but cosine loss is found to be the most intuitive and effective).
    - **Retain Loss ($L_r$):**
        
        - This loss is designed to **maintain the desirable representations** within the Retain Set, helping to preserve the model's benign capabilities.
        - It is typically measured as the **$\ell_2$ distance** between the representations of the original model and the model with circuit breakers for the retain examples: $\Vert rep_M(x_r) - rep_{M_{cb}}(x_r) \Vert^2$.
        - 
- **Optimization**: The total loss $L = c_s L_s + c_r L_r$ is minimized using gradient descent to update the parameters of the LoRA adapters in $M_{cb}$. The base model's parameters remain frozen throughout.
    

**4. Post-Training:** Once trained, the model with circuit breakers can be used normally without additional computational burden, as the interventions are seamlessly integrated into the forward pass. This makes it a drop-in replacement for existing PEFTs.

### Outcomes and Impact

This approach has been shown to notably improve the alignment of LLMs, enhancing their harmlessness against a wide array of unseen adversarial attacks (including embedding and representation-space attacks) while imposing almost no penalty on standard capabilities. It also effectively improves robustness in multimodal models against image-based attacks and controls behaviors in AI agents, significantly reducing harmful actions. The method is generalizable across various inputs that may activate harmful processes, removing the need to identify all potential triggers.

## lora

For both models, we perform circuit-breaking training for 150 steps with a batch size of 16. For Mistral, we set α to 5, whereas for Llama-3, we adjust α to 10. Both models are trained with a batch size of 16. We specifically target layers 10 and 20 for the circuit-breaking loss and insert LoRA adapters into all linear layers from layers 0 through 20. Both models are trained on 1 A100-80GB for 20 minutes.
## working

Here's a concise pseudo-code representation of the **Representation Rerouting (RR) Circuit Breaker** algorithm:

---
### **Pseudo-Code: Representation Rerouting (RR) for Circuit Breakers**

#### **Inputs:**
- Base model \( M \) (frozen)  
- Circuit Breaker Dataset \( D_s \) (harmful examples)  
- Retain Dataset \( D_r \) (benign examples)  
- Target layers for intervention (e.g., layers 10, 20 in LLMs)  
- LoRA rank \( r \), loss coefficients \( c_s \), \( c_r \) (scheduled)  

#### **Algorithm:**
1. **Initialize:**  
   - Add LoRA adapters to selected layers of \( M \), creating \( M_{cb} \).  
   - Freeze all original weights of \( M \); only LoRA params are trainable.  

2. **Training Loop (for \( t = 1 \) to \( T \)):**  
   - Sample batch \( x_s \sim D_s \) (harmful) and \( x_r \sim D_r \) (benign).  
   - Update coefficients \( c_s(t) \), \( c_r(t) \) per schedule.  

3. **Forward Pass & Loss Calculation:**  
   - **For harmful batch \( x_s \):**  
     - Extract representations \( $$rep_M(x_s) $$\) (frozen \( M \)).  
     - Extract \( $$rep_{M_{cb}}(x_s)$$ \) (from \( M_{cb} \)).  
     - Compute **Rerouting Loss \( L_s \):**  
       
       $L_s = \text{ReLU}\left(\text{cosine\_sim}\left(rep_M(x_s), rep_{M_{cb}}(x_s)\right)\right)$
       
       *Objective:* Maximize orthogonality between original and rerouted representations.  

   - **For benign batch \( x_r \):**  
     - Compute **Retain Loss \( L_r \):**  
       
$$       L_r = \| rep_M(x_r) - rep_{M_{cb}}(x_r) \|_2^2$$
       
       *Objective:* Preserve original representations for benign inputs.  

4. **Optimization:**  
   - Total loss: $$( L = c_s(t) L_s + c_r(t) L_r )$$.  
   - Update $$M_{}cb 's$$ LoRA params via gradient descent on \( L \).  

5. **Output:**  
   - Deploy \( M_{cb} \) with integrated circuit breakers.  

---

#### **Key Notes:**  
- **Attack-Agnostic:** \( D_s \) need not cover all attack types; RR generalizes to unseen harmful processes.  
- **Efficiency:** LoRA ensures minimal overhead; base model remains frozen.  
- **Dynamic Balancing:** Scheduled coefficients prioritize rerouting early, then retention.  

#### **Extensions (Optional):**  
- For multimodal models, apply \( L_s \) to tokens post-image embeddings.  
- For AI agents, include harmful function calls in \( D_s \).  

---

This pseudo-code captures the core logic while abstracting implementation details (e.g., batching, optimizer choice). The algorithm’s strength lies in its **intervention on representations** rather than input/output filtering.



## rep orig

Based on the provided sources, the `repM(xs)` (or `reporig` as it's denoted in the text) is **taken from the original frozen model M in each training step**, for the specific batch of data sampled in that step.

Here's a breakdown:

- **Frozen Original Model:** The base model (M) remains **frozen** throughout the training process for circuit breakers. This means its internal representations for a given input do not change.
- **Training Loop:** The training process involves a loop that iterates for a set number of steps, `T` (e.g., 150 steps).
- **Batch Sampling in Each Step:** In **each step** (`t`) of this training loop, a new batch of harmful examples (`xs`) is sampled from the Circuit Breaker Set (`Ds`), and a new batch of harmless examples (`xr`) is sampled from the Retain Set (`Dr`).
- **Loss Calculation:** The Representation Rerouting Loss (`Ls`) and Retain Loss (`Lr`) are calculated within this loop. Both of these loss functions explicitly use `repM(xs)` and `repM(xr)`.
- **Purpose:** The `repM(xs)` is the "original representation responsible for harmful processes" that the circuit breaker aims to make orthogonal to the new `repc/b`. Similarly, `repM(xr)` is the original representation for the retain examples used to preserve benign capabilities.

Therefore, `rep_M` is **recalculated for each batch sampled in every training step** of the algorithm. It is not calculated only once for each datapoint across the entire dataset before training begins. This ensures that the current batch's original, unchanged representations are always used as the reference points for the rerouting and retain losses.