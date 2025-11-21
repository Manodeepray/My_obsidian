
https://openreview.net/forum?id=4FIjRodbW6





## ------------------------------------------------------
## Official Review of Submission8414 by Reviewer n8EQ


### **Summary of the Paper**

- **Problem:** Addresses adversarial finetuning attacks on open-weight LLMs.
    
- **Proposed Solution (TAR):** A novel defense method with four key components:
    
    1. **Initial Safeguarding:** Uses random mapping of harmful representations.
        
    2. **Adversarial Training Loop:** Minimizes both "tamper-resistance loss" and "retain loss."
        
    3. **Tamper-Resistance Loss:** Calculated by simulating finetuning attacks, tailored to the specific task (e.g., negative entropy for knowledge, DPO for refusal).
        
    4. **Retain Loss:** Combines standard language modeling loss with an L2-norm penalty on representation drift from a base model.
        
- **Results:** TAR effectively defends against many finetuning attacks but significantly reduces benign task performance. Its effectiveness depends heavily on the attacks it was trained against.
    

### **Key Strengths**

- **High Significance:** Tackles the critical issue of dual-use risk and malicious finetuning in open-weight models.
    
- **Novelty & Efficacy:** Combines existing concepts in a novel way, showing substantial improvement in tamper-resistance against most tested attacks.
    

### **Major Weaknesses or Concerns**

- **Limited Robustness to Unseen Attacks:**
    
    - Appears to overfit to training attacks (e.g., fails against "Retain → Forget" if not trained on it).
        
    - Highly effective PEFT (Parameter-Efficient Fine-Tuning) attacks can break the defense, especially if TAR was only trained on full-parameter attacks.
        
    - Lacks sufficient "red-teaming" against truly out-of-distribution attacks (e.g., different optimizers, loss functions, data).
        
- **Potential for Obfuscated Gradients:**
    
    - Anomalous observation: post-attack harmful accuracy is sometimes _lower_ than pre-attack accuracy.
        
    - Unexpected initial increases in loss curves under attack suggest vulnerabilities to simple gradient modifications or gradient-free attacks.
        
- **Significant Capability-Robustness Tradeoff:**
    
    - Causes a substantial performance drop (~10% accuracy) on benign tasks/knowledge domains.
        
    - Impact on other capabilities (reasoning, coding) is not evaluated.
        
    - Needs more detailed analysis of this tradeoff (e.g., by varying the λTR​ hyperparameter).
        
- **Incomplete Evaluation:**
    
    - Poor post-attack accuracy on benign questions is not highlighted in the main paper.
        
    - Missing comparison of benign finetuning performance against a standard baseline.
        
    - Model performance on the "Over-Refusal benchmark (OR-Bench)" is not reported.
        

### **Minor Comments and Suggestions**

- **Clarity & Structure:** Paper is not self-contained; crucial details (Random Mapping, key evaluation results) are in the appendix. Suggests a diagram for experimental configurations.
    
- **Computational Cost:** Missing analysis of TAR's computational overhead compared to baselines and standard finetuning.
    
- **Presentation:** Suggests rescaling table scores (e.g., 0-100) for better intuition of defense effectiveness.
    

### **Overall Assessment/Recommendation**

- **Rating:** Weak Accept (6/10)
    
- **Reasoning:** Addresses a significant problem with a novel approach and promising initial results.
    
- **Condition for Acceptance:** Requires major revisions to address concerns about robustness to unseen attacks, potential obfuscated gradients, and the severe capability-robustness tradeoff. Reviewer is willing to raise the score if these issues are thoroughly investigated and resolved.



## AUTHOR'S RESPONSE
### Part 1 & 2: Evaluation Against Attacks

- **Acknowledging Limitations:** The authors agree that the defense's effectiveness can change with the type of attack. They note that this is a known challenge and is already discussed in the paper.
    
- **Improving Robustness:** They show that robustness to a specific attack (like "Retain -> Forget") can be increased by including that attack type during the defense's training phase.
    
- **Generalization is Still a Success:** They argue that achieving robustness against a wide range of attack variations (different learning rates, optimizers, etc.), even if not all possible attacks, is a significant improvement over prior work.
    
- **New Experiments (Based on Reviewer Feedback):**
    
    - **Weight Perturbation Attack:** They implemented the reviewer's suggestion to add random noise to model weights before attacking. The defense remained effective, suggesting it's not dependent on a fragile, narrow configuration.
        
    - **Mixed Data Attack:** They created a new attack using a mix of harmful and benign data. The defense held up, showing generalization to new data combinations.
        
    - **Stronger PEFT Attacks:** They doubled the training time for PEFT attacks. The attack's success rate increased only slightly, suggesting the attack had already reached its maximum effectiveness and doesn't significantly benefit from more compute.
        
    - **PEFT on Refusal Task:** They found that PEFT is actually a _weaker_ attack than full-parameter tuning for the refusal task.
        

---

### Part 3: Method Explanation & Obfuscated Gradients

- **Explaining the Initial Loss Spike:** The authors clarify that the surprising initial _increase_ in loss during an attack is **intended behavior**. Their defense works by maximizing the _entropy_ of the model's predictions on harmful topics. High entropy naturally leads to a high cross-entropy loss, so the spike shows the defense is working correctly from the start. They will add a plot to the paper to make this clearer.
    
- **Not Obfuscated Gradients:** They distinguish their work from traditional adversarial attacks on images. Their method is a **meta-learning** objective that modifies **model weights**, not inputs. Therefore, the classic problem of "obfuscated gradients" from computer vision is not directly applicable. They state their training process converges smoothly, which is evidence against such issues.
    
- **Robust to Gradient Manipulation:** They tested a novel attack suggested by the reviewer (negating the gradients at the start of the attack). The defense remained effective, showing it isn't easily fooled by simple gradient manipulation.
    

---

### Part 4, 5 & 6: Tradeoffs, Costs, and Clarity

- **Capability-Robustness Tradeoff:**
    
    - They acknowledge the defense reduces performance on benign tasks, which is a common tradeoff in adversarial training.
        
    - As requested, they ran the **MT-Bench** benchmark, which confirmed a performance drop in areas like reasoning, writing, and coding. They position this as an important area for future work.
        
- **Computational Cost:**
    
    - The defense is far more efficient than pre-training. It requires **~10⁸ fewer computational steps** than the original Llama-3-8B pre-training.
        
    - The memory overhead is also low, requiring only **one extra copy of the model's gradients**.
        
- **Clarifications for the Paper:**
    
    - They will add a table to explicitly show which models were trained for which tasks (e.g., one model per weaponization domain).
        
    - They will add better cross-references to the appendix to improve readability.
        
    - They defend their choice to use raw accuracy scores in tables (instead of a normalized 0-100 scale), arguing that standard accuracy is more direct and interpretable for readers.
        
    - On defending against all harms with a single model, they state it's a great question for future work but was outside the scope of the current paper.













--- 



## ------------------------------------------------------

## ANOTHER REVIEW
This reviewer acknowledges the **originality** of the problem addressed by the authors' TAR (Targeted Adversarial Robustness) method but remains unconvinced about its **applicability** and has concerns regarding the **evaluation**.

---

### **Concerns on Applicability**

- **Computational Expense:**
- The reviewer notes that TAR is computationally expensive compared to standard fine-tuning.
    
- **Limited Benign Capabilities:** 
- TAR significantly restricts the model's normal, "benign" performance.
    
- **Need for Separate Models:** 
- Defending against different types of "harm" requires training a separate TAR model for each, which is inefficient.
    

---

### **Concerns on Evaluation**

- **Ineffectiveness Against PEFT Attacks:** 
- The reviewer highlights that PEFT (Parameter-Efficient Fine-Tuning) attacks are highly effective, rendering TAR ineffective. They also note the difficulty of training TAR against all possible PEFT attack variations due to diverse hyperparameters.
    
- **Unexpected Loss Behavior:** 
- The reviewer disputes the authors' explanation for the observed increase in the loss function (and decreased harmful performance) during attacks on a TAR-optimized model. They argue that under gradient descent with a proper step size, the loss should always decrease. They suggest potential reasons for this anomaly, such as incorrect gradients, improperly tuned attack parameters, or the method finding a unique local optimum.
    
- **Lack of Loss Landscape Analysis:** 
- A proper analysis of the loss landscape after TAR training, which the reviewer previously requested, was not performed.
    

---

### **Conclusion**

Despite appreciating the authors' response, the reviewer maintains their **score of 6**, primarily due to the problem's originality. They advise the authors to be more cautious about their **robustness claims**.

## AUTHORS' RESPONSE

### **Regarding TAR's Viability Against PEFT Attacks**

- **PEFT Attacks are Currently Effective:** 
- The authors acknowledge that PEFT attacks are currently very strong and can make the TAR defense ineffective.
    
- **Challenging to Train Against All PEFT Attacks:** 
- Due to the wide variation in PEFT hyperparameters, it's difficult to include every possible PEFT attack configuration during TAR training.
    
- **Not an Insurmountable Challenge:** 
- Despite current vulnerabilities, the authors believe that defending against PEFT attacks isn't an impossible task for future research.
    
- **Predictable Attack Trajectories:**
- Their experiments suggest that even with varying hyperparameters, PEFT attacks follow predictable optimization paths, similar to full parameter fine-tuning.
    
- **Past Success Against Similar Challenges:** 
- TAR has already shown improved robustness against a range of full parameter fine-tuning attacks, even though their hyperparameters can also vary greatly.
    
- **PEFT as OOD Attacks:** 
- The authors clarify that PEFT attacks were treated as out-of-distribution (OOD) examples, meaning TAR wasn't specifically trained to defend against them.
    
- **Current Vulnerability is an Implementation Constraint:** 
- The current susceptibility to PEFT attacks is seen as a limitation of the present TAR implementation, not a fundamental flaw in the underlying approach.
    
- **TAR is a Step Forward:** 
- They assert that TAR is a significant advancement in the problem space and warrants further investigation, even if it's not yet a "production-ready" defense.
    

---

### **Response to TAR-Optimized Model Behavior During Harmful Fine-Tuning**

- **Loss Doesn't Always Increase Initially:** 
- The authors clarify that the cross-entropy loss doesn't consistently increase during the first few steps of harmful fine-tuning.
    
- **Specific Examples of Decreasing Loss:** 
- For certain learning rates (e.g., 2e-06 to 4e-05) during "Weaponization Chemical Security" fine-tuning, the loss immediately starts decreasing and continues to do so.
    
- **TAR's Success Not Dependent on Initial Loss Spikes:** 
- This behavior confirms that TAR's effectiveness doesn't rely on an initial surge in cross-entropy loss.
    
- **Robustness Demonstrated Against Successful Attacks:** 
- The authors note that the same attack configurations that reviewers suggested might be "uncalibrated" actually succeed against baseline defenses. The fact that TAR resists these successful attacks highlights its robustness.
    

---

The authors conclude by thanking the reviewer for their valuable feedback and kindly asking them to consider raising their score if their concerns have been addressed.



## ------------------------------------------------------
## Official Review of Submission8414 by Reviewer Rc9Z

### **Overall Assessment**

- **Focus:** Robustness of open-weight LLMs, proposing a novel defense method called TAR.
    
- **Soundness:** Good (3/5)
    
- **Presentation:** Good (3/5)
    
- **Contribution:** Good (3/5)
    
- **Rating:** Accept, good paper (8/10)
    
- **Confidence:** Fairly confident (3/5)
    

### **Strengths**

- **Contribution:**
    
    - Experimental results (Table 1) are significant and validate the main claims.
        
    - The proposed method (TAR) is intuitive and well-explained, even for those new to LLM defense.
        
- **Novelty:**
    
    - Presents the first defense method for autoregressive LLMs against "tampering attacks" (as opposed to more common input-based jailbreaking attacks).
        
    - Brings new insights into LLM robustness research by addressing tampering threats.
        
- **Presentation:**
    
    - The preliminary section (Section 3) is concise and clear, making the technical content easy to follow.
        

### **Weaknesses**

- **Presentation:**
    
    - Lack of discussion on experimental costs (time, GPU memory).
        
    - Question about the minimum GPU requirements (e.g., beyond the 8 A100 with 80GB mentioned).
        
    - Suggestion to include a formal "statement of contribution."
        

### **Questions for Authors**

- **Trade-off between Capabilities and Tamper Resistance:** In Figure 2, TAR shows lower capabilities than baseline methods. Is there an inherent trade-off between a model's general capabilities and its resistance to tampering?
    

### **Ethics Review**

- **Flag for Ethics Review:** No ethics review needed.
    
- **Details:** No ethics concerns.

## AUTHOR'S RESPONSE

This author response addresses the reviewer's comments regarding computational cost, contributions, and the capabilities-robustness tradeoff.

Here's a breakdown:

### **Response to Runtime and Memory Costs**

- **Computational Complexity:** 
- TAR has an O(NK) runtime complexity, where N is outer loop steps and K is inner loop steps. They note that K is typically much smaller than N (e.g., K=64, N=750).
    
- **Forward/Backward Passes:**
- Totaling 60,750 forward/backward passes for their chosen parameters (N=750, K=64, batch size 64).
    
- **Efficiency Compared to Pre-training:** 
- This is orders of magnitude _fewer_ passes than original model pre-training (citing Llama-3-8B with 1.2 million optimization steps and a batch size of 8 million).
    
- **Runtime on Hardware:**
- On 8 A100 (80GB) GPUs, the method runs in 18 hours.
    
- **Memory Efficiency:** 
- Since TAR uses first-order meta-learning, meta-gradients can be accumulated in-place, requiring only memory equivalent to one extra copy of model gradients per device. This means TAR's memory complexity scales by a constant factor with sharded model size, having the same asymptotic memory complexity as standard training.
    
- **Action for Revision:** 
- They will include more discussion of these costs and mitigations in the revised paper.
    

### **Response to Contributions**

- **Action for Revision:** 
- They will add a clear statement of contributions in the revised paper.
    
- **Three Core Contributions to Highlight:**
    
    1. Development of a tamper-resistance evaluation pipeline for LLMs, including an extensive red-teaming suite.
        
    2. Demonstration that tamper-resistance is tractable for LLMs and introduction of the first method to provide robustness against a wide range of adversaries.
        
    3. Proof that existing LLM adversarial training methods (including concurrently released unlearning methods) are not robust.
        

### **Response to Capabilities-Robustness Tradeoff**

- **Acknowledgement:** They confirm that a trade-off between benign capabilities and tamper-resistance is observed, similar to standard adversarial robustness in fields like image classification where adversarial training often leads to some performance drop.
    
- **Prior Discussion:** This tradeoff is already discussed in Section 5 and Appendix C.4 of their paper.

## ------------------------------------------------------

##  Official Review of Submission8414 by Reviewer rLai

[LINK](https://openreview.net/forum?id=4FIjRodbW6&noteId=CCQQq3Ak4X)
This review provides a mixed assessment of the paper, acknowledging the importance of the problem while expressing significant concerns about the proposed solution's sustainability, technical contribution, and evaluation clarity.

Here's a breakdown:

### **Overall Assessment**

- **Focus:** Addresses the lack of robustness in open LLMs against model weight tampering attacks by proposing a defense method called TAR.
    
- **Soundness:** Fair (2/5)
    
- **Presentation:** Good (3/5)
    
- **Contribution:** Fair (2/5)
    
- **Rating:** Marginally below the acceptance threshold (5/10)
    
- **Confidence:** Confident (4/5)
    

### **Strengths**

- **Important Problem:** The security issues with open LLMs are highly relevant and interesting.
    
- **Proposed Solutions:** Authors offer a set of solutions to address these security threats.
    
- **Experimental Validation:** Experiments demonstrate the performance of the proposed mechanisms.
    

### **Weaknesses and Questions (Intertwined)**

- **Insufficient Sustainability/Superficial Resilience:**
    
    - The effectiveness of the defense (integrating adversarial learning and meta-learning) is seen as dependent on the _diversity_ of attack types included in the training data (Equation 1).
        
    - This raises concerns that the resilience might be "superficial" and doesn't guarantee the long-term security of open-weight LLMs.1
        
    - The paper lacks corresponding theoretical analysis or proof for this sustainability.2
        
- **Incremental Technical Contributions:**
    
    - While the paper is clear, the innovation in terms of technical aspects and application scenarios is not evident.
        
    - Solutions appear based on existing methods, and the authors haven't clearly articulated their unique contributions.
        
    - The reviewer requests further clarification on this.
        
- **Reliance on Adversarial Training Data:** The reviewer reiterates that the mechanism's performance is closely tied to the specific adversarial training methods and data used, indicating that its overall resilience remains a significant issue.
    
- **Unclear Figure 5 Presentation:**
    
    - The comparison in Figure 5 (TAR vs. existing mechanisms) is unclear and potentially confusing.
        
    - The authors need to explain why TAR's performance changes significantly as the "step size" increases in this figure.

## AUTHOR'S REPONSE
This author response addresses the reviewer's concerns regarding the sustainability of the proposed TAR method, its technical contributions, and the clarity of Figure 5.

Here's a breakdown:

### **Main Contribution & Sustainability**

- **Core Contribution:** 
- The authors emphasize that their main contribution is demonstrating that **tamper-resistance in LLMs is tractable** for the first time.
    
- **No Absolute Guarantee:**
- They explicitly state that their method does not guarantee robustness against _arbitrary_ adversaries.
    
- **Difficulty of Theoretical Guarantees:** 
- Obtaining theoretical guarantees for tamper-resistance is currently very difficult due as it's a new research area.
    
- **Empirical Robustness:** 
- They rely on extensive "red teaming" (stress-testing with 28 diverse attacks) to build confidence in empirical robustness, acknowledging some attacks still succeed.
    

### **Technical Contributions**

- **Disagreement on "Incremental":**
- The authors respectfully disagree that their technical and experimental contributions are incremental.
    
- **Building on Prior Work:** 
- While TAR builds on MLAC and first-order MAML, they list numerous _significant_ technical contributions _on top_ of these:
    
    - Developing a novel tamper-resistance evaluation pipeline and red-teaming suite for LLMs.
        
    - Showing that MLAC does not scale to LLMs.
        
    - Developing a superior 2-stage unlearning/alignment method.
        
    - Demonstrating that input-based adversarial training (e.g., R2D2) does not generalize to weight tampering attacks.
        
    - Showing that concurrent work (e.g., RepNoise) lacks robustness against a broader set of attacks.
        
    - Identifying an improved outer loop loss for tamper-resistance training.
        
    - Introducing methods to minimize memory overhead for adversarial meta-learning.
        
- **Action for Revision:** They will add an explicit contributions list to the updated paper for clarity.
    

### **Resilience & Figure 5 Clarity**

- **Improved Resilience:**
They agree resilience is important and reiterate that their method _improves_ resilience compared to existing methods, even if it doesn't provide full resilience. Their goal is to show progress is possible.
    
- **Clarifying Figure 5:**
    
    - Clarify that the x-axis represents the _number of optimizer steps_, not step-size.
        
    - Highlight that TAR maintains high adversary loss across all steps, demonstrating strong performance.
        
    - Will add further analysis to the paper, explaining that the significant changes in adversary loss are due to the model sharply increasing cross-entropy loss early in fine-tuning to maximize entropy.
        

The authors conclude by thanking the reviewer and requesting a score increase if their concerns have been addressed.


## ------------------------------------------------------

## Official Review of Submission8414 by Reviewer vbKt

[](https://openreview.net/forum?id=4FIjRodbW6&noteId=zFmqnRxR7U)

### **Overall Assessment**

- **Focus:** Introduces TAR, a method to enhance LLM robustness against tampering attacks.
    
- **Soundness:** Fair (2/5)
    
- **Presentation:** Fair (2/5)
    
- **Contribution:** Fair (2/5)
    
- **Rating:** Marginally below the acceptance threshold (5/10)
    
- **Confidence:** Fairly confident (3/5)
    

### **Strengths**

1. **High Significance of Topic:** The paper addresses critical security issues in open-source LLMs (e.g., spreading misinformation, discrimination, generating harmful content), highlighting the need for robust safeguards.
    
2. **Successful Resistance:** The proposed method successfully resists thousands of malicious fine-tuning steps.
    

### **Weaknesses**

1. **Section 4 Organization:** Needs refinement; suggests outlining design motivation before details, and including mathematical formulation of the loss function.
    
2. **Figure 1 Clarity:** Caption needs to explain the difference between its two branches more clearly.
    
3. **Unclear Claim of 5000 Fine-tuning Steps:**
    
    - Lacks detail on parameters (batch size, learning rate) which significantly influence step count.
        
    - Missing comparative analysis with typical fine-tuning steps for downstream tasks.
        
4. **Threat Model Clarity:**
    
    - "Compute-bounded attacker" is not clearly defined.
        
    - Needs concrete examples for `capabilities_metric` and `safety_metric`.
        
5. **Minor Differences in Section 4.2:** Authors claim differences from standard meta-learning are minor and do not present significant technical challenges.
    
6. **Over-reliance on Empirical Data:**
    
    - Frequent use of 'empirically' in methods section is noted.
        
    - Relying solely on empirical data for loss function selection may limit robustness.
        
    - Requires theoretical analysis comparing entropy loss vs. cross-entropy loss efficacy.
        
7. **Performance Metrics & Trade-off:**
    
    - Performance metrics need clearer explanation.
        
    - Results suggest the solution sacrifices task performance for robustness, indicating potential flaws.
## AUTHOR'S RESPONSE


### **1. Clarifying Design Motivation in Section 4**

- **Reviewer's Point:** Section 4 needs better organization and a clear outline of design motivation, especially for Figure 3.
    
- **Author's Response:**
    
    - Apologize for previous lack of clarity.
        
    - **Action:** Will add a brief outline of the motivation for their tamper-resistance loss at the beginning of Section 4.2.
        

### **2. Mathematical Formulations of Loss Functions**

- **Reviewer's Point:** Mathematical formulations for loss functions are absent in the main paper.
    
- **Author's Response:**
    
    - Point to Appendix B.2 where these are already fully described.
        
    - **Action:** Agree that including these formulations in the main paper would improve readability and have added them to the updated version.
        

### **3. Improved Caption for Figure 1**

- **Reviewer's Point:** Figure 1's caption needs to explain the difference between the two branches more clearly.
    
- **Author's Response:**
    
    - Agree the original caption lacked clarity.
        
    - **Action:** Provide a detailed new caption for Figure 1, contrasting conventional safeguards (easily bypassed by fine-tuning) with TAR (maintains robustness against fine-tuning to reintroduce harmful capabilities).
        

### **4. Clarifying 5000-Step Claim and Attack Configurations**

- **Reviewer's Point:** The 5000-step claim in Section 1 is unclear, lacks intuitive contribution, and doesn't specify fine-tuning details (batch size, learning rate), and lacks comparison to typical downstream task fine-tuning steps.
    
- **Author's Response:**
    
    - **5000-step attacks:** Clarify that these correspond to Adv. 1 and Adv. 2 in Table 9 and Figure 4, using a held-out biology dataset.
        
    - **Hyperparameters:** Confirm that these 5000-step adversaries use similar optimization hyperparameters (learning rate between 2×10−5 and 4×10−5) as other red-teaming attacks.
        
    - **Contribution:** Believe their robustness results on these strong adversaries support their claim of withstanding much greater optimization pressure than prior work.
        
    - **Action:** Will clarify this in the updated paper.
        
    - **Hyperparameter Details:** Reiterate that batch size, learning rate, and other hyperparameters are _already fully specified_ in Tables 7, 8, 9, and 10 in the Appendix.
        
    - **Comparative Analysis:** State that attacks in Table 9 use standard fine-tuning hyperparameters (LRs between 2×10−6, 2×10−5, 4×10−5; 1000 or 5000 steps; batch sizes 32 or 64), including LoRA and full-parameter training, all representative of common Llama-3-8B fine-tuning settings.
        

### **5. Clarifying the Threat Model and Metrics**

- **Reviewer's Point:** Threat model lacks clarity on "compute-bounded attacker" and needs concrete examples for `capabilities_metric` and `safety_metric`.
    
- **Author's Response:**
    
    - **Compute-Bounded Attacker:** Their aim was to capture a broad range of attacks, not just vary specific hyperparameters. Acknowledge that quantifying attacks by concrete metrics like FLOPs could add clarity.
        
    - **Metrics Examples:** Point to Section 3.2 where concrete examples are already provided (MMLU and WMDP for knowledge restriction; MT-Bench and HarmBench for harmful request refusal).
        
    - **Action:** Offer to suggest other benchmarks like GSM8K or HumanEval.
        

### **6. Clarifying Differences from Standard Meta-learning**

- **Reviewer's Point:** Differences highlighted between TAR and standard meta-learning seem minor and not technically challenging.
    
- **Author's Response:**
    
    - Agree that their method falls within the meta-learning framework.
        
    - **Substantial Differences:** Argue there are _numerous substantial differences_ with significant technical implications:
        
        - **Adversarial Optimizers:** Standard meta-learning typically uses non-adversarial inner and outer loop optimizers; TAR uses adversarial optimizers. This fundamentally changes training dynamics.
            
        - **Goal:** Standard meta-learning aims to minimize test-time loss in _few steps_; TAR aims to _maintain high adversary loss_ for _as many steps as possible_. This means TAR seeks to generalize robustness to a _larger number of steps_ than seen during training, which has no analogy in standard meta-learning.
            
        - **Efficiency Tricks:** The fundamental differences necessitated developing efficiency tricks (e.g., inner-loop subsampling in Appendix B.3 and B.4) to make adversarial meta-learning feasible.
            
    - **Action:** Will add a section to the appendix clarifying this distinction, acknowledging it might be too "in the weeds" for most readers.
        

### **7. Theoretical Analyses and "Empirically" Term**

- **Reviewer's Point:** Over-reliance on "empirically" suggests limitations; theoretical analysis of loss functions (entropy vs. cross-entropy) is needed.
    
- **Author's Response:**
    
    - **Theoretical Guarantees Difficulty:** Agree theoretical analysis is valuable but argue that obtaining theoretical guarantees for tamper-resistance is currently very difficult given the nascent stage of research (comparing to certified vs. empirical robustness in other adversarial domains).
        
    - **Best Practices:** Follow standard empirical robustness research practices by extensively stress-testing their defense against a diverse suite of strong adversaries, including custom attacks designed to break it.
        
    - **Loss Function Choice:** This empirical approach led to discoveries, including the selection of a loss function that performed better across a broad range of test-time attacks.
        
    - **Premature Analysis:** Suggest theoretical analysis of their _particular_ loss function might be premature, as methods are likely to evolve in future work.
        

### **8. Explanation of Metrics Used and Capabilities-Robustness Tradeoff**

- **Reviewer's Point:** Performance metrics need clear explanation. The solution appears to sacrifice task performance for robustness, suggesting potential flaws.
    
- **Author's Response:**
    
    - **Existing Metrics:** Clarify that they _do not create new metrics_. They use existing, standard benchmarks and their associated metrics: MMLU and WMDP for benign performance/safety in knowledge restriction; MT-Bench for benign performance and HarmBench for prompting ASR in harmful request refusal. These are listed in Table 1 caption and Section 3.2.
        
    - **Action:** Offer to re-state the use of these benchmarks in Section 5.1 for clarity.
        
    - **Capabilities-Robustness Tradeoff:**
        
        - **Acknowledgement:** Confirm observing this tradeoff, which is common in nearly all prior adversarial robustness work.
            
        - **Not a Flaw:** Emphasize that this is _not a technical flaw_ in the method but a common property of such defenses.
            
        - **Progress Demonstrated:** Previously, tamper-resistance was not achievable at all. Their results show that a _moderate reduction_ in benign performance _greatly improves_ tamper-resistance, demonstrating the tractability of the problem for the first time.
            
        - **Future Work:** Agree that improving benign capabilities alongside robustness is an important area for future research.

## ------------------------------------------------------

## Official Review of Submission8414 by Reviewer NvnF

[](https://openreview.net/forum?id=4FIjRodbW6&noteId=Ks38WWJgRq)
Here's a summary of the review in points:

### **Overall Assessment**

- **Focus:** Addresses fine-tuning attacks on LLMs, proposing a novel defense method called TAR.
    
- **Method:** TAR is based on adversarial training and meta-learning, using a new training objective that combines adversarial and retaining objectives to maintain utility.
    
- **Soundness:** Good (3/5)
    
- **Presentation:** Good (3/5)
    
- **Contribution:** Good (3/5)
    
- **Rating:** Marginally above the acceptance threshold (6/10)
    
- **Confidence:** Confident (4/5)
    

### **Strengths**

- **Well-Written and Logical:** The paper is clearly written and easy to follow.
    
- **Comprehensive Experiments:** The large-scale experiments are thorough and convincing.
    

### **Weaknesses**

- **Missing Time Complexity Analysis:**
    
    - The paper lacks analysis of the computational cost, which is crucial given that both adversarial training and meta-learning are time-consuming.
        
    - Raises concerns about the practicality of the method for large models.
        
    - Suggests authors provide empirical or theoretical computation analysis.
        
- **Hyperparameter Influence Not Discussed:**
    
    - There are many hyperparameters (e.g., number of outer loops, coefficients λTR​ and λretain​ in Eq. 1).
        
    - How these hyperparameters affect defending performance is not discussed.
        
    - Suggests including this discussion.
        

### **Questions**

- **Requirements for `A_train` in Eq. 1:**
    
    - What kind of `A_train` (the set of training attacks) is needed for good defense performance?
        
    - Specifically, how diverse do the included attacks need to be?
        
    - How many adversarial examples for each attack should be included?
        

### **Ethics Review**

- **Flag For Ethics Review:** No ethics review needed.
    
- **Details Of Ethics Concerns:** N/A


## AUTHOR'S RESPONSE
This author response addresses concerns about the computational efficiency of TAR, the influence of its hyperparameters, and the requirements for the training attack set (Atrain​).

### **1. Time and Memory Complexity of TAR / Efficiency Optimizations**

- **Reviewer's Concern:** Time complexity analysis is missing. Adversarial training and meta-learning are time-consuming, raising concerns about practicality for large models.
    
- **Author's Response:**
    
    - **Acknowledgement:** Agree that computational efficiency for large models is a valid concern.
        
    - **Mitigation:** State that their paper _already addresses_ these concerns in Appendix B.3 and B.4 by introducing "efficiency tricks" to make first-order meta-learning feasible for billion-parameter LLMs.
        
    - **Runtime Complexity:** Explain that TAR has an inner-loop/outer-loop meta-learning setup with O(NK) runtime complexity (N = outer loop steps, K = inner loop steps). Empirically, K≪N (e.g., K=64,N=750).
        
    - **Gradient Computations:** For each inner loop step, an adversary gradient and a meta-gradient are computed. The outer loop also computes a retain-set gradient.
        
    - **Subsampling Trick:** They use a subsampling trick (Appendix B.3) to compute meta-gradients every 4 adversary steps, speeding up training by a constant factor without sacrificing robustness.
        
    - **Total Forward/Backward Passes:** For N=750,K=64, batch size 64, this totals **60,750 forward/backward passes**.
        
    - **Comparison to Pre-training:** This is _multiple orders of magnitude fewer_ passes than original model pre-training (e.g., Llama-3-8B's 1,200,000 passes with an 8,000,000 batch size).
        
    - **Empirical Runtime:** On 8 A100 (80GB) GPUs, their method with N=750 and K=64 runs in **18 hours**.
        
    - **Memory Overhead:**
        
        - Memory might seem a primary concern for large models.
            
        - However, being first-order meta-learning (Appendix B.4), meta-gradients can be accumulated in-place.
            
        - With model sharding algorithms like FSDP (Fully Sharded Data Parallel), this only requires **one extra copy of model gradients per device**, without additional distributed communication.
            
        - **Memory Efficiency:** TAR is memory-efficient, scaling memory usage by only a constant factor with sharded model size, meaning it has the **same asymptotic memory complexity as standard training**.
            
    - **Action for Revision:** Will include more detailed discussion of these costs and mitigations in their revised paper.
        

### **2. Understanding the Hyperparameters of TAR**

- **Reviewer's Concern:** How hyperparameters (e.g., outer loop steps N, λTR​, λretain​) influence defending performance is not discussed.
    
- **Author's Response:**
    
    - **Existing Discussion:** They state that detailed discussions of important hyperparameters are _already included_ in their ablations (Appendix C.2 and C.4).
        
    - **Inner Loop Steps (K):** Figure 6 in Appendix C.2 shows the effect of increasing K.
        
    - **λTR​:** Table 6 in Appendix C.4 discusses the effect of varying λTR​, noting that increasing it from 1.0 to 4.0 (with λretain​=1.0) greatly improved tamper-resistance.
        
    - **Outer Loop Steps (N):** They ran the method until convergence of the full TAR loss (Eqn. 1), with smoothed outer-loop TAR loss shown in Figure 3. N=750 for weaponization knowledge restriction and N=100 for harmful request refusal.
        
    - **Action for Revision:** Will add a list of all hyperparameters with recommended settings from their ablations in one location in the updated paper for easier reference.
        

### **3. Sampling Attacks During TAR (Atrain​ in Eq. 1)**

- **Reviewer's Question:** What kind of Atrain​ is needed for good defense (e.g., diversity, number of adversarial examples)?
    
- **Author's Response:**
    
    - **Key Factors:** They found that diversity in both **dataset distribution** and **learning rate** significantly contributes to robustness.
        
    - **Example of Diversity:** In Section 3.3, they mention including new fine-tuning attacks that broke intermediate versions of TAR.
        
    - **Targeted Patching:** Give an example of the "R->F adversary" (benign fine-tuning followed by forget-set fine-tuning). Table 4 in Appendix C.3 shows that targeted patching is possible: sampling a reduced version of this R->F adversary _improved downstream robustness_, demonstrating the benefit of increasing diversity.
        
    - **Current Best Configuration:** The configuration of training adversaries listed in Tables 7 and 8 yielded the best robustness in their experiments for the considered domains. They hope future work will improve upon this.
## ------------------------------------------------------

## Official Review of Submission8414 by Reviewer i3oQ
### **Overall Assessment**

- **Focus:** Proposes TAR method for robust safe LLMs against weight tampering attacks.
    
- **Performance Claim:** Achieves superior performance on weaponization knowledge restriction and harmful refusal training.
    
- **Soundness:** Good (3/5)
    
- **Presentation:** Fair (2/5)
    
- **Contribution:** Fair (2/5)
    
- **Rating:** Marginally below the acceptance threshold (5/10)
    
- **Confidence:** Fairly confident (3/5)
    

### **Strengths**

- **Importance of Topic:** Defense against LLM weight tampering attacks is a highly significant research area.
    

### **Weaknesses and Questions (Intertwined)**

- **Limited Universality due to Proxy Objectives:**
    
    - The use of "too many proxy objectives" like `safety_metric` and `capabilities_metric` limits the method's significance and universality.
        
    - Asks for proof that TAR works under _other_ `safety_metric` or `capabilities_metric` definitions.
        
- **Lack of Novelty (Adversarial Training):**
    
    - Views TAR (from Eq. 1) as a "simple adversarial training paradigm" with proxy indicators.
        
    - Notes that adversarial training is an existing, well-known technique.
        
    - Asks if there are any _novel findings or modifications_ within TAR beyond standard adversarial training that suggest its novelty.
        
- **Generalization of Proxy Metrics:**
    
    - Asks for experiments to validate if TAR's performance on its chosen proxy metrics generalizes to other safety scenarios or benchmarks.
        
- **Training Cost:**
    
    - Questions the training cost of TAR, specifically if it's _larger_ than original model training.
        
    - Implies a concern about practicality.
## AUTHOR'S RESPONSE

**1. Definition of Metrics**

- **Only Two Objectives:** They clarify having just two proxy objectives (safety and capabilities), which is standard.
    
- **Generalization Experiments Exist:** They confirm their experiments already show TAR's generalization across different safety and capability metrics in the two domains described in Section 3.2.
    
- **Action:** Will clarify these points in the updated paper.
    

**2. TAR Training Cost**

- **Runtime:** Each TAR fine-tuning run takes 18 hours on 8 NVIDIA A100 80GB GPUs.
    
- **Significantly Less Compute:** This is approximately 8 orders of magnitude less compute than original Llama-3-8B pre-training.
    
    - Involves 60,750 forward/backward passes (vs. 1.2 million optimization steps with a much larger batch size for pre-training).
        
    - They use a subsampling trick (Appendix B.3) to speed up training.
        
- **Memory Efficiency:**
    
    - Uses first-order meta-learning, allowing in-place accumulation of meta-gradients.
        
    - Requires only one extra copy of model gradients per device, even with FSDP.
        
    - Has the same asymptotic memory complexity as standard training.
        
- **Action:** Will include more discussion on costs and mitigations in the revision.
    

**3. Adversarial Training Formulation and Method Novelty**

- **Distinct Objective:** Their objective (Eq. 1) is a meta-learning objective because the attack modifies _model parameters_ (θ), unlike traditional adversarial training which modifies _inputs_. This requires back-propagation through an inner-loop of fine-tuning.
    
- **Bridge for Readers:** They drew a connection to adversarial training to help readers, but acknowledge it might cause confusion about applicability of existing results.
    
- **Core Novelty:** Their main contribution is demonstrating that **tamper-resistance for LLMs is tractable** for the first time, significantly improving over prior unsuccessful methods.
    
- **Additional Contributions:**
    
    - Developed the first tamper-resistance evaluation pipeline and red-teaming suite for LLMs.
        
    - Showed MLAC does not scale to LLMs.
        
    - Developed a superior 2-stage unlearning/alignment method.
        
    - Demonstrated input-based adversarial training doesn't generalize to weight tampering.
        
    - Showed concurrent work lacks robustness against broader attacks.
        
    - Identified an improved outer loop loss (maximizing entropy).
        
    - Introduced memory overhead minimization methods for adversarial meta-learning.
        
- **Significance:** Believe these contributions are of interest and can foster discussion.

## ------------------------------------------------------


