# Abstract
### Core Differences in Learned Solutions

- **Spectral Property Disparity:** The central finding is that LoRA and full fine-tuning produce model weight matrices with **structurally different singular value decompositions (SVDs)**.
    
- **LoRA's Intruder Dimensions:** LoRA-trained weight matrices uniquely develop **new, high-ranking singular vectors**, termed **"intruder dimensions"**. These are _not_ observed in models trained with full fine-tuning.
    

---

##  Impact and Mechanism of Forgetting

- **Localized Forgetting in LoRA:** While LoRA is known to cause less forgetting of the pre-training knowledge than full fine-tuning, this forgetting is found to be **vastly localized to the intruder dimensions**.
    
- **Causal Role of Intruder Dimensions:** Through causal intervention (changing the associated singular values), the study demonstrates that the **intruder dimensions _cause_ forgetting** of the original pre-training distribution.
    
- **Mitigation Strategy:** **Scaling down the singular values associated with the intruder dimensions** significantly improves the model's ability to retain the pre-training distribution with only a minimal loss in downstream task performance.
    

---

##  Implications for Continual Learning

- **Harmful Accumulation:** The analysis suggests that the **accumulation of intruder dimensions is detrimental**, leading to increased forgetting, especially in sequential fine-tuning settings.
    
- **Practical Validation:** The study confirms that **LoRA models used in continual learning** _do_ accumulate these intruder dimensions and subsequently exhibit **worse performance**, underscoring the practical significance of these spectral findings.

---
### LoRA Parameterization vs. Full Fine-Tuning

- **Full Fine-Tuning (FFT):** Treats **every parameter** in the model as trainable.
    
- **LoRA Approach:** Parameterizes the **learned weight update ($\Delta W$)** as the product of two **low-rank matrices** ($\Delta W = BA$), drastically reducing the number of trainable parameters.
    

---

###  Intrinsic Dimension Hypothesis (IDH) and LoRA's Rationale

- **Hypothesis Core:** The IDH posits that the meaningful update learned during fine-tuning (even full fine-tuning) lies within a much **lower-dimensional subspace** than the full parameter space.
    
    - It suggests the update has a **low intrinsic rank**.
        
    - Prior work has shown that **pre-training implicitly minimizes** this intrinsic dimension, making LLMs easier to fine-tune.
        
- **Implication for LoRA:** The IDH offers a key theoretical explanation for LoRA's success, suggesting that LoRA's low-rank parameterization ($r \ll \text{rank}(W)$) is sufficient to **recover an approximately equivalent solution** to FFT because the _true_ necessary update dimension is inherently small.
    

---

### Empirical Challenges to LoRA/FFT Equivalence

- Despite the IDH, a clear **principled explanation** for why LoRA _matches_ FFT performance remains elusive.
    
- **Observed Differences:** Prior work noted that LoRA and FFT have different abilities to independently adjust a neuron's **angle and magnitude** when transforming its input.
    
- **Performance Gaps:** LoRA has demonstrated **difficulty matching FFT performance** on complex tasks like:
    
    - Code generation.
        
    - Long-form text generation.
        
- **Open Question:** It is still debated whether these performance issues stem from a fundamental **limit in LoRA's fitting ability** for difficult tasks, or if LoRA and FFT are learning **inherently different solutions**—a question the analysis of **spectral properties** (intruder dimensions) attempts to answer.

# Introduction

In this paper, we show that full fine-tuning and LoRA learn different solutions with characteristic differences in their spectral properties (as shown in Fig. 1 for LLaMA2-7B [Touvron et al., 2023b]) and that these spectral differences are causally related to different model behaviors. We observe: 

1. ==LoRA and full fine-tuning produce structurally different parameter updates, characterized by the existence of intruder dimensions in weight matrices tuned by LoRA.== Intruder dimensions are singular vectors with large associated singular values that are very dissimilar to the singular vectors in the pre-trained weight matrix. In contrast, fully fine-tuned models remain spectrally similar to the pre-trained model and do not contain intruder dimensions. 
2. ==LoRA forgets less than full fine-tuning...but not always. ==We extend the findings of Biderman et al. [2024] that LoRA forgets less to the case even when there is ==equal fine-tuning performance between LoRA and full fine-tuning,== showing that a difference in fit is not simply the cause of this finding but rather is inherent to these methods. However, this is not always the case: despite nearly identical fine-tuning task accuracies, we show that ==different selections of LoRA alpha and learning rate lead to starkly different generalization behaviors, ==even leading to LoRA forgetting more than full fine-tuning. We also find that models with the best generalization for each of these hyperparameter settings have the fewest intruder dimensions.
3. ==Intruder dimensions cause forgetting of the pre-training distribution.== Scaling down the associated singular values of high-ranking intruder dimensions leads to a large drop in loss on the pre-training distribution (forgetting) but only a minimal drop in test performance. The ==drop in forgetting we observe when scaling down singular vectors is unique to intruder dimensions ==and indicates that they ==interfere with the pre-trained language modeling ability== of these models. Given this finding, we should expect accumulating intruder dimensions to be ==harmful and lead to more forgetting.== To amplify this accumulation and examine its effect, we fine-tune in a ==continual learning  setting== (sequentially fine-tuning on many tasks) and show that ==LoRA models do indeed tend to forget more on previously learned tasks in this setting, ==providing additional support for our findings.



##  **Intrinsic Dimension Hypothesis (IDH)** 

The **Intrinsic Dimension Hypothesis (IDH)** is a foundational concept that offers a powerful theoretical lens to ==explain the effectiveness and efficiency of fine-tuning large machine learning models,== particularly techniques like LoRA.1

The hypothesis states that while a Large Language Model (LLM) may have billions of parameters ($D$), the ==**effective number of parameters** or degrees of freedom required to adapt it to a specific downstream task ($d_{int}$) is **remarkably low**== ($d_{int} \ll D$).5 This effective dimension represents the low-dimensional manifold where the optimal solutions for the fine-tuning objective reside.



## INTRUDER DIMENSIONS
singular vectors that are dissimilar to those in the pre-trained weight matrix and are learned during finetuning

# references

original paper
https://arxiv.org/pdf/2410.21228

Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning
https://arxiv.org/abs/2012.13255
[[INTRINSIC DIMENSIONALITY EXPLAINS THE EFFECTIVENESS OF LANGUAGE MODEL FINE-TUNING]]


-