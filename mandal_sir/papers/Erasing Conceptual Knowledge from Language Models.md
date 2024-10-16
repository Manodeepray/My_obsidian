code : https://elm.baulab.info/

 We propose an evaluation paradigm centered on three critical criteria: 

 1. innocence (complete knowledge removal), 
 2. seamlessness (maintaining conditional fluent generation)
 3. specificity (preserving unrelated task performance). 
  
   ELM employs targeted low-rank updates to alter output distributions for erased concepts while preserving overall model capabilities including fluency when prompted for an erased concept. 

robust concept erasure should simultaneously satisfy all three criteria.

Representation Misdirection for Unlearning (RMU) (Li et al., 2024) fine tunes the earlier layers of model to unlearn a concept by randomizing and amplifying the internal activations when prompted with text related to the concepts being erased, but it suffers from a lack of seamlessness, since the method creates a model that generates obvious gibberish in response to a dangerous prompt


![[Pasted image 20241016002217.png]]


WhoIsHarryPotter (Eldan & Russinovich, 2023), employ a two-stage approach, training a reinforced model for the concept being erased and then training an unlearned model that behaves differently on the reinforced logits. Our analysis reveals that this kind of approach falls short in innocence, since the erased knowledge can still be recovered through multiple-choice prompting which was consistent with prior findings


### core idea

Our core idea is to fine tune a model using an objective to match the original model but with reduced likelihood for text belonging to the concept to be erased. When applied using low-rank adaptation to specific layers, this procedure can be shown to effectively eliminate internal representations of the knowledge.

same objective to synthesize fine-tuning training data that can be used to ensure seamlessness

### related work

Our work is most directly comparable to three state-of-the-art techniques: Representation Misdirection for Unlearning (RMU) (Li et al., 2024), WhoIsHarryPotter (WHP) (Eldan & Russinovich, 2023), and Representation Noising (RepNoise) (Rosati et al., 2024)

**RMU** fine-tunes models to align internal activations with random scaled vectors when processing targeted concepts

**WHP** (Eldan & Russinovich, 2023) employs a two-stage approach, first training a reinforced model for the concept being erased and then training an unlearned model to behave differently on the reinforced logits

 **RepNoise** (Rosati et al., 2024) focuses on removing information about harmful representations across all layers of the LLM, by doing gradient ascent along with some representation noising (training internal representations to match Gaussian noise)


Distilling generative model outputs



