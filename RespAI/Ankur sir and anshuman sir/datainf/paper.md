 


![[Pasted image 20241207015747.png]]

DataInf can be computed in O(PLl=1 ndl) 
operations with O(maxl∈[L] dl) memory. In terms ofcomputational complexity, DataInf is much faster than LiSSA, and it does not require iterative operations.

Moreover, DataInf has a better memory complexity than LiSSA because it does notrequire storing Hessian matrices. 
Table 1 compares DataInf with the exact computation of theinfluence function (equation 2, denoted by Exact) and LiSSA when a model is a multilayer perceptron.



EXPERIMENTS

models used:
- RoBERTa 
- Llama-2-13B-chat
- stable_diffusion-v1.5

Low-Rank Adaptation (LoRA), a technique that significantly reduces thememory and computation required for fine-tuning large models (Hu et al., 2021). We fine-tune models by minimizing a negative log-likelihood function.


baseline influence computation methods,
- LiSSA with 10 iterations
- Hessian-free which computes a dot product of the first-order gradients
-  proposed method DataInf.

proposed algorithm ;

![[Pasted image 20250331194632.png]]