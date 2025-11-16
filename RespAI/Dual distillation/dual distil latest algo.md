

# single model

1. take model xbase
2. xbase --> harmful ft --> x harm
3. eval toxicity --> x harm
#influence----get-the-mariginal-gain-for-unlearning-objective
4. repeat 2 and 3 --> till toxicity goal achieved
5. xbase --> safe ft --> x safe
6. eval safety metrics --> x safe
7. repeat 5 and 6 --> till safety goal achieved
8. get log probs from xharm  , xsafe for all the req datasets for distillation
9. prep  peft params for --> xbase
	1. peft --> lora w/o regrets
	2. reft
10. if lat or canary 
	1. prep model lora
	2. prep lat / canary
11. dual distillation --> xbase
	1. rev dpo + revkl --> adv dataset
	2. dpo + kl --> safe dataset
	3. ppl check every n steps
12. save model
13. eval on hs and fa


# modular framework

token translation


# additional



use marginal gain or use influence function --> identity matrix for the inverse hessian for understanding  which datapoint (harmful)  is more responsible
 

#todo 
get new dataset
fix data leakage
solve influence problem

#test
influence +npo
lat
hyper network 
>to get perturbation
>to lora

