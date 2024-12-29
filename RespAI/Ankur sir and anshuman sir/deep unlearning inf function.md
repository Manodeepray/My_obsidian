### idea :
Majority of these approaches are best on llms output response 
Which is problematic in a sense and depends on context so heavily biased 
Ours will be unique. 
I liked null space projection idea
Which can be used with ours as our G” would be much more closer  to ideal representation

### TO-DO
1. Read the Paper “deep unlearning in LLM”
2. Harry Potter dataset, follow the setup in the above paper
3. Llama 2/3.2/Mistral/Phi 8B. Fine tune the model on new dataset - Harry Potter
4. Sample on different temperatures
5. Influence function



### sources

	
https://anonymous.4open.science/r/deep_unlearning_anonymous-2C73

https://github.com/ykwon0407/DataInf;


graph based editing methods
https://github.com/jianghoucheng/AlphaEdit

https://github.com/princeton-nlp/MQuAKE

https://aclanthology.org/2024.emnlp-main.1261/

https://arxiv.org/abs/2402.13593

https://dl.acm.org/doi/10.1145/3627673.3679722



### initial step pipeline 
- Read the paper we discussed on deep unlearning (https://openreview.net/pdf?id=CIN2VRxPKU) and familiarize yourself with their dataset (EDU-RELAT) and codebase.
- Read the influence function paper DataInf (https://openreview.net/pdf?id=9m02ib92Wz) and understand their codebase.
- The goal is to see for different test samples in the EDU-RELAT dataset what training samples are positively influential (as discovered by the influence function). 
- You need to adapt the DataInf code to use Llama-3.2-8B and the EDU-RELAT dataset.




take the edurelat datasets ..make a dataset
	
tokenize from influential data identification
	the 'prompt' in lora engine.create_tokenized_dataset needs to be according to the dataset
	 drop columns i.e some q or a

then run scr/sft_trainer.py with all the arguments

get datainf/models/maths_with_reasoning_13b and have adapter for the lora from sft trainer
then run the lora engine.compute gradient

compute the influence function



## litrerature review