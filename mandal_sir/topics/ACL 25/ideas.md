
### Main objective : generalization analyzing failure condition of MMLM m-editing methods

- failure cases why is it failing 
- evaluation matrix 
- how efficient
- what led to the failure



Questions to ask :

- Where and why do MLLM  editing methods fail? (to generalize)
- What are the typical pattern in those failures - i.e. find the reasons of failure and pattern
- find out the non llm related conditions - (language of dataset , syntactic structure etc) - find  which contributes the most
- how the editing  method impact their accuracy 
	- from the MMEDIT paper
- dataset preparation 
	- in diff languages
	- add intentional challenges
	- add already causes of failures in it
- editing task 
	- stress testing the model
	- paraphrasing
	- editing entity names or info regarding the entity
	- synthetic data from other llms
- failure cases analysis
	- manual or automated
	- pre and post editing analysis
- use evaluating metrics from other papers
- use other frameworks for evals and failures , compare against other baselines 
- do these on quantized models https://x.com/llm_sec/status/1853533265774436816
- 



ideas till 5-11-14:
- test the editing methods in the MMEDIT repo and check for any methods 
- look for effect of unlearning and editing in geometry of concepts also connect lobes to layers after editing
- check for failure cases in EasyEdit repo - llms and methods
- 
