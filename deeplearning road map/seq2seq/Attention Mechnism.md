



- [ ] Soft attention (Deterministic)
- [ ] Hard attention (Stochastic) 
- [x] Self-attention 
- [ ] Cross-Attention (Encoder-Decoder attention)  
- [x] Multi-Head Attention (MHA)   [[Multi-Head Attention]]
- [ ] Multi-Head Latent Attention (MLA)  
- [ ] Memory-Based attention  
- [ ] Adaptive attention 
- [ ] Scaled Dot-Product attention 
- [ ] Additive attention  
- [ ] Global attention  
- [ ] Local attention 
- [ ] Sparse attention 
- [ ] Hierarchical attention 
- [ ] Temporal attention



### Introduction

challenge in translation -> eng to french -> due to the order , length etc

encode decoder mechanism

attention is all you need - encoder and decoder model

encoder   - BERT - bidirectional encoder representation of transformers
		- DistilBERT
		- RoBERT
		- ALBERT
		- SBERT
		- RAGS
		- Embedding models
Decoders - gpts


### Main Ideas behind Attention

Word embeddings
Positional embeddings
Attention

![[Pasted image 20250317235700.png]]

Q,K,V
Q = query 
K = key
V =  value
dk = dimensions of the column..i.e. length of work embeddings

softmax -> makes it so that the sum of each row is one
https://learn.deeplearning.ai/courses/attention-in-transformers-concepts-and-code-in-pytorch/lesson/xy1tc/self-attention-vs-masked-self-attention
https://www.kaggle.com/code/raymanodeep/attention/edit

### Self attention vs Masked Attention

- same words for different meaning
- create word embedding networks
- predict the n^th  word from 0 to n-1^th words 
- use multiple activation functions
- each word has its embedding weights
- but this does not take the sequence / order of the words into consideration,
thus we use positional encodings on the word embeddings and produce **context-aware embeddings** 

2 types of transformers
				- Encoder type 
					- uses self-attention 
					- context aware embeddings
				- Decoder Type
					- uses masked self-attention
					- good at generating responses to prompts
					- GPT
  
	    
main diff
- self - attention :
	- looks at words before and after the word of interest
- masked self - attention :
	- ignores the words that come after the word of interest



![[Pasted image 20250317235641.png]]

M is added to scaled similarities

M = masking matrix -> prevents tokens from including anything that comes after them when calculating attention
add negative infinity to mask those we dont want

![[Pasted image 20250318000153.png]]

### Encoder-Decoder attention

Cross attention
- in this we use the output from self attention from encoder as k , v for decoder masked attention

### Multi-head attention

multiple attention heads that have different multiple attention modules whose outputs are concatenated at the end



