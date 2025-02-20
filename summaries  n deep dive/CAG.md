
### Cache Augmented Generation

Advantages
- its fast
- more reliable
- quite simple

basic working principle
- preload knowledge in kv cache of LLM instead of a retrieval system 
- to do so create a model with kv cache
- using a method kv encode to save the text in the models kv cache
 
limitations
- limited by context window
- super efficient but costlier
- not finding the relevant bits

when you should consider it
- when the entire knowledge base fits within the model's context window
- when you need extremely fast responses
- when information doesnt change frequently
- when increased overall costs are ok for your use case
- when you dont see relevant decreases in output quality with respect to RAG
- when you hace sufficient GPU memory, if using API just set cache feature to True and forget about GPU memory