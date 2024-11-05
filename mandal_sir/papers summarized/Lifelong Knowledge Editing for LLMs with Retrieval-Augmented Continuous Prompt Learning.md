
 we introduce RECIPE, a RetriEval-augmented ContInuous Prompt Learning method, to boost editing efficacy and inference efficiency in lifelong learning.
  RECIPE first converts knowledge statements into short and informative continuous prompts, prefixed to the LLM’s input query embedding, to efficiently refine the response grounded on the knowledge. 
  It further integrates the Knowledge Sentinel (KS) that acts as an intermediary to calculate a dynamic threshold, determining whether the retrieval repository contains relevant knowledge. Our retriever and prompt encoder are jointly trained to achieve editing properties, i.e., reliability, generality, and locality. In our experiments, RECIPE is assessed extensively across multiple LLMs and editing datasets, where it achieves superior editing performance. RECIPE also demonstrates its capability to maintain the overall performance of LLMs alongside showcasing fast editing and inference speed. 
 
 
 Previous efforts in model editing have primarily focused on single and batch edits. Notable examples include ROME (Meng et al., 2022), MEND (Mitchell et al., 2022), and MEMIT (Meng et al., 2023), which achieve edits by applying offsets to part of the model’s parameters. However, in the real world, LLMs frequently require continuous knowledge updates to stay abreast of emerging knowledge. Thus, the concept of lifelong editing has been introduced (Hartvigsen et al., 2022).

Retrieval-based methods separate knowledge from the model

 This concept involves continuously updating the model's knowledge base without retraining, allowing it to correct outdated or incorrect information. In this context, the document introduces _RECIPE_, a framework for editing LLMs, which efficiently integrates retrieval methods to keep the knowledge base up-to-date. Key elements include:

1. **Knowledgeable Continuous Prompting**: Transforming knowledge into concise prompts that adjust the model's responses without needing full model retraining.
2. **Dynamic Prompt Retrieval with Knowledge Sentinel (KS)**: A dynamic mechanism that determines whether relevant information exists in the repository, improving the efficiency of knowledge retrieval and integration.




The methodology of the RECIPE framework for lifelong knowledge editing in LLMs involves several key components:

1. **Knowledge Retrieval Repository**: The repository begins empty and is progressively updated with each new knowledge statement. Each knowledge entry, represented as a continuous prompt, is associated with an embedding that captures the semantic content of the statement. This knowledge is stored as a key-value pair, where the key is the semantic representation, and the value is the continuous prompt that guides the LLM’s response.

2. **Continuous Prompt Learning**: Each knowledge statement is transformed into a continuous prompt that integrates with the LLM’s input. This prompt is prefixed to the LLM’s input embeddings, modifying the LLM's response based on the updated knowledge. This technique is rooted in prompt tuning, ensuring that LLMs can adhere to updated information efficiently without extensive retraining.

3. **Dynamic Retrieval with Knowledge Sentinel (KS)**: The Knowledge Sentinel (KS) dynamically computes a retrieval threshold to determine the relevance of stored knowledge for each query. It ensures the LLM retrieves knowledge that aligns with the query context, avoiding irrelevant information. KS relies on a contrastive learning mechanism that refines this similarity measure for different types of queries.

4. **Joint Training**: RECIPE trains the prompt encoder and retrieval system to align with three core editing properties—reliability, generality, and locality. This ensures that the model can consistently retrieve and apply knowledge updates without degrading in accuracy or general performance.

5. **Efficient Model Inference**: During inference, the retrieved continuous prompt is concatenated with the input query’s word embedding to directly modify the response on-the-fly, maintaining the LLM’s original structure and speed while applying knowledge edits dynamically  . 

This methodology helps RECIPE avoid "catastrophic forgetting" while allowing it to maintain LLM performance across numerous incremental edits.







