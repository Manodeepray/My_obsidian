
 we introduce RECIPE, a RetriEval-augmented ContInuous Prompt lEarning method, to boost editing efficacy and inference efficiency in lifelong learning.
  RECIPE first converts knowledge statements into short and informative continuous prompts, prefixed to the LLM’s input query embedding, to efficiently refine the response grounded on the knowledge. 
  It further integrates the Knowledge Sentinel (KS) that acts as an intermediary to calculate a dynamic threshold, determining whether the retrieval repository contains relevant knowledge. Our retriever and prompt encoder are jointly trained to achieve editing properties, i.e., reliability, generality, and locality. In our experiments, RECIPE is assessed extensively across multiple LLMs and editing datasets, where it achieves superior editing performance. RECIPE also demonstrates its capability to maintain the overall performance of LLMs alongside showcasing fast editing and inference speed. 
 
 
 Previous efforts in model editing have primarily focused on single and batch edits. Notable examples include ROME (Meng et al., 2022), MEND (Mitchell et al., 2022), and MEMIT (Meng et al., 2023), which achieve edits by applying offsets to part of the model’s parameters. However, in the real world, LLMs frequently require continuous knowledge updates to stay abreast of emerging knowledge. Thus, the concept of lifelong editing has been introduced (Hartvigsen et al., 2022).

Retrieval-based methods separate knowledge from the model

