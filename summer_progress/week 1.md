
# LLM

| **Day**    | **Tasks**                                                                                                                                                                                                                                                                                                                                         |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Day 1 ** | ==📖 Read **"Attention Is All You Need" (Vaswani et al.)** carefully.==  <br>📖 ==Go through **"The Illustrated Transformer" (Jay Alammar)** to visually understand the architecture.==                                                                                                                                                           |
| **Day 2 ** | ==📖 Study **BERT** architecture (focus on Pre-training objectives, Masked LM + Next Sentence Prediction).==  <br>📖 ==Begin studying **GPT architecture** (attention masks, autoregression).==<br>                                                                                                                                               |
| **Day 3 ** | 📖 Study **LLaMA architecture** and its differences from GPT and BERT.  <br>==🛠️ Start **implementing self-attention from scratch** in PyTorch (no full Transformer yet, just self-attention module).==<br>- stanford language modelling lec 1                                                                                                   |
| **Day 4 ** | 📖 Study **Tokenization techniques**:  <br>- Byte-Pair Encoding (BPE)  <br>- WordPiece  <br>- SentencePiece (read original papers or tutorials)  <br>📖 Skim Hugging Face Course (Tokenization chapter).<br>- stanford language modelling lec 2                                                                                                   |
| **Day 5 ** | 🛠️ Experiment:  <br>- Build tokenizers using `tiktoken` and 🤗 `tokenizers`.  <br>- **Train your own custom tokenizer** on a small dataset.  <br>📌 Watch Andrej Karpathy’s "GPT from Zero to Hero" lecture (at least half).                                                                                                                     |
| **Day 6 ** | ✅ **Implement a Mini-Transformer (small GPT-2)** from scratch in PyTorch (use previous self-attention code).  <br>📖 Watch the remaining Karpathy lecture + optionally Fast.ai NLP transformer lessons.<br>- stanford language modelling lec next                                                                                                 |
| **Day 7 ** | 🔁 **Pending Task Completion** (catch up on anything missed).  <br>- stanford language modelling lec next<br>🔁 **Full Recap**:  <br>- Summarize key learnings from Attention paper, BERT, GPT, LLaMA.  <br>- Review code implementations (self-attention, MiniTransformer).  <br>- Reflect: Where are you still confused? Make notes for Week 2. |


# DSA

|                              |                  |                              |                                                                                |
| ---------------------------- | ---------------- | ---------------------------- | ------------------------------------------------------------------------------ |
| **Day**                      | **Topic**        | **Sub-Topics / Problems**    | **Resources**                                                                  |
| **Week 1: Arrays & Strings** |                  |                              |                                                                                |
| ==1==                            | ==Arrays Basics==    | ==Traversal, Declaration==       | ==[GFG Arrays](https://www.geeksforgeeks.org/arrays-in-data-structure/)==          |
| 2                            | Array Operations | Insertion, Deletion          | [GFG Array Operations](https://www.geeksforgeeks.org/array-data-structure/)    |
| 3                            | Array Problems   | Easy-Medium Practice         | [GFG Practice](https://practice.geeksforgeeks.org/explore?page=1&topic=Arrays) |
| 4                            | Strings Basics   | Declaration, Functions       | [GFG Strings](https://www.geeksforgeeks.org/cpp-strings/)                      |
| 5                            | String Problems  | Palindrome, Reverse, Anagram | [GFG String Problems](https://www.geeksforgeeks.org/string-data-structure/)    |
| 6                            | Mixed Practice   | Arrays + Strings             | [LeetCode Arrays](https://leetcode.com/tag/array/)                             |
| 7                            | **Revision Day** | Quiz & Recap                 | Self-Review                                                                    |

# DL + Maths
Got it! Here's your updated schedule, incorporating the **7-day PyTorch deep learning basics** from your Udemy course **before** starting the original DLOps plan (so DLOps now begins from Day 8):

---

| **Day**    | **Topics**                                                   | **Details**                                                                                                                              |
| ---------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Day 1**  | ==🔥 PyTorch Basics – Part 1==                               | ==- Tensors, basic tensor ops, autograd - Numpy vs PyTorch - Scalars, vectors, and shapes==                                              |
| **Day 2**  | 🔥 PyTorch Basics – Part 2                                   | - `nn.Module`, forward pass, loss functions - Training loops                                                                             |
| **Day 3**  | 🔥 PyTorch Basics – Part 3                                   | - Optimizers (`SGD`, `Adam`) - DataLoaders and batching - Model evaluation                                                               |
| **Day 4**  | 🔥 PyTorch Basics – Part 4                                   | - GPU vs CPU, `.cuda()` vs `.to()` - Saving/loading models                                                                               |
| **Day 5**  | 🔥 PyTorch Basics – Part 5                                   | - Custom datasets and transforms - Image loading using `torchvision`                                                                     |
| **Day 6**  | 🔥 PyTorch Basics – Part 6                                   | - `nn.Sequential` and modularity - Weight init, learning rate schedulers                                                                 |
| **Day 7**  | 🔥 PyTorch Basics – Part 7                                   | - TensorBoard integration - Debugging tips and common pitfalls                                                                           |
| **Day 8**  | 📌 What is DLOps?                                            | - Differences from MLOps - Real-world applications and role in AI lifecycle                                                              |
| **Day 9**  | 📌 Case Studies in DLOps                                     | - Study Uber Michelangelo, Meta FBLearner - Understand real industry pipelines                                                           |
| **Day 10** | 📌 DL Math - Linear Algebra I                                | - Scalars, vectors, matrices, tensors - Operations: addition, dot product, transpose - Hands-on: NumPy matrix operations - HF Agents vid |
| **Day 11** | 📌 Cloud & On-Prem Infrastructure                            | - GPU vs TPU vs CPU overview - AWS SageMaker, GCP Vertex AI, Azure ML overview - HF Agents vid                                           |
| **Day 12** | 📌 DL Math - Linear Algebra II                               | - Eigenvalues, eigenvectors, SVD, norms - PCA and backprop intuition - HF Agents vid                                                     |
| **Day 13** | 📌 Containerization & Virtualization Basics (Part 1)         | - Docker: images, containers, volumes, networking - Write Dockerfile, set up containers - Set up NVIDIA CUDA & cuDNN                     |
| **Day 14** | 📌 Containerization & Virtualization Basics (Part 2) + Recap | - Dockerize a DL model - Deploy on Kubernetes (GKE, EKS, AKS) - Full recap and catch-up - HF Agents vid                                  |

---

Let me know if you'd like to extend this for the next weeks (e.g. offloading, MoE models, Deepspeed, etc.).