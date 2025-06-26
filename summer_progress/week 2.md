					

## other courses




## LLMs


### 📅 Week 2: Day-Wise Timeline – Training & Optimization

| **Day**   | **Focus Area**                      | **Resources & Topics Covered**                                                                                                                                                                            | **Outcomes / Experiments**                                                                                 |     |
| --------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | --- |
| **Day 1** | Transformer Training Fundamentals   | ==🎥 [Training a Transformer from Scratch (Umar Jamil)](https://www.youtube.com/watch?v=ISNdQcPhsts)==  <br>==📖 Review model architecture + training loop==  <br>📖 Study _Scaling Laws_ (Kaplan et al.) | ✅ ==Refreshed Transformer training pipeline==  <br>✅ Took notes on compute vs performance scaling insights |     |
| **Day 2** | Data Pipeline for LLMs              | ==📖 Read about data **sharding** and **streaming**==<br>📖 ==Explored Hugging Face `datasets` and `WebDataset` formats==  <br>📌 Skimmed data preprocessing tutorials for large corpora                  | ==✅ Built a streaming dataset loader==<br>✅ Preprocessed sample dataset for next day's training            |     |
| **Day 3** | Training Optimizations              | 📖 Study:  <br>• **Gradient Checkpointing**  <br>• **Mixed Precision Training** (FP16/BF16)  <br>• **ZeRO, FSDP, DeepSpeed**  <br>📌 Read docs/tutorials from Microsoft and Hugging Face                  | ✅ Enabled mixed precision in PyTorch  <br>✅ Tested DeepSpeed config on local experiment                    |     |
| **Day 4** | Efficient Batching Techniques       | 📖 Study batching techniques:  <br>• **Prefetching**, **Bucketing**, **Packing**  <br>📌 Watched related Stanford CS25 snippets  <br>📌 Skimmed 🤗 `Trainer` batching logic                               | ✅ Implemented smart batching in tokenizer pipeline  <br>✅ Benchmarked different strategies                 |     |
| **Day 5** | Model Training: TinyStories Dataset | 🛠️ Train GPT-like model on **TinyStories**  <br>📖 Review GPT-1 architecture for model choice and config                                                                                                 | ✅ Successfully trained GPT-style model  <br>✅ Logged training curves and attention patterns                |     |
| **Day 6** | Fine-Tuning with DeepSpeed/FSDP     | 📖 Fine-tuning strategy for small datasets  <br>📌 Setup DeepSpeed with offload + FSDP  <br>📌 Traced training memory usage and GPU utilization                                                           | ✅ Fine-tuned TinyStories model on a small custom dataset using DeepSpeed                                   |     |
| **Day 7** | LoRA, QLoRA, Quantization           | 📖 Read:  <br>• _LoRA: Low-Rank Adaptation_  <br>• _GPTQ: Quantization of LLMs_  <br>📌 Try integrating LoRA into PyTorch training loop  <br>📌 Notes from Stanford CS25 & Hugging Face blogs             | ✅ Prototype LoRA-based fine-tuning loop  <br>✅ Wrote summary + comparison of LoRA, QLoRA, GPTQ methods     |     |








## dl + maths

| **Day 8**  | 📌 What is DLOps?                                            | - Differences from MLOps - Real-world applications and role in AI lifecycle                                                              |
| ---------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Day 9**  | 📌 Case Studies in DLOps                                     | - Study Uber Michelangelo, Meta FBLearner - Understand real industry pipelines                                                           |
| **Day 10** | 📌 DL Math - Linear Algebra I                                | - Scalars, vectors, matrices, tensors - Operations: addition, dot product, transpose - Hands-on: NumPy matrix operations - HF Agents vid |
| **Day 11** | 📌 Cloud & On-Prem Infrastructure                            | - GPU vs TPU vs CPU overview - AWS SageMaker, GCP Vertex AI, Azure ML overview - HF Agents vid                                           |
| **Day 12** | 📌 DL Math - Linear Algebra II                               | - Eigenvalues, eigenvectors, SVD, norms - PCA and backprop intuition - HF Agents vid                                                     |
| **Day 13** | 📌 Containerization & Virtualization Basics (Part 1)         | - Docker: images, containers, volumes, networking - Write Dockerfile, set up containers - Set up NVIDIA CUDA & cuDNN                     |
| **Day 14** | 📌 Containerization & Virtualization Basics (Part 2) + Recap | - Dockerize a DL model - Deploy on Kubernetes (GKE, EKS, AKS) - Full recap and catch-up - HF Agents vid                                  |
## 🗓️ Week 1: Linear Algebra + Analytic Geometry

|Day|Topic|Chapter Sections|Time (hrs)|
|---|---|---|---|
|Day 1|Intro + Systems of Equations|2.1 – 2.3|2.5|
|Day 2|Solving + Vector Spaces|2.4 – 2.5|2.5|
|Day 3|Independence + Basis + Rank|2.6 – 2.7|2.5|
|Day 4|Linear Mappings + Affine Spaces|2.8 – 2.9|2.5|
|Day 5|Exercises (Linear Algebra)|–|2.5|
|Day 6|Norms, Inner Products, Distances|3.1 – 3.3|2.5|
|Day 7|Catch-up + Light Review|Buffer / Rest|1–2|
