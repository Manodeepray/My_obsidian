[[AI security roadmap]]
# Beginner to Intermediate Projects

### 1. English to Newari Translation using Multi-Technique Finetuning
[[1.English to Newari Translation using Multi-Technique Finetuning]]

---

### 2. Paper Reimplementations: LoRA / QLoRA / CoT / Self-Instruct

[[2 Paper Reimplementations LoRA -QLoRA - CoT - Self-Instruct]]

---

# Intermediate to Advanced Projects

### 3. Memory-Efficient Backpropagation

**Title:** Advanced Backpropagation Techniques for Large Transformer Training

- Implemented memory-efficient backward pass per PyTorch deep dive.
    
- Used manual checkpointing, custom autograd, and recomputation.
    
- Applied to HuggingFace Transformers with custom training loop.
    
jit?
https://www.youtube.com/results?search_query=autograd
https://www.reddit.com/r/learnmachinelearning/comments/1epgitp/autograd_tutorial_on_gradient_calculation_in/
https://docs.pytorch.org/docs/2.9/notes/autograd.html
https://aschrein.github.io/jekyll/update/2025/08/23/compute_graph.html
https://arxiv.org/abs/2503.13795
https://github.com/atalw/fromthetensor
**Keywords:** Backpropagation, Memory Optimization, Custom Autograd

---

### 4. Paged Optimizer + Distributed Engine

**Title:** Scaling Paged Optimizer Training with Distributed Transformers

- Integrated PagedAdamW with DDP and FSDP.
    
- Created distributed engine for DistilGPT2 and custom LLMs.
    
- Benchmarked against AdamW for throughput, memory, convergence.
    

**Keywords:** Paged Optimizer, FSDP, DDP, Transformers

---

### 5. GRPO/DAPO Reasoning Distillation

**Title:** Distilling Chain-of-Thought Reasoning into Student Models

- Reproduced GRPO and DAPO pipelines.
    
- Applied to GSM8K, StrategyQA, ARC.
    
- Used CoT + distillation with DistilBERT, TinyLLMs.
    

**Keywords:** Reasoning Distillation, GRPO, DAPO, Student Models

---

### 6. Distributed Training & Inference with Core PyTorch

**Title:** End-to-End Distributed Training & Serving with Native PyTorch + Triton

- Built pipeline using DDP, FSDP, TorchDynamo, torch.compile, TorchServe.
    
- Converted NF4/BNB 4-bit models to Triton-compatible kernels.
    
- Manual integration of profiling, benchmarking, CLI tools.
    

**Keywords:** PyTorch, FSDP, TorchDynamo, Triton, TorchServe

---

### 7. LLM + RL Planning Agent over Knowledge Graphs

**Title:** LLM-RL Agent for Hierarchical Reasoning over Symbolic Knowledge Graphs

- Combined LLMs with PPO/DQN to reason over graphs.
    
- Prompted with long-horizon goals, CoT reasoning, reward shaping.
    
- Tested on MiniGrid, KG games, and simulations.
    

**Keywords:** LLM-RL, Symbolic Reasoning, PPO, Planning

---

### 8. Finetuning LLaMA2-70B [[70 b]]

**Title:** Finetuning of 70B Parameter Instruction Models with Efficient Memory Techniques

- Used FSDP, activation checkpointing, gradient offloading, paged optimizer.
    
- Benchmarked with MT-Bench, AlpacaEval, QA tasks.
    
- Served via Triton Inference Server with quantized weights.
    

**Keywords:** 70B LLMs, LLaMA2, FSDP, Triton

---

# Advanced & Research-Level Projects

iilya sutskev 30 papers
https://aman.ai/primers/ai/top-30-papers/
yash twitters models
### 9. Trajectory Generation with Diffusion Transformer (DiT-style)

- Integrated Diffusion Transformer (DiT) policy head with multimodal embeddings.
    
- Predicted joint-space trajectories with classifier-free guidance.
    
- Trained using MSE loss, cosine noise schedule, sinusoidal timestep embeddings.
    
- Evaluated diversity, robustness under partial observations.
    

**Tools:** PyTorch, Diffusers, MuJoCo, Isaac Gym

---
### 10. mechanistic interpretability

https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/edit?tab=t.knytn7x826kv#heading=h.ut8va0qotfbj
### 11. Low-Level Torque and Motion Control

- Real-time torque/velocity control for Franka and UR5 via FCI, MuJoCo, Isaac Gym.
    
- Mapped trajectories to torques with PID controller.
    
- Language-guided tasks, feedback logs, and safety checks implemented.
    

**Environments:** Franka Emika Panda, UR5, MuJoCo, Isaac Gym

---
### 12. anti deepfake service


image anti deepfake services
voice anti deepfake services


--- 
### 13. Project astra recreate


---
### 14. Advanced Paper Reproductions (2023–2025)

**Includes:**

- DreamerV3: World models with latent planning.
    
- Hierarchical Reasoning Model (Wang et al., 2025): Subgoal-based QA.
    
- DeepSeek V2/V3: MoE layers, instruction decoupling, routing loss.
    
- Toolformer: API-calling LLMs with dynamic tool-use.
    
- RecurrentGPT: Recurrent memory in transformers.
    
- Kosmos-2: Vision-language model with grounding.
    
- FlashAttention v2: Efficient attention kernels.
    
- Diffusion Transformers (DiT, SDXL): For generative image modeling.
    

**Keywords:** Diffusion, MoE, Vision-Language, FlashAttention, World Models

---

## Tooling & Infrastructure (Used Across Projects)

- **Training:** HuggingFace Accelerate, DeepSpeed, FSDP
    
- **MLOps:** Weights & Biases, PyTorch Profiler, CLI tools
    
- **Deployment:** TorchServe, Triton, ONNX
    
- **Prompting:** LangChain, ReAct, AutoGPT
    







# liquid ai
## 🧱 1️⃣ One-Month Project — “Scalable Multi-GPU Inference Server with Dynamic Batching & Quantization”

### 🎯 Goal

Build a **production-grade inference server** that serves transformer models (LLMs or Vision Transformers) efficiently with:

- **Dynamic batching**
    
- **Ragged sequence support**
    
- **Quantization**
    
- **KV-cache management**
    
- **Multi-GPU load balancing**
    

Essentially, you’ll implement a _mini TensorRT/DeepSpeed-Inference_–style system for your own model.

---

### 💡 Concept

You’ll design a modular inference stack capable of:

- Handling _requests of variable sequence lengths_ efficiently (ragged batching)
    
- Managing _KV-cache across requests_ for speedups in LLM decoding
    
- Using _quantized weights_ to reduce GPU memory load
    
- Distributing requests across _multiple GPUs_ based on real-time load
    
- Benchmarking and visualizing performance metrics (latency, throughput, GPU utilization)
    

---

### ⚙️ Architecture Overview

`User Request → API Gateway → Scheduler → Dynamic Batch Builder →  GPU Workers (Quantized Model + KV Cache) → Output Stream`

---

### 🧩 Components Breakdown

1. **Model Backend:**
    
    - Use a transformer model (e.g., LLaMA-3–instruct, Mistral-7B, or ViT).
        
    - Implement quantization using:
        
        - `bitsandbytes` (INT8 / INT4)
            
        - `torchao` (FP8 quantization)
            
    - Save both quantized and full-precision checkpoints for accuracy comparison.
        
2. **Dynamic Batching:**
    
    - Implement a _batch scheduler_ that groups requests with similar sequence lengths.
        
    - Use **ragged batching** (via padding + attention mask) for efficient throughput.
        
3. **KV-Cache Management:**
    
    - Store key-value caches for each session.
        
    - Reuse cached tokens to minimize recomputation during autoregressive decoding.
        
4. **Load Balancer:**
    
    - Implement a basic load balancer using Python multiprocessing or gRPC.
        
    - Dispatch inference jobs dynamically to GPUs based on utilization (use `torch.cuda.memory_allocated()` to monitor).
        
5. **Performance Dashboard:**
    
    - Create a small **Streamlit** or **Grafana** dashboard to visualize:
        
        - Average latency
            
        - Throughput (req/sec)
            
        - GPU memory usage
            
        - Quantization impact on accuracy
            
6. **Benchmarking:**
    
    - Compare:
        
        - Quantized vs full-precision model performance
            
        - Single-GPU vs multi-GPU latency/throughput
            
        - Static vs dynamic batching
            

---

### 🚀 Deliverables

- GitHub repo: `multi-gpu-inference-server`
    
- Streamlit dashboard for metrics visualization
    
- Technical report comparing optimization techniques
    
- Demo notebook: “From naive inference to production-grade serving”
    

---

### 🧠 Skills Demonstrated

- Multi-GPU inference design
    
- Ragged batching and KV-cache handling
    
- Quantization tradeoffs (FP8/INT4)
    
- Profiling and optimizing model serving
    
- Systems-level thinking
    

---

## ⚡ 2️⃣ One-Week Project — “LLM Quantization & Inference Optimization Benchmark”

### 🎯 Goal

Perform a **comparative study** of quantization and optimization strategies for a transformer model, and benchmark their effect on speed, memory, and accuracy.

This is a **compact but impactful** project that shows your understanding of the inference pipeline and performance trade-offs.

---

### 💡 Concept

Benchmark the same model (say, `Mistral-7B` or `OPT-1.3B`) under different inference setups:

|Setup|Quantization|GPU Mode|KV Cache|Dynamic Batching|
|---|---|---|---|---|
|Baseline|FP16|Single GPU|Off|Off|
|Optimized 1|INT8|Single GPU|On|Off|
|Optimized 2|INT4|Single GPU|On|On|
|Optimized 3|FP8|Multi-GPU|On|On|

You’ll collect and visualize metrics such as:

- Latency per token
    
- Throughput (req/s)
    
- GPU memory footprint
    
- Output quality degradation (e.g., BLEU/F1 drop)
    

---

### 🧩 Implementation Plan

1. **Load Model:**
    
    - Use `transformers` or `vllm` to load model.
        
    - Wrap with different quantization methods (`bitsandbytes`, `torchao`, or `auto-gptq`).
        
2. **Add Caching:**
    
    - Implement token-level KV-cache reuse for autoregressive inference.
        
3. **Enable Dynamic Batching:**
    
    - Use `vllm.Engine` or implement a simple batch scheduler.
        
4. **Run Benchmarks:**
    
    - Test each setup with increasing concurrency (1–16 requests).
        
    - Log GPU utilization and memory stats via `torch.cuda`.
        
5. **Visualize:**
    
    - Use Matplotlib or Streamlit to plot tradeoffs between speed, memory, and accuracy.
        

---

### 🚀 Deliverables

- GitHub repo: `llm-inference-benchmarks`
    
- Notebook comparing 4 inference configurations
    
- Clear graphs of latency vs accuracy vs quantization type
    
- 2-page summary: “Optimizing Transformer Inference for Speed and Efficiency”
    

---

### 🧠 Skills Demonstrated

- Quantization strategies (INT4, FP8)
    
- GPU performance profiling
    
- KV-cache and batching effects
    
- Hands-on experience with inference frameworks (`vllm`, `bitsandbytes`, `torchao`)
    
- Practical understanding of serving tradeoffs
    

---

## 🧩 Summary Table

|Duration|Project Title|Focus|Outcome|
|---|---|---|---|
|🗓️ 1 Month|**Scalable Multi-GPU Inference Server with Dynamic Batching & Quantization**|Large-scale serving, batching, caching, load balancing|Full inference stack + dashboard + metrics report|
|⚡ 1 Week|**LLM Quantization & Inference Optimization Benchmark**|Quantization & caching benchmarks|Compact comparative analysis + visualizations|
---





## 🧩 **1️⃣ One-Month Project — “Real-World Feedback Alignment for Multimodal Foundation Models”**

### 🎯 **Goal**

Build a **multimodal alignment system** that improves how a **vision-language model** (like CLIP or BLIP-2) understands _subjective human feedback_, such as aesthetics or emotional tone, rather than simple correctness.

---

### 💡 **Concept**

Most models optimize for accuracy, not preference.  
In this project, you’ll implement a **feedback-aligned fine-tuning setup** — similar to RLHF but for **multimodal data**.

You’ll:

- Collect or simulate human preference signals (like “which image caption feels more inspiring / visually accurate / engaging”),
    
- Design a **custom loss** that integrates these feedback signals,
    
- Fine-tune an open-source **vision-language model** (e.g., **BLIP-2**, **LLaVA**, or **OpenFlamingo**),
    
- Evaluate the model on both **standard benchmarks** and **custom feedback metrics**.
    

---

### ⚙️ **Architecture Overview**

`Image + Caption Dataset → Preference Collector (synthetic/human) →  Reward Model → Fine-tuned BLIP-2 → Evaluation on Aesthetic + Accuracy metrics`

---

### 🧩 **Detailed Components**

|Component|Description|
|---|---|
|**Base Model**|Use **BLIP-2** (Salesforce) or **LLaVA-1.5** — both are PyTorch-based multimodal models.|
|**Feedback Generation**|Collect synthetic ratings using CLIP similarity + language heuristics (e.g., “emotionally positive captions get higher score”).|
|**Reward Model**|Train a small MLP to predict preference scores given image-text pairs.|
|**Custom Training Setup**|Implement **Direct Preference Optimization (DPO)** or **Reward-Weighted CE Loss** to align the base model with feedback.|
|**Distributed Training**|Use `torch.distributed` or `accelerate` for multi-GPU fine-tuning.|
|**Evaluation Suite**|Measure performance on COCO captions + custom feedback metric (e.g., user preference win rate).|

---

### 🧰 **Tech Stack**

- **PyTorch**, **Transformers**, **Accelerate**
    
- **BLIP-2 / LLaVA / OpenFlamingo**
    
- **Weights & Biases** (for experiment tracking)
    
- **Streamlit** (for qualitative before/after demo)
    

---

### 🧠 **Key Deliverables**

- GitHub repo: `multimodal-feedback-alignment`
    
- Custom reward model code + loss function
    
- Report: “Improving Human Preference Alignment in Vision-Language Models”
    
- Streamlit demo: upload an image → compare base vs. aligned model captions
    

---

### 💥 **Impact**

- Demonstrates ability to design _new loss functions_ and _feedback systems_
    
- Works beyond LLMs (vision-language domain)
    
- Shows autonomy, research direction, and production-ready prototyping
    

---

## ⚡ **2️⃣ One-Week Project — “Custom Objective Playground: Fine-Tuning CLIP on Subjective Feedback”**

### 🎯 **Goal**

Quickly prototype how **custom loss functions** affect representation learning in **CLIP**.  
You’ll train or fine-tune CLIP with new objectives based on _semantic similarity_ and _feedback signals_ rather than the standard cosine contrastive loss.

---

### 💡 **Concept**

CLIP normally learns to match images and texts by cosine similarity.  
You’ll introduce **a new contrastive loss** that accounts for _user preference scores_, like “caption relevance + creativity” or _semantic distance between similar captions_.

---

### ⚙️ **Implementation Flow**

`Image + Captions → CLIP Encoder → Similarity Matrix →  Custom Weighted Contrastive Loss → Fine-tuned CLIP → Evaluation`

---

### 🧩 **Steps**

1. Load a pretrained CLIP (OpenAI or LAION version via `open_clip`).
    
2. Use a small image-caption dataset (e.g., Flickr8k or a subset of COCO).
    
3. Generate synthetic feedback (e.g., rate captions by CLIP similarity or lexical richness).
    
4. Implement **Weighted Contrastive Loss**:
    
    L=−wi⋅log⁡esim(xi,yi)/τ∑jesim(xi,yj)/τL = -w_i \cdot \log \frac{e^{sim(x_i, y_i)/\tau}}{\sum_j e^{sim(x_i, y_j)/\tau}}L=−wi​⋅log∑j​esim(xi​,yj​)/τesim(xi​,yi​)/τ​
    
    where wiw_iwi​ = feedback-based weight per pair.
    
5. Fine-tune CLIP for 1–3 epochs, log embeddings, and visualize t-SNE of image-text pairs.
    
6. Compare results vs baseline contrastive loss visually and quantitatively.
    

---

### 🧰 **Tech Stack**

- PyTorch, open_clip
    
- Matplotlib, scikit-learn (for visualization)
    
- Weights & Biases for metric tracking
    

---

### 🧠 **Deliverables**

- GitHub repo: `clip-custom-objective`
    
- Jupyter notebook comparing baseline vs. feedback-weighted loss
    
- Visual report: t-SNE plots, CLIP similarity distributions
    
- Optional: blog-style write-up — “Tuning CLIP with Custom Feedback Objectives”
    

---

### 💥 **Impact**

- Compact project (1 week) with measurable improvement and clean visuals
    
- Demonstrates deep understanding of **PyTorch internals & loss design**
    
- Shows creativity + ability to move from **idea → prototype** fast
    

---

## 🧾 **Summary Table**

|Duration|Project Title|Focus|Core Model|Key Technical Highlights|
|---|---|---|---|---|
|🗓️ 1 Month|**Real-World Feedback Alignment for Multimodal Foundation Models**|Multimodal alignment, reward modeling, adaptive loss|**BLIP-2 / LLaVA**|Custom feedback loss, DPO fine-tuning, multimodal evaluation|
|⚡ 1 Week|**Custom Objective Playground for CLIP**|Quick-turn contrastive loss experimentation|**CLIP (OpenAI/LAION)**|Weighted contrastive loss, embedding visualization, PyTorch custom training|

---

## 💬 **How to Present on Resume**

> **Real-World Feedback Alignment for Multimodal Foundation Models** — Designed a feedback-driven fine-tuning framework for BLIP-2 that learns from aesthetic and preference signals. Implemented custom loss functions and reward modeling using PyTorch Distributed Training, achieving improved caption alignment with human preferences.

> **Custom Objective Playground for CLIP** — Built a modular PyTorch framework to experiment with new loss functions for CLIP. Introduced a feedback-weighted contrastive loss to align image-text embeddings with subjective human ratings.












# Project To-Do List

====== Beginner to Intermediate ======

- [ ] Ilya Sutskever – 30 papers  
  - [ ] Read & summarize  
  - [ ] Replicate core ideas  

- [ ] Yash’s Twitter models  
  - [ ] Explore  
  - [ ] Replicate  

- [ ] English → Newari Translation (mBART/BLOOMZ)  
  - [ ] Implement LoRA  
  - [ ] Implement QLoRA  
  - [ ] Full finetuning  
  - [ ] Evaluate (BLEU, chrF++, TER)  
  - [ ] Error analysis across domains  
  - [ ] Deploy (Streamlit + HF Spaces + mini MLOps)  

- [ ] Paper Reimplementations (LoRA / QLoRA / CoT / Self-Instruct)  
  - [ ] LoRA/QLoRA with GPT2, Mistral, T5  
  - [ ] Self-Instruct + Evol-Instruct pipelines  
  - [ ] Chain-of-Thought experiments (GSM8K, StrategyQA, SVAMP)  


====== Intermediate to Advanced ======

- [ ] Memory-Efficient Backpropagation  
  - [ ] Manual checkpointing & recomputation  
  - [ ] Custom autograd backward pass  
  - [ ] Apply to HuggingFace Transformers  

- [ ] Paged Optimizer + Distributed Engine  
  - [ ] Integrate PagedAdamW with DDP/FSDP  
  - [ ] Benchmark throughput, memory, convergence  

- [ ] GRPO/DAPO Reasoning Distillation  
  - [ ] Reproduce GRPO, DAPO pipelines  
  - [ ] Distill CoT into student models (DistilBERT, TinyLLMs)  

- [ ] Distributed Training & Inference (Core PyTorch + Triton)  
  - [ ] End-to-end DDP, FSDP, TorchDynamo  
  - [ ] Triton inference with NF4/BNB models  
  - [ ] Profiling + benchmarking CLI  

- [ ] LLM + RL Planning Agent over Knowledge Graphs  
  - [ ] PPO/DQN with symbolic graphs  
  - [ ] Long-horizon CoT reasoning  
  - [ ] Experiments: MiniGrid, KG games  

- [x] Finetuning LLaMA2-70B  
  - [x] Efficient training (FSDP + checkpointing + paged optimizer)  
  - [ ] Benchmark with MT-Bench, AlpacaEval  
  - [ ] Deploy via Triton inference  


====== Advanced & Research-Level ======

- [ ] Trajectory Generation with Diffusion Transformer  
  - [ ] DiT policy head + multimodal embeddings  
  - [ ] Train with cosine schedule, timestep embeddings  
  - [ ] Evaluate trajectory robustness & diversity  

- [ ] Low-Level Torque & Motion Control  
  - [ ] PID-based torque mapping (Franka, UR5)  
  - [ ] Simulations in MuJoCo + Isaac Gym  
  - [ ] Language-guided tasks & feedback loops  

- [ ] Advanced Paper Reproductions (2023–2025)  
  - [ ] DreamerV3 (world models)  
  - [ ] Hierarchical Reasoning (Wang et al., 2025)  
  - [ ] DeepSeek V2/V3 (MoE)  
  - [ ] Toolformer (API calling)  
  - [ ] RecurrentGPT (transformer memory)  
  - [ ] Kosmos-2 (vision-language grounding)  
  - [ ] FlashAttention v2 (kernel optimization)  
  - [ ] Diffusion Transformers (DiT, SDXL)  


====== Infra / Tooling (Ongoing) ======

- [ ] HuggingFace Accelerate, DeepSpeed, FSDP  
- [ ] Weights & Biases logging + profiling + CLI tools  
- [ ] TorchServe / Triton / ONNX deployment  
- [ ] LangChain, ReAct, AutoGPT for prompting  


