## Beginner to Intermediate Projects

iilya sutskev 30 papers
yash twitters models

### 1. English to Newari Translation using Multi-Technique Finetuning

**Title:** Multistrategy Translation for Low-Resource Language Pairs (English → Newari)

- Implemented LoRA, QLoRA, and full finetuning on mBART/BLOOMZ for English-Newari translation.
    
- Evaluated using BLEU, chrF++, TER, and conducted error pattern analysis across domains.
    
- Deployed with a Streamlit app and added a mini MLOps pipeline (logging, checkpointing, profiling, CLI). hf spaces
    

**Keywords:** NMT, LoRA, QLoRA, Low-Resource Translation, chrF++, HuggingFace

---

### 2. Paper Reimplementations: LoRA / QLoRA / CoT / Self-Instruct

**Includes:**

- LoRA, LoRA8, and QLoRA with GPT2, Mistral, T5; integrated with HuggingFace PEFT and eval harness.
    
- Self-Instruct and Evol-Instruct pipelines for instruction tuning with GPT-4 filtering.
    
- Chain-of-Thought prompting on GSM8K, StrategyQA, SVAMP; compared CoT, direct answer, distilled CoT.
    

**Keywords:** LoRA, Self-Instruct, CoT, Instruction Finetuning

---

## Intermediate to Advanced Projects

### 3. Memory-Efficient Backpropagation

**Title:** Advanced Backpropagation Techniques for Large Transformer Training

- Implemented memory-efficient backward pass per PyTorch deep dive.
    
- Used manual checkpointing, custom autograd, and recomputation.
    
- Applied to HuggingFace Transformers with custom training loop.
    
jit?

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

## Advanced & Research-Level Projects

### 9. Trajectory Generation with Diffusion Transformer (DiT-style)

- Integrated Diffusion Transformer (DiT) policy head with multimodal embeddings.
    
- Predicted joint-space trajectories with classifier-free guidance.
    
- Trained using MSE loss, cosine noise schedule, sinusoidal timestep embeddings.
    
- Evaluated diversity, robustness under partial observations.
    

**Tools:** PyTorch, Diffusers, MuJoCo, Isaac Gym

---

### 10. Low-Level Torque and Motion Control

- Real-time torque/velocity control for Franka and UR5 via FCI, MuJoCo, Isaac Gym.
    
- Mapped trajectories to torques with PID controller.
    
- Language-guided tasks, feedback logs, and safety checks implemented.
    

**Environments:** Franka Emika Panda, UR5, MuJoCo, Isaac Gym

---

### 11. Advanced Paper Reproductions (2023–2025)

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
    

---

## Optional

Let me know if you'd like:

- A LaTeX ".tex" export for resumes
    
- GitHub README.md template
    
- Notion portfolio layout