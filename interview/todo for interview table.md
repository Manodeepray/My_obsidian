# timetable
Absolutely — here's a **high-intensity 3-day crash schedule** to help you get interview-ready **as a Research Assistant for time-series deep learning**, optimized for maximum outcome under tight time:

---
Challenges:
1. Convert nf4 / BnB 4bit to Triton
2. Make FSDP2 work with QLoRA
3. Remove graph breaks in torch.compile
4. Memory Efficient Backprop

quantization algorithm

---

## ✅ OVERALL GOAL (LLM-Focused)

By the end of **Day 3**, you’ll have:

- ✅ A working **PyTorch LLM (decoder-only Transformer)**
    
- ✅ Comparison with **baseline (e.g., GPT2-small, distilled model)**
    
- ✅ One **reproduced LLM paper** (e.g., LoRA, QLoRA, DistilGPT2)
    
- ✅ Mini **MLOps pipeline** (logging, checkpoints, CLI, profiling)
    
- ✅ Ready for **technical + behavioral interviews**
    
- ✅ (Optional) **GitHub-ready project or blog-style notebook**
    

---

## ⚡️ Day 1: Deep Dive + LLM Foundations

| **Time**     | **Topic**                      | **Tasks**                                                                                                                                                                                         |
| ------------ | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 9 AM – 1 PM  | ==Core Concepts & Baselines==  | ==- Refresh `nn.Transformer`, decoder-only architecture - Understand tokenization, causal attention, and autoregressive generation - Load and tokenize dataset (e.g., WikiText-2)==               |
| 2 PM – 6 PM  | ==LLM from Scratch (PyTorch)== | ==- Implement minimal GPT2-style Transformer - Use sliding window for sequence generation - Train on toy dataset, plot loss, samples==                                                            |
| 8 PM – 10 PM | ==GitHub + Blog Inspiration==  | ==- Explore: [nanoGPT](https://github.com/karpathy/nanoGPT), [minGPT](https://github.com/karpathy/minGPT) - Read blog: [A GPT in 60 lines of code](https://jaykmody.com/blog/gpt-from-scratch/)== |
|              |                                |                                                                                                                                                                                                   |
- [x] check nanogpt from nn/nn/llm/nanogpt

## ⚡️ Day 2: Paper Reproduction + MLOps + System Setup

| **Time**     | **Topic**             | **Tasks**                                                                                                                                                                           |
| ------------ | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 9 AM – 1 PM  | Research Reproduction | - Pick and reproduce: [LoRA](https://arxiv.org/abs/2106.09685), [DistilGPT2](https://huggingface.co/distilgpt2), [QLoRA](https://arxiv.org/abs/2305.14314) - Modify on your dataset |
| 2 PM – 6 PM  | MLOps + Cloud + AMP   | - Add experiment tracking (Weights & Biases / MLflow) - Save checkpoints via `torch.save()` - Add AMP (`torch.cuda.amp`) - Create Dockerfile - Run model on EC2 / Colab Pro GPU     |
| 8 PM – 10 PM | Interview Topics      | - Prep Qs: What is LoRA? When use QLoRA vs Full FT? What is RLHF? - Update resume: add LLM training project                                                                         |

---

## ⚡️ Day 3: Scaling Laws + Inference + Interview Polishing

| **Time**     | **Topic**                        | **Tasks**                                                                                                                                         |
| ------------ | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 9 AM – 1 PM  | Scaling Laws + Sparse Experts    | - Read: Chinchilla, Gopher scaling laws - Understand Switch Transformer, MoE routing - Plot toy scaling curve - Answer: "Why sparse experts?"     |
| 2 PM – 6 PM  | Inference Optimization + CLI     | - Add `argparse` CLI: `python train.py --model gpt2` - Export model with TorchScript - Measure latency with `torch.profiler`, optimize batch size |
| 8 PM – 10 PM | Final Polish + Interview Dry Run | - Mock Q&A: “Explain your model”, “Fine-tune on low memory?”, “LoRA vs PEFT” - Polish GitHub: README, inference notebook, Colab link, model cards |

---

## ⚡️ Day 4 (Optional Bonus): LLM Fine-Tuning Methods

| **Time**     | **Topic**                      | **Tasks**                                                                                                                                      |
| ------------ | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| 9 AM – 1 PM  | PEFT/LoRA/QLoRA Refresher      | - Fine-tune LLaMA or GPT2 using PEFT/QLoRA on Alpaca dataset - Understand adapter weights and delta checkpoints                                |
| 2 PM – 6 PM  | Distillation Project           | - Implement distillation: Train a smaller GPT2 using the outputs of a larger model - Compare logits, loss curves                               |
| 8 PM – 10 PM | GitHub Polish + Interview Prep | - Push project to GitHub - Add model metrics, inference demo - Prepare STAR answers for “LLM infra”, “model scaling”, “inference optimization” |
https://towardsai.net/p/artificial-intelligence/how-i-built-my-own-custom-8-bit-quantizer-from-scratch-a-step-by-step-guide-using-pytorch
---

## 📦 Final Deliverables

|Item|Deliverable|
|---|---|
|✅ Project Code|GitHub repo: GPT2-clone vs baseline vs LoRA/Distilled version|
|✅ Paper Reproduction|Repro notebook: LoRA/DistilGPT2/QLoRA|
|✅ MLOps|Logging, CLI, GPU usage, TorchScript export|
|✅ Resume Update|Add 1-line: “Built and fine-tuned decoder-only LLM with MLOps + benchmarking”|
|✅ Interview Readiness|STAR-format answers, diagram-based explanations, scaling trade-offs|

---

Let me know if you want:

- A **starter repo for GPT2-style LLM** from scratch
    
- A **LoRA/PEFT fine-tuning script**
    
- An **interview Q&A cheatsheet** for LLM infra & deployment
    

You're ready to build something impressive — and it’ll stand out if you polish it well!
---

Let me know if you want:

- A project starter template (PyTorch time-series boilerplate)
    
- A mock interview session (technical or behavioral)
    
- Help creating the GitHub structure
    

You’re in a great place — 3 days is intense but **totally doable with focus and smart execution**.



