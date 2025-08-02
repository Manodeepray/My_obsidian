# interview questions
Perfect — here’s an **all-in-one Post-Processing & Scaling Interview Drill Day**. This is packed with **questions only**, grouped by topic: quantization, FSDP, mixed precision, training/inference optimization, scaling laws, MoE, and attention architectures.

---

## 🧮 **Quantization (INT8, FP16, PTQ, QAT)**

1. What is quantization and why is it used in deep learning?
    
2. What’s the difference between post-training quantization (PTQ) and quantization-aware training (QAT)?
    
3. What are the tradeoffs of using INT8 vs FP16?
    
4. How does quantization affect model accuracy and inference latency?
    
5. Can all layers in a model be quantized? Why or why not?
    
6. What is dynamic quantization in PyTorch?
    
7. What are per-channel vs per-tensor quantization strategies?
    
8. How does calibration work in quantization?
    
9. Why might a Transformer architecture perform poorly under quantization?
    
10. What is activation quantization and how does it differ from weight quantization?
    

---

## 🧵 **FSDP (Fully Sharded Data Parallelism)**

1. What is FSDP and how is it different from DDP (DistributedDataParallel)?
    
2. How does FSDP help in memory efficiency?
    
3. What is parameter sharding and when is it beneficial?
    
4. How does FSDP handle optimizer state and gradients?
    
5. What are the challenges in using FSDP with mixed precision training?
    
6. Compare ZeRO stage 1/2/3 with FSDP. When would you use each?
    
7. What are FSDP “wrapping policies”? How do they work?
    
8. How does checkpointing behave differently in FSDP compared to traditional training?
    
9. What are common issues when combining FSDP with Transformer models?
    
10. What does FSDP offload to CPU and why?
    

---

## 🌀 **Mixed Precision Training (AMP / bf16 / fp16)**

1. What is mixed precision training and what are its advantages?
    
2. How do you implement AMP in PyTorch and TensorFlow?
    
3. When would you prefer bf16 over fp16?
    
4. What types of layers are sensitive to precision changes?
    
5. What is gradient scaling and why is it necessary in mixed precision?
    
6. What are the risks of numerical instability in AMP?
    
7. How does autocast work in PyTorch?
    
8. What’s the difference between inference-time mixed precision and training-time mixed precision?
    
9. What impact does mixed precision have on GPU memory and throughput?
    
10. Can mixed precision help during fine-tuning small models? Why or why not?
    

---

## ⚙️ **Training and Inference Optimization**

1. How would you reduce inference latency in a transformer-based model?
    
2. What is lazy loading and when is it useful?
    
3. How do you use TorchScript or ONNX for optimizing inference?
    
4. What is LayerDrop and how does it help with training speed?
    
5. What are common bottlenecks in training large models?
    
6. How can you optimize input pipelines to speed up training?
    
7. How do fused operations improve GPU efficiency?
    
8. How do you handle large sequence lengths during inference?
    
9. What are the best practices for batch size selection during training vs inference?
    
10. How would you deploy a model trained on A100 to an edge device?
    

---

## 📈 **Scaling Laws (Training compute vs performance)**

1. What are deep learning scaling laws?
    
2. Who proposed the Chinchilla scaling law and what does it say?
    
3. How does compute-optimal scaling differ from parameter-optimal scaling?
    
4. Why does performance often saturate with more parameters but not more data?
    
5. What are the implications of scaling laws for training budget planning?
    
6. How do token counts, model size, and FLOPs interplay in scaling laws?
    
7. What is the marginal utility of training larger models without more data?
    
8. How do scaling laws apply differently to vision and language tasks?
    
9. How do you determine when to stop pretraining based on scaling trends?
    
10. What does it mean to be “compute-optimal” in model design?
    

---

## 🧠 **MoE (Mixture of Experts)**

1. What is a Mixture of Experts model?
    
2. How does sparse MoE reduce compute cost?
    
3. How is routing done in MoE models?
    
4. What are top-k and gate functions in MoE?
    
5. How is load balancing handled in MoE training?
    
6. What are the challenges of backpropagation in MoE?
    
7. How do MoE models scale compared to dense transformers?
    
8. What is the role of auxiliary loss in MoE?
    
9. How do you combine experts in inference-time execution?
    
10. How does GShard or Switch Transformer work?
    

---

## 🧲 **Attention Architectures & Variants**

1. How does self-attention work in transformers?
    
2. What is multi-head attention and why is it beneficial?
    
3. What are common attention bottlenecks at scale?
    
4. What is Linear Attention? How does it differ from vanilla attention?
    
5. What are alternatives to softmax attention?
    
6. What is sparse attention and where is it used?
    
7. How do you deal with long-sequence attention (e.g., 4K tokens+)?
    
8. What is FlashAttention and how does it optimize attention computation?
    
9. How does cross-attention work in encoder-decoder models?
    
10. What’s the role of positional encodings in attention, and what are rotary embeddings?
    

---

## 🧪 BONUS: Rapid-Fire Mix

1. How would you fine-tune a 7B model on 8GB GPU?
    
2. What is speculative decoding and how does it accelerate inference?
    
3. How does LoRA affect model weights during training?
    
4. What is the tradeoff between batch size and learning rate?
    
5. Why does gradient checkpointing help scale models?
    
6. What’s the difference between FP16 vs INT8 inference speedup?
    
7. How does activation checkpointing differ from standard checkpointing?
    
8. Can you use MoE with quantization?
    
9. What is activation recomputation?
    
10. When is zero-shot generalization expected to fail?
    

---

Let me know if you want flashcards, answers, or mock quiz from this list!