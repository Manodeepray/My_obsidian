
# LLM

| **Day**    | **Tasks**                                                                                                                                                                                                                                                                                                                                                                 |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Day 1 ** | ==📖 Read **"Attention Is All You Need" (Vaswani et al.)** carefully.==  <br>📖 ==Go through **"The Illustrated Transformer" (Jay Alammar)** to visually understand the architecture.==                                                                                                                                                                                   |
| **Day 2 ** | ==📖 Study **BERT** architecture (focus on Pre-training objectives, Masked LM + Next Sentence Prediction).==  <br>📖 ==Begin studying **GPT architecture** (attention masks, autoregression).==<br>                                                                                                                                                                       |
| **Day 3 ** | ==📖 Study **LLaMA architecture** and its differences from GPT and BERT.==  <br>==🛠️ Start **implementing self-attention from scratch** in PyTorch (no full Transformer yet, just self-attention module).==<br>==- stanford language modelling lec 1==                                                                                                                   |
| **Day 4 ** | 📖 ==Study **Tokenization techniques**:==  <br>==- Byte-Pair Encoding (BPE)==  <br>==- WordPiece==  <br>- ==SentencePiece== (read original papers or tutorials)  <br>==📖 Skim Hugging Face Course (Tokenization chapter).==<br>==- stanford language modelling lec 1==                                                                                                   |
| **Day 5 ** | ==🛠️ Experiment:==  <br>==- Build tokenizers using `tiktoken` and 🤗 `tokenizers`.==  <br>==- **Train your own custom tokenizer** on a small dataset.  [cs336](https://youtu.be/SQ3fZ1sAqXI?si=7nmkSUDjAXtec_Do&t=4627)==<br>==📌 Watch Andrej Karpathy’s "GPT from Zero to Hero" lecture (at least half).==                                                             |
| **Day 6 ** | ==✅ **Implement a Mini-Transformer (small GPT-2)** from scratch in PyTorch (use previous self-attention code).==  <br>==📖 Watch the remaining Karpathy lecture + optionally Fast.ai NLP transformer lessons.==                                                                                                                                                           |
| **Day 7 ** | ==🔁 **Pending Task Completion** (catch up on anything missed).==  <br>==- stanford language modelling lec next==<br>==🔁 **Full Recap**:==  <br>==- Summarize key learnings from Attention paper, BERT, GPT, LLaMA.==  <br>==- Review code implementations (self-attention, MiniTransformer).==  <br>==- Reflect: Where are you still confused? Make notes for Week 2.== |


# DL + Maths
Got it! Here's your updated schedule, incorporating the **7-day PyTorch deep learning basics** from your Udemy course **before** starting the original DLOps plan (so DLOps now begins from Day 8):

---

| **Day**   | **Topics**                     | **Details**                                                                                 |
| --------- | ------------------------------ | ------------------------------------------------------------------------------------------- |
| ==**Day 1**== | ==🔥 PyTorch Basics – Part 1== | ==- Tensors, basic tensor ops, autograd - Numpy vs PyTorch - Scalars, vectors, and shapes== |
| ==**Day 2**== | ==🔥 PyTorch Basics – Part 2==     | ==- `nn.Module`, forward pass, loss functions - Training loops==                                |
| ==**Day 3**== | ==🔥 PyTorch Basics – Part 3==     | ==- Optimizers (`SGD`, `Adam`) - DataLoaders and batching - Model evaluation==                  |
| ==**Day 4**== | ==🔥 PyTorch Basics – Part 4==     | ==- GPU vs CPU, `.cuda()` vs `.to()` - Saving/loading models==                                  |
| ==**Day 5**== | ==🔥 PyTorch Basics – Part 5==     | ==- Custom datasets and transforms - Image loading using `torchvision`==                        |
| ==**Day 6**== | ==🔥 PyTorch Basics – Part 6==     | ==- `nn.Sequential` and modularity - Weight init, learning rate schedulers==                    |
| ==**Day 7**== | ==🔥 PyTorch Basics – Part 7==     | ==- TensorBoard integration - Debugging tips and common pitfalls==                              |
|           |                                |                                                                                             |


---

Let me know if you'd like to extend this for the next weeks (e.g. offloading, MoE models, Deepspeed, etc.).