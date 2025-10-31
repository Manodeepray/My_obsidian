

# Daniel Talk – Round 2

## New Track: OpenEnv

- OpenEnv has around **2000 environments**, which we can fully utilize.
    
- Example discussed: **2048 solution**. They asked **GPT-OSS** to create a strategy for the game.
    
- You can **customize the notebook** for different environments.
    
- one idea is to **create a universal reward function** for all environments or games, such as giving a reward for **win/loss** only.
    
- You can also **design new environments** by defining a new **OpenEnv specification** and then integrating it into the notebook.
    
- Daniel prefers not to use the **2048 environment**, and instead try an **Atari environment** — if possibe visulaize it - its cooler
    

### Schedule

- There is an **early deadline** on **Sunday, 7 PM PT**.
    
- If you want a **sticker**, you have to win.
    

---

## Reinforcement Learning (RL)

- **Goal of RL:** Encourage  good actions more and discourage  bad actions less.
    
- Example:
    
    - If the problem is `2 + 2`, and the model outputs something other than 4 (like `b` or `c`), it should receive **less penalty**.
        
    - If by chance it gets the correct answer, **massively reward it**.
        
    - If it outputs `3`, give **some reward**.
        
    - Always reward **legal actions**./formats
        
- Motto: _“Luck is all you need” / “Patience is all you need.”_
    
- In the beginning, **0 reward is common** — you have to **wait** for progress.
    
- RL works by **reducing the probability of bad answers** and **shifting the distribution** toward good ones.
    

---

## OpenAI Dev Day Notes

- They showcased a **2048 game** solved using a reinforcement learning strategy.
    
- The model was only given a **single question**, like:
    
    > “Please create a strategy to solve the 2048 game.”
    
- The focus is shifting toward **specifying the task** rather than manually collecting data.
    
- Current focus: **Creating the environment** and **designing the reward function**.
    

---

## Technical Mentions

- **GPT-OSS ultra long context training**.
    
- **Fast inference** achieved via **BF16** or **4-bit GRPO** using **Unsloth**.
    
- Refer to the **Unsloth GitHub page** for examples.
    
- Example: **Dynamic GGUF 1.58-bit inference**.
    

### What We Can Do

- Perform **benchmarking experiments**.
    
- Review the following:
    
    - **GRPO + Unsloth (Long Context)** blog post for consumer GPUs.
        
    - **Memory-efficient RL** implementations.
        
    - Example notebooks on:
        
        - Kernel generation
            
        - 2048 game
            
        - OpenEnv setup
            
    - Blog on **RL using AMD GPUs**.
        
- Use the model:
    
    `unsloth/gpt-oss-20b-BF16, load_in_4bit=False`
    
- Look into **negative reward functions** for certain conditions.
    

---

## Submission Details

- For the **OpenEnv track**, you can use OpenEnv to do **any open-ended task**.
    
- Submit before **Sunday 7 PM PT** for early evaluation.
    
- For the **Classic track**:
    
    - Benchmark the **trained 70B model**.
        
    - Focus on **improving the reward**.
        
    - Try **GPT-OSS 120B** with **quantization + QLoRA** (AMD GPU error fixed according to Daniel for GRPO Trainer).
        

### Notebook Command

`wget "https://raw.githubusercontent.com/unslothai/notebooks/refs/heads/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game_BF16.ipynb" -O "OpenEnv.ipynb"`

## Example Prompt (2048)

---



![[Pasted image 20251025232336.png]]



