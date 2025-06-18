# Group Relative Policy Optimization



# DeepSeek-R1 Dissection: Understanding PPO & GRPO Without Any Prior Reinforcement Learning Knowledge
[Article](https://huggingface.co/blog/NormalUhr/grpo)

Analogy : RL training process as an elementary school exam scenario  
We ( the model being trained ) are like students trying to get high grades , the teacher who grades our exams are like the reward model ,
while our parents handling out pocket money based on or grades is similar to the critic .

- final scores alone are insufficient ,
	- Critic 
	- clip 
	- Reference model are needed

## Naive approach
**only using reward**


- Absolute scores mean higher score gets more reward
- Analogy : The teacher grades our exams and gives an “absolute score.” I typically score above 80 out of 100, while my brother often gets around 30. We then take these scores directly to our dad to ask for pocket money—meaning our “reward” (in RL terms) is simply our raw exam score. Whoever gets a higher score receives more pocket money.
- *problem* : 
	- Unfairness : if my brother improves from 30 to 60 points through a lot of hard work, he still pales in comparison to my usual 80+. He doesn’t get the encouragement he deserves.
	- Instability : Chasing higher scores myself could lead me to extreme study methods (e.g., cramming at all hours, staying up very late). Sometimes I might get 95, other times only 60, so my score—and hence the reward signal—fluctuates dramatically.
- **using absolute scores as Reward** causes large reward fluctuations, and my brother ends up feeling it’s not worth trying to improve in small increments.


### Mathematical Correspondence

$Jnaive​(θ)=E(q,o)∼(data,πθ​)​[r(o)],$

- **“optimize only the final reward,”**
- **high variance and insufficient incentives for partial improvements.**


Actor lacks a **baseline** that matches its own current level, and that hinders training efficiency.







## Introducing the Critic: Using a “Predicted Score Line” to Improve Rewards

***it’s not just about the absolute score; it’s about how much you’ve improved relative to your current level.***


- so it is decided to *- Set my “predicted score line” at 80 points and my brother’s at 40. If we exceed these lines on an exam, we get more pocket money; if not, we get very little or none.*


- higher increment from their respective selected baselines means higher reward gains and less increment mean less more reward

- encourages each person to improve on their own baseline instead of based on absolute scores

- the baselines are also **readjusted** as the children progress

### Mathematical Correspondence

- this “**score line**” is known as the **value function**, 
- $$Vψ​(s). $$
- It acts as a baseline. Our training objective evolves from “just reward” to “how much we outperform that baseline,”
- expressed by the Advantage:


					$At​=rt​−Vψ​(st​).$

- For a given state $$st$$​ and action $$ot​$$
if the actual **reward exceeds the Critic’s expectation**, it means the action **performed better than predicted**.
 
If it’s **lower**, that action **underperformed**


Formula

$$Jadv​(θ)=E[A(o)],where A(o)=r(o)−Vψ​(o).$$


By subtracting this “score line,” 
- we reduce variance in training,
- giving higher gradient signals to actions that exceed expectations and 
- penalizing those that fall short.



## Adding  Clip & Min: Keeping Learning in Check

> 🧠 **Analogy**: A single great test score shouldn’t trigger a complete change in how I study.

- Imagine I score 100 on one exam. My dad doesn’t instantly double my allowance — that would encourage erratic behavior.
    
- Instead, he limits how much he rewards me, keeping things **steady and controlled**.
    

### PPO Mechanism: Clipping

- PPO controls policy updates using a **clip function**:
    
$$    min(rt​(θ)At​, clip(rt​(θ),1−ϵ,1+ϵ)At​)$$
- Where 
$$ rt​(θ)=\frac{πθ​(ot​∣st​)}{πθold​​(ot​∣st​)​} $$
- is the ratio of the new and old policy probabilities.
    ``
- If this ratio deviates too much, it’s **clipped**, ensuring stable updates.
    

### Benefit:

Prevents overconfidence or drastic changes from a single good outcome.

---

##  Reference Model: Preventing Cheating

> 🧠 **Analogy**: Even if I score high, I can't cheat or manipulate the teacher.

- If I start using unfair tactics just to get good grades, Dad steps in.
    
- He compares my behavior to how I used to study honestly and punishes extreme shifts.
    

### PPO Mechanism: KL Penalty

- Adds a penalty if the current policy strays too far from the **reference policy**:
    
    $$-\beta \, D_{KL}(\pi_\theta \, \| \, \pi_{\text{ref}})$$
- This acts like a **"cheating detector"**, ensuring the model stays close to its original intent.
    

###  Benefit:

Maintains ethical, aligned outputs and prevents policy drift.

---

## GRPO: Removing the Critic via Multiple Simulated Averages
or
**Replacing the Value Function with “Multiple Simulated Averages**


> 🧠 **Analogy**: Instead of Dad estimating how I should do, I take 5 practice tests and use the **average score** as my benchmark.

- PPO relies on a Critic (Dad) to evaluate each action.
    
- In LLMs, the Critic is expensive and hard to train for delayed rewards (e.g., final answer score).
    

### 🔧 GRPO Mechanism:

- Replace the Critic with **multiple outputs from the old policy**.
    
- Compute the **average reward** from these outputs.
    
- Use this as the baseline:


- According to DeepSeekMath’s technical report, the GRPO objective (omitting some symbols) is:

$$JGRPO​(θ)=E[​i=1∑G​(min(\frac{πθ​(oi​)}{πθold​​(oi​)}​Ai​ , clip(\frac{πθ​(oi​)}{πθold​​(oi​)}​,1−ε,1+ε)Ai​)− β DKL​(πθ​ ∥ πref​))],​$$
    where
    $$Ai=\frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$



PPO relies on the Actor + Critic + Clip + KL penalty framework. However, in large language model (LLM) scenarios, the Critic (value function) **often needs to be as large as the Actor** to accurately evaluate states, which can be costly and sometimes impractical—especially if you only have a single final reward at the end (like a final answer quality).

Hence, **Group Relative Policy Optimization (GRPO)** steps in. Its core idea:

- **No separate value network** for the Critic,
- Sample multiple outputs from the old policy for the same question or state,
- **Treat the average reward of these outputs as the baseline**,
- Anything above average yields a “positive advantage,” anything below yields a “negative advantage.”

Meanwhile, GRPO **retains** PPO’s Clip and KL mechanisms to ensure stable, compliant updates. But drops the need for a separate value function.
    
### Benefit:

Reduces memory and compute cost while preserving stable learning.

---

## Summary: From PPO to GRPO

- **raw absolute scores** to PPO’s full mechanism (Critic, Advantage, Clip, Reference Model), and finally to **GRPO** (leveraging multiple outputs’ average scores to eliminate the value function).


- **Role of the Critic**: Provides a “reasonable expectation” for each state, significantly reducing training variance.
- **Clip & min Mechanism**: Constrains the update magnitude, preventing overreacting to a single “breakthrough” exam.
- **Reference Model**: Discourages “cheating” or extreme deviations, ensuring the policy remains reasonably aligned with its initial state.
- **Advantages of GRPO**: In large language models, it removes the need for a separate value network, reducing memory and compute costs while aligning well with “comparative” Reward Model designs.

- GRPO avoids maintaining a massive Critic while still offering a relative reward signal. It preserves the stability and compliance features of PPO but streamlines the process.

|Feature|PPO|GRPO|
|---|---|---|
|Critic (Value Function)|Required|Not Required|
|Advantage Estimation|At=r−V(s)A_t = r - V(s)At​=r−V(s)|Average over multiple old policy outputs|
|Update Control|Clip + Min|Clip + Min|
|Reference Alignment|KL Penalty|KL Penalty|
|Analogy|Dad tracks study & score lines|Kids simulate exams; Dad uses average score|

---

## 🏁 Final Thought:

GRPO keeps the strengths of PPO (safe updates, ethical alignment) but simplifies training by letting models **self-evaluate using simulated outputs**. It’s like letting students run mock exams and rewarding them only if they beat their own average — no need for a full-time evaluator.













---

[PAPER]()
## PAPER SUMMARY




---

[trl grpo trainer]()

## understanding the functions plus maths

excercise :


here's the exercise - 
i hope we're all acquainted with the GRPO loss. 
this is the `grpo_trainer` from HF, the code containing the GRPOTrainer class - https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py

2 exercises from  the code - 
1> what is the shape of `prompt_ids` from the code.
2> based on what's present in the code, how would you implement the DAPO loss - https://arxiv.org/pdf/2503.14476


