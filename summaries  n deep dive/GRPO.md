# Group Relative Policy Optimization



## DeepSeek-R1 Dissection: Understanding PPO & GRPO Without Any Prior Reinforcement Learning Knowledge
[Article](https://huggingface.co/blog/NormalUhr/grpo)

Analogy : RL training process as an elementary school exam scenario  
We ( the model being trained ) are like students trying to get high grades , the teacher who grades our exams are like the reward model ,
while our parents handling out pocket money based on or grades is similar to the critic .

- final scores alone are insufficient ,
	- Critic 
	- clip 
	- Reference model are needed

### Naive approach
**only using reward**

- *problem* : 






 
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

