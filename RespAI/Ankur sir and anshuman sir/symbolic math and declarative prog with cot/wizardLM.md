This paper introduces **WizardMath**, a large language model specifically designed to enhance mathematical reasoning abilities. WizardMath is built upon the **Llama-2** model and is trained using a novel method called **Reinforcement Learning from Evol-Instruct Feedback (RLEIF)**.

### RLEIF

RLEIF is a three-step process:

1. **Supervised fine-tuning**: The base Llama-2 model is fine-tuned on a dataset of instruction-response pairs, including step-by-step solutions for mathematical problems and open-domain conversations.
    
2. **Reward model training**: Two reward models are trained:
    
    - **Instruction Reward Model (IRM)**: Evaluates the quality of evolved instructions.
    - **Process-supervised Reward Model (PRM)**: Assesses the correctness of each step in a generated solution.
3. **Active Evol-Instruct and PPO training**: The original mathematical instructions from datasets like GSM8k and MATH are evolved, increasing their complexity and diversity. The IRM and PRM are used to provide rewards, guiding the model's learning through Proximal Policy Optimization (PPO).
    

### Evaluation

WizardMath is evaluated on two mathematical reasoning benchmarks:

- **GSM8k**: Contains grade-school level math problems.
- **MATH**: Contains challenging math problems from competitions like AMC 10, AMC 12, and AIME.

### Results

WizardMath achieves state-of-the-art performance, surpassing all other open-source LLMs on both GSM8k and MATH.

- WizardMath shows significant improvement over the baseline Llama-2 70B model, with a **+24.8%** increase in accuracy on GSM8k and **+9.2%** on MATH.
    
- WizardMath also outperforms several closed-source LLMs on GSM8k, including ChatGPT-3.5, Claude Instant-1, and PaLM-2.
    

### Analysis

The success of WizardMath is attributed to the RLEIF method, which effectively combines the strengths of Evol-Instruct and reinforcement learning. Evol-Instruct helps generate diverse and complex instruction data, while the reward models provide fine-grained feedback, enabling the model to learn robust mathematical reasoning skills.

### Limitations and Future Work

While WizardMath shows impressive performance, it still lags behind models like GPT-4 and Claude-2. Future work will focus on further improving RLEIF and exploring better methods to enhance the model's reasoning capabilities. Additionally, addressing the ethical implications and potential biases of LLMs is crucial for responsible development and deployment.



## as 5 yr old 
## Using Games to Teach a Computer Math

Imagine you are playing a game where you need to solve math problems to win. You might start with easy problems, like 1+1, and as you get better, the game gives you harder problems, like 2+3.

**Reinforcement Learning from Evol-Instruct Feedback** is like a special way to teach a computer how to play that math game and get really good at solving math problems. It's like having a teacher who gives the computer harder and harder math problems and then tells it how well it did.

Here's how it works:

- **Evol-Instruct:** This part is like the teacher who comes up with new and harder math problems for the computer to solve. It's like evolving the problems to make them more challenging.
- **Reinforcement Learning:** This is how the computer learns from its mistakes and successes. Imagine the computer gets a point for every math problem it solves correctly. The computer wants to get as many points as possible, so it learns to solve the problems better and better.
- **Feedback:** This is how the teacher tells the computer how well it's doing. The teacher uses two special tools to give feedback:
    - **Instruction Reward Model (IRM):** This tool tells the computer how good the math problem is.
    - **Process-supervised Reward Model (PRM):** This tool checks each step the computer takes to solve the problem and tells it if it's on the right track.

By using this special way of teaching, the computer can learn to solve even very difficult math problems! That's how **WizardMath**, a special computer program, was taught to be really good at math. It's like the computer became a math wizard!


### How WizardMath Uses Reinforcement Learning

The sources do not explicitly state whether reinforcement learning is run for each question or in batches. However, based on the description of **Proximal Policy Optimization (PPO)** training, it is likely that reinforcement learning is applied in batches.

- PPO is a type of reinforcement learning algorithm that updates the model's policy based on the collected experience from interacting with the environment (in this case, solving math problems) [outside source].
- PPO typically involves collecting a batch of experience data and then using that data to update the model's policy [outside source]. This is more efficient than updating the policy after each individual question.

The sources do mention that the goal of reinforcement learning is to **maximize a reward** that combines the outputs of two reward models:

- **Instruction Reward Model (IRM)**: This model provides a reward (**rI**) based on the quality of the evolved instruction.
- **Process-supervised Reward Model (PRM)**: This model provides a reward (**rA**) for each step in the solution, based on its correctness.

The final reward (**r**) used in PPO training is calculated as the product of these two rewards: **r = rI ⋅ rA**.

While the sources do not explicitly state that **accuracy** is the metric being maximized, it is implied that the reward models are designed to encourage accurate problem-solving.

- The IRM rewards high-quality instructions, which are likely to lead to accurate solutions.
- The PRM rewards correct steps in the solution process, ultimately leading to a higher overall reward for accurate answers.

Therefore, it can be inferred that maximizing the combined reward indirectly leads to maximizing the accuracy of the model on mathematical reasoning tasks.