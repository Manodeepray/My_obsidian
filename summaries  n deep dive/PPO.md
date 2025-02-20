## Proximal Policy Optimization

3 pass Explanation

### 1. Concepts of ppo

Architectures
- make use of 2 main neural networks
	1. Policy Network 
		-  Input = state
		- output = action
			- output layer has the same number of neurons as the number of actions that can be taken i.e. each neuron is probability of each action is given as output ..prob of taking A1,prob of taking A2 .....prob of taking An
	2. Value Network
		- Input =
			- state
			- action
		- Output = 
			- q-value i.e. it quantifies how good was this decision by the model
			- output layer has the same number of neurons as the number of actions that can be taken i.e. for each neuron here is q value of each action and state is given as output ..Q(S,A1),Q(S,A2),Q(S,A3)...Q(S,An)

### 2. Overview of PPO
}