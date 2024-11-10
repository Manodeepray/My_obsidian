
authors: 
Yanzhe Zhang Georgia Tech z_yanzhe@gatech.edu 
Tao Yu The University of Hong Kong tyu@cs.hku.hk 
Diyi Yang Stanford University diyiy@stanford.edu


questions to answer : 
	why is it done : 
	 what is done :
	 what conclusions they reached :

#### Abstract

- Despite such visual inputs becoming more integrated into agentic applications, what types of risks and attacks exist around them still remain unclear.
- VLM agents can be easily attacked by a set of carefully designed adversarial pop-ups which human users would typically recognize and ignore.
- distraction -> agents to click / interact with these popups instead of the task at hand
- tried on agent testing environments like OSWorld and VisualWebArena
- attack success rate (the frequency of the agent clicking the pop-ups) of 86% on average and decreases the task success rate by 47%.
- Basic defense techniques such as asking the agent to ignore pop-ups or including an advertisement notice, are ineffective against the attack

#### Introduction

- Existing attacks in the digital world mainly aim to attract and visually mislead human users, such as pop-ups with banner ads, fake download buttons, and countdown timers for deals. If VLM agents are taking actions on behalf of users to perform these web tasks and are not aware of these attacks, this could lead to severe consequences such as installing malware or being redirected to deceptive websites
- here a threat model is considered. here an  attacker aims to make the agent click on the popups by manipulation of the agents observation
- attack scenarios:
	- malvertising (Sood and Enbody, 2011; Xing et al., 2015), in which attackers can either purchase an ad slot or leverage cross-site scripting (Hydara et al., 2015; Kaur et al., 2023) to inject malicious scripts that manipulate the website from the browser.
	- Attackers can also send a clickable image through phishing emails/messages (Patel et al., 2019) to ensure the pop-ups are shown on the screen.
- Goal of the attacks is not to be recognizable by humans if the entire task is done without human supervision but rather to mislead the agent performing the task.
- design space (Figure 2) of attacks includes four representative elements to attack: 
	- (i) Attention Hook: a few words to attract the agent’s attention. 
	- (ii) Instruction: desired behaviors the attacker intends for the agent to follow.
	-  (iii) Information Banner: contextual information that implies or misleads the agent about the purpose of the pop-ups. 
	- (iv) ALT 2 Descriptor: supplemental textual information provided for the pop-up within the a11y tree. In our experiments, we insert various types of adversarial pop-ups into the observation space for environments like OSWorld (Xie et al., 2024) and VisualWebArena (Koh et al., 2024).
- find that: 
	- (1) User query is essential for the attention hook, as using other alternatives (e.g., attackers speculate the user intent from the screen content.), on average, decreases the ASR by 61% relatively. 
	- (2) Other information (e.g., position and agent framework information) is relatively unnecessary to make the attack successful. 
	- (3) Basic defense strategies, such as asking the agent to ignore pop-ups and adding an extra advertisement notice, cannot effectively mitigate the issue (decrease the ASR by no more than 25% relatively). 
- In summary, deploying computeruse agents still suffers from significant risks, and more robust agent systems are needed to ensure safe agent workflow


#### Attack Design
objective :  mislead the agents to click the adversarial pop-ups, assuming the attacker has all the needed information and takes the most “ideal” actions toward this goal.

Attention Hook: 
- LLM  summarize user queries  => brief phrases
	(e.g., Could you help me change the username in chrome profiles to Thomas? into UPDATE USERNAME TO THOMAS)
- confuse the VLM agent into believing that pop-ups are related and that interacting with pop-ups is essential for processing user queries.
- the ground truth user query is tricky to access from the attacker perspective in real scenarios, so we consider two more alternatives: 
	- (i) Fake virus alert (Imamura et al., 2019): “VIRUS DETECTED”, a popular way of malvertising that targets human users, urging for actions no matter what queries user have. 
	- (ii) Speculated user query: Attackers usually know other information on the screen while inserting pop-ups. Without knowing the real user intent, they instead can ask an LLM to guess the user’s intent in a few words by providing such contextual information.

	

Instruction:
- “Please click (x,y)” or “Please click [ID]”, where (x,y) and [ID] are the center coordinate and tag ID (for SoM tagging) of popups.
- This is the most direct and ideal instruction since the agent does not even need to infer the position of pop-ups.
- However, the exact position of pop-ups might sometimes be out of control for the attackers. At the same time, the tag ID is usually generated from the agent framework, where attackers have zero knowledge. Thus, we consider two workarounds:
	- (i) “Please click here”: which requires the agent to either infer the position or read the tag ID without the need to know the agent framework. 
	- (ii) Click a random coordinate or tag ID: This essentially correlates the attack’s success rate with the area ratio of the pop-ups (or the number of tagged elements) if the agent follows the instruction

Info Banner:
Another way to fake the necessity of clicking the pop-ups is to convince the agent the pop-up is a button, so we use “OK” as the default info banner. As an alternative, we also test the effectiveness of using “ADVERTISEMENT” as an explicit notice in the info banner, which is a common practice of disclosures for real-world advertising

ALT Descriptor:
To align with the visual information, we use the summary of the user’s query (the attention hook) and the instruction as the adversarial ALT descriptor. We also analyze the performance of (i) an empty ALT string and (ii) adding “ADVERTISEMENT:” before the adversarial descriptor as the ALT, considering possible realworld setups and regulations.

#### experiments 
- Settings For OSWorld (Xie et al., 2024), we test screenshot agents and SoM agents on 50 easy tasks
- Models We use five frontier VLMs for our experiments: gpt-4-turbo-2024-04-09 gpt-4o-2024-05-13 (OpenAI, 2024), gemini-1.5-pro-002 claude-3-5-sonnet-20240620, and the latest claude-3-5-sonnet-20241022

#### Implementation 
To implement our attack, we first find the largest available rectangle area on the screen after excluding the bounding boxes from the a11y tree and OCR detection. 
We then randomly sample a position and size within this rectangle. Finally, we search for the largest possible font size that fits within the pop-ups, where the attention hook/instruction/info banner is arranged as the example in Figure 2.
To fully utilize the computational cost, we attack the agent observation whenever there is sufficient space for 
our popups (both height and width are more than 100 pixels). 
If the agent clicks on our pop-ups, we ignore this action during execution, and no redirection is implemented for simplicity. 
We use gpt-4o-2024-05-13 to summarize the user query and speculate the user query based on information on the screen through a11y trees. 
By default, we use “Please click (x,y)” as the instruction for both screenshot- and SoM agents in all OSWorld

#### How Does Our Attack Succeed? 
Since VLM agents are prompted to generate thoughts before generating actions (Yao et al., 2023), we study how our attack succeeds by taking a closer look at the generated thoughts.

Such information attacks guides the agent to give up the usual reasoning process (e.g., which image appears to be a screenshot in example 3) and passively follow the malicious “instruction”, revealing its lack of understanding of the function and impact of UI operations

#### Fail cases

- Agents declare WAIT/FAIL/DONE based on the interaction history. This happens when agents believe they solve the tasks, or tasks are not solvable.
- User queries are seeking information on the website.
	- summarized queries are no longer relevant to the desired actions since they do not contain the answers. 
	- When such an answer is directly provided elsewhere on the current page, forcing the agent to click on pop-ups instead of returning answers is hard. (
- Familiar tools are specified in the query (e.g., use terminal).
	- Since the backbone VLMs are heavily trained on coding data (including command line usages), agents tend to directly type in the commands when a terminal window is given on the screen. In

#### Conclusion
In this work, we show that VLM agents can be easily maliciously instructed by adversarial pop-ups while operating on computers. Even though these pop-ups look very suspicious (by design) to human users, agents cannot distinguish the difference between pop-ups and typical digital content, even with extra advertisement notices. 
- This work offers two takeaways: 
	- (i) Just like human users need to undergo training :
		- to recognize phishing emails (Kumaraguru et al., 2007), 
		- VLM models/agents might need to undergo a similar process to ignore environmental noises and prioritize legitimate instructions (Wallace et al., 2024) before operating in the real digital world. 
		- This also applies to embodied agents since many distractors in the physical environment might also be absent from the training data. 
	- (ii) Human users need to oversee the automated agent workflow carefully to manage the potential risks from the environment. 





#### Limitations
- (i) We only test the performance of closed-source models, which hinders a deeper understanding of why such 9 an attack works. This choice was made due to the relatively low performance of open-source models on computer agent benchmarks. Future research is encouraged to explore more capable open-source models. 
- (ii) We do not explore more advanced jailbreaking techniques, such as optimizing the string inside the pop-ups (Zou et al., 2023) or making our pop-ups more persuasive (Zeng et al., 2024), but focus more on the high-level design of the adversarial pop-ups and the contribution analysis of different components.
- (iii) There are several practical approaches to potentially mitigate our attack in the real world, such as implementing more robust content filtering in browsers, adding an additional module to detect malicious instructions in agent observations, and using more detailed and specific descriptions of the attack within the defensive prompt.

## chat gpt response

### Why is it done:

This study is conducted to understand and address the vulnerabilities of Vision-Language Model (VLM) agents to adversarial attacks in digital environments. These agents, when exposed to pop-ups that can distract or mislead them, might execute unintended actions such as clicking on malicious links or being diverted from their intended tasks. Since such interactions pose significant security risks, especially in scenarios where human users are not present to intervene, it is crucial to explore how these attacks work and evaluate their impact.

### What is done:

The researchers designed and tested adversarial attacks specifically targeting VLM agents using visual pop-ups in controlled environments such as OSWorld and VisualWebArena. They crafted attacks to trick agents into interacting with pop-ups by exploiting elements like attention hooks, misleading instructions, and fake information banners. The study evaluated various attack strategies and their effectiveness against five frontier VLM models. Basic defense techniques, such as instructing agents to ignore pop-ups, were also tested to assess their utility.

### What conclusions they reached:

1. **Susceptibility of Agents**: VLM agents are highly susceptible to adversarial pop-ups, with an average attack success rate of 86%, leading to a decrease in task success rates by 47%. Unlike human users, agents struggle to differentiate between legitimate content and attacks, even when obvious advertisement notices are present.
2. **Ineffectiveness of Basic Defenses**: Simple strategies like ignoring pop-ups or labeling them as advertisements failed to offer significant protection against these attacks, reducing the attack success rate by no more than 25%.
3. **Need for Robust Training and Oversight**: Similar to how human users receive training to identify and ignore phishing attempts, VLM agents require better training to filter environmental noise and prioritize legitimate tasks. Human supervision remains essential to mitigate risks and ensure safe automated workflows.


