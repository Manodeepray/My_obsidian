murari : I need to know about some ideas where we pit 2 or more LLMs against each other or make them work in collaboration to do something.. what are some of the tasks that would benefit from such multi Agent LLM interactions? Don't answer randomly, give me some references/real world use cases...

Think of Game Theory Principles, Cooperative Game Theory, Nash Equilibrium.. etc...

Also, we should think theideas at Inference level, should not change the original model but play around with the CoTs, RAGs, etc.


neel : one of the first things that strike me is reasoning. I haven't seen any papers on this, but if 2 LLMs, both acting as each other's critic engage in a loop till a certain theta threshold (theta being some amount of equality in both LLM answers).

a lot like how human critical conversations work.

this needs to be empirically tested and if prepared with the right frameworks, could open up a branch for multi LLM discussion and reasoning improvement.



people want experts in a certain field instead of jack of all trades









### 1. **SQ3R Method (Survey, Question, Read, Recite, Review)**

- This five-step approach involves surveying a text (scanning headings and summaries), asking questions about the content, reading to find answers, reciting or summarizing key points, and reviewing everything to reinforce learning.




28-10-24


Agent LLMs are autonomous language models designed to perform specific tasks by following a series of pre-defined steps or "policies." These "agents" are capable of:

1. **Interpreting Commands**: They take natural language instructions to understand what task to perform.
2. **Planning Actions**: Use logic or decision-making frameworks to sequence actions.
3. **Executing Actions**: They call APIs, use tools, or interact with environments to complete tasks (like data retrieval, calculations, or other functions).
4. **Providing Feedback**: They can track and update their progress, adjusting strategies if needed.

Agent LLMs often rely on frameworks like LangChain or OpenAI's function-calling API to access external tools and integrate seamlessly into applications.


## INVERSE RAG 

first how does RAG work?

1. Retrieval : using similarity search -> context from database
2. Argument : context + query as argument into LLM with some prompt 
3. Generate : LLM generates the response
### my idea

1. an agent (LLM) takes query for editing a topic or unlearning it from a LLM
2. LLM generates what to unlearn or edit -> give everything about the topic/ person -> method to extract everything related to it
3. Agent then  edits data as per the request
4. the data is then converted to appropriate format for unlearning method or editing method by the agent
5. then as per the query the agent applies the method on LLM with the forgetting / editing dataset


what to gather from the query
-->topic
-->request type
-->data to be edited/unlearned

---

chat gpt expanded on the idea
### My Idea: Agent-Based Framework for Topic Editing or Unlearning in LLMs

1. **Agent-Driven Query Processing**
   - An intelligent agent (powered by a Large Language Model, or LLM) receives a query specifying the need to either *edit* or *unlearn* knowledge related to a particular topic, entity, or individual in an LLM. 
   - The agent interprets the request type (e.g., remove, modify, or restrict information) and identifies key topic on which these changes are required.

2. **Comprehensive Knowledge Extraction**
   - The LLM (acting as the core knowledge source) generates all information associated with the target topic or entity. This could include descriptions, historical facts, attributes, or any specific associations.
   - The agent then employs a method to systematically extract everything relevant to the topic, ensuring all potential connections and references are captured for a thorough unlearning or editing process.

3. **Customized Data Editing**
   - The agent reviews the extracted data and applies necessary modifications based on the query. 
   - If the task is to unlearn (forget) information, the agent flags or removes sensitive details. If the task is to edit, the agent modifies relevant data to reflect new, corrected, or redacted information.

4. **Preparation of Unlearning/Editing Dataset**
   - The edited or flagged data is then formatted appropriately by the agent, aligning it with the required structure for unlearning or editing processes.
   - This step ensures that the data can seamlessly integrate with the LLM’s unlearning or knowledge editing mechanism, which may involve specific formatting or metadata tagging for easy access and processing.

5. **Application of Unlearning or Editing**
   - Based on the query’s requirements, the agent initiates the selected method (unlearning or editing) on the LLM.
   - Using the prepared dataset, the agent applies either an unlearning approach (to “forget” the information) or an editing approach (to update or modify the information) within the LLM. 
   - This step finalizes the request, making the necessary changes to the LLM’s outputs without affecting unrelated knowledge.

---
Q. how can this be done to the llm itself and evaluate it too


This approach provides a structured, agent-driven workflow for precise knowledge management in LLMs, making it adaptable for both open-source and closed-source environments. The method ensures minimal model disruption while offering customizable and controlled knowledge updates.


# IDEA 2

method of knowledge editing using long continuous prompting and combining it with a methodology  like that of astute rag to minimize hallucination and conflict between the system and llm... maybe somehow integrating multimodal editing on contextual knowledge...



prompts for editing
repo                              prompt / edited data
continuous prompt


user query

network similar to knowledge sentinel to find similar prompts from repo
system similar to astute rag paper and machine unlearning paper with rag  that would compare the retrieved prompts/data (or unlearned/edited database ) and internal knowledge     
then it either allows or does not allow the info to the user 

here we can add more methods at the end to get a better output


# ideas

1. ACL generalizatrion analysing failure condition of mmlm m-editing methods
failure cases why is it failing 
- evalutaion matrix 
- how efficient
- what led to the failure

2. inverse rag+

3. use cases for mathematical solving using editing-specific set of problems- incontext learning -reasoning paper

specific




mandal sir : Generalization of Multimodal LLM Models.... Different Tasks...


ankur sir : Compare declarative programming with COT for symbolic and math data


# compressed RAG

compress the text into encoded results and then see the result

