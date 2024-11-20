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