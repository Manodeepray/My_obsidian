Collaborative Memory is a framework designed to manage knowledge sharing among multiple users and large language model (LLM) agents in environments with dynamic and asymmetric access controls. It enables safe, efficient, and interpretable cross-user knowledge sharing while ensuring adherence to time-varying policies and providing full auditability of memory operations.

Here's a step-by-step algorithm explaining how the Collaborative Memory system works:

**Core Concepts and Components:** At its foundation, Collaborative Memory operates with three key components:

- **Dynamic Bipartite Access Graphs**: These graphs formally define time-dependent permissions. 
	- One graph links users to agents (`GUA(t)`), indicating which users can invoke which agents at a given time. 
	- The other graph links agents to resources (`GAR(t)`), showing which agents can access specific resources. 
	- These graphs are dynamic, evolving over time to reflect changes like user onboarding, role adjustments, or shifting policy constraints.
- **Two-Tier Memory System**: Each agent maintains a dual memory structure:
    - **Private Memory**: This tier holds information visible only to its originating user, isolating sensitive data within individual user contexts. For each user, a private store (`Mprivate(u, t)`) includes all memories from that user's past interactions with agents.
    - **Shared Memory**: This tier facilitates knowledge transfer among users when permitted by the system's access policies. For each agent, a shared store (`Mshared(a, t)`) includes all fragments the agent generated for any user up to time `t`.
    - Each memory fragment carries immutable provenance attributes, including its creation time, the contributing user, the agents involved, and the resources accessed during its creation.
- **Fine-Grained Read and Write Policies**: These policies govern how memory is interacted with:
    - **Write Policy (`πwrite/private`, `πwrite/shared`)**: This policy processes raw interaction logs and transforms them into structured memory fragments, then selectively stores these fragments into either the private or shared memory tiers. It can anonymize, redact, or block sensitive content.
    - **Read Policy (`πread`)**: This policy dynamically constructs a memory view tailored to an agent's current permissions, selectively incorporating memory fragments based on fine-grained access constraints. It can filter fragments by keywords or limit the number retrieved.
    - Both policies are highly configurable, supporting specifications and enforcement at system-wide, agent-specific, and user-specific levels, and are adaptive over time.

## working

The Collaborative Memory framework is designed for **multi-user, multi-agent environments** where different users and specialized LLM agents interact with various resources and share knowledge, all while adhering to **dynamic, asymmetric access controls**. This system aims to maximize the utility of collective memory while ensuring information sharing conforms to defined permissions.

Here is a step-by-step algorithm outlining how Collaborative Memory works:

### Collaborative Memory Working Algorithm:

1. **System Setup and Access Control Definition**
    
    - The framework begins by formalizing **time-dependent permissions** using two **dynamic bipartite graphs**:
        - **User-to-Agent Permissions ($G_{UA}(t)$)**: This graph defines which user `u` may invoke which agent `a` at a given time `t`.
        - **Agent-to-Resource Permissions ($G_{AR}(t)$)**: This graph specifies which agent `a` can access which external resource `r` (e.g., tools, APIs, structured data sources) at time `t`.
    - These graphs are **time-evolving**, reflecting changes such as user onboarding, role changes, or evolving policy constraints. This is illustrated in Scenario 3, where permissions are granted or revoked in real-time.
    - A **two-tier memory system** is established for each agent, consisting of **private memory** and **shared memory**.
        - **Private memory** (`Mprivate`) isolates sensitive information to individual user contexts.
        - **Shared memory** (`Mshared`) enables knowledge transfer among users when allowed by system access policies.
2. **Query Submission by User**
    
    - At each timestep `t`, a user `u` submits a query `q` to the multi-agent memory-sharing framework.
3. **Agent Selection by Coordinator**
    
    - A **coordinator LLM** receives the user query, along with the natural-language specialization string for agents accessible by the user, and the conversation history.
    - The coordinator selects the **most appropriate agent(s)** from the set of agents that the user `u` is permitted to invoke at time `t` (`A(u,t)`),
    - ensuring that only authorized agents are eligible to respond. 
    - For complex queries that span multiple domains, the coordinator may determine if sequential agent consultation is needed.
4. **Response Generation by Selected Agent(s)**
    
    - Each selected agent `a` (e.g., `a1`) receives the query `q` and the relevant memory, which is filtered by a **read policy** ($\pi^{read}_{u,a,t}$).
    - The **read policy** dynamically constructs a **memory view tailored to the agent's current permissions**, selectively incorporating memory fragments. 
    - This policy can, for example, limit the number of memory fragments retrieved or filter them by specific keywords. 
    - A simple read policy returns admissible fragments verbatim.
    - The agent then generates a response, denoted `y_{u,a,t}`, based on the query, the filtered memory, and the resources `R(a,t)` that it can access. 
    - The set of memory fragments potentially accessible by an agent `a` serving user `u` at time `t` (`M(u,a,t)`) is constrained by the provenance attributes of the fragments, ensuring that only fragments created by agents within the user's allowed set (`A(m) ⊆ A(u,t)`) and with resources within the agent's allowed set (`R(m) ⊆ R(a,t)`) are considered.
    - Agents interact with external tools (resources) through interfaces like OpenAI function-calling, where resources are exposed as Python callables with JSON schemas.
5. **Memory Update (Write to Collaborative Memory)**
    
    - Upon producing an output `y_{u,a,t}`, the system applies **write policies** to insert new information into the memory.
    - Two types of write policies are defined:
        - **Private Write Policy ($\pi^{write/private}_{u,a,t}$)**: Updates the user's private store (`Mprivate(u,t)`) with fragments from the interaction, ensuring information is kept confidential to that user's context.
        - **Shared Write Policy ($\pi^{write/shared}_{u,a,t}$)**: Updates the shared memory (`Mshared(u,t)`) with fragments that can be shared across users.
    - These policies are **highly configurable** and can be specified and enforced at system-wide, agent-specific, or user-specific levels, and are adaptive over time. They may transform memory fragments, for instance, by anonymizing entities, redacting sensitive information, or blocking certain content from being stored.
    - Memory fragments have **immutable provenance attributes**, including their time of creation, contributing user, contributing agents, and resources accessed during creation. This supports retrospective permission checks and auditability.
    - In the system's current implementation, memory is updated using the _intermediate responses_ from agents.
6. **Response Aggregation**
    
    - After all eligible agents have responded, their intermediate outputs (e.g., `{y_{u,a,t}}`) are aggregated by an **aggregator LLM**.
    - The aggregator synthesizes these agent outputs into a **single, coherent final response** for the user.

This detailed pipeline ensures that users' queries are addressed effectively through collaborative intelligence, while critical **privacy and access constraints are strictly enforced** at every step, leveraging the system's ability to maintain a balance between knowledge isolation and controlled collaboration.