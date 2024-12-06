This paper presents an approach to automatically generating step-by-step solutions to math word problems. The approach involves combining a large language model (LLM) with an external symbolic solver.

- The LLM is used to formalize the word problem as a set of variables and equations.
- The symbolic solver is then used to solve the equations.

The authors evaluate their approach on two datasets: GSM8K and ALGEBRA.

- **GSM8K** is a benchmark dataset of grade-school level math word problems.
- **ALGEBRA** is a new dataset of more challenging word problems collected from Algebra textbooks.

The authors’ approach achieves comparable accuracy to the original Program-Aided Language Model (PAL) on GSM8K and outperforms PAL by an absolute 20% on ALGEBRA.

The authors find that their approach is particularly effective for solving Algebra problems that require declarative reasoning. This is because their approach allows the LLM to focus on formalizing the problem in a declarative way, while the symbolic solver takes care of the actual calculation. The authors also find that encouraging incremental formalization is beneficial, especially when using declarative representations.

The paper concludes that equipping an LLM with an external symbolic solver is an effective way to automatically generate step-by-step solutions to math word problems. The authors suggest that their approach is particularly useful for math education, as it can help students to understand the underlying concepts behind the problems.

### STEPS

This response draws on all the sources you provided.

The sources you have provided focus on using a large language model (LLM) in combination with a symbolic solver to solve math word problems. While they don't explicitly outline a universal problem-solving process, they do describe a two-step approach within their specific context:

- **Step 1: Declarative and Incremental Formalization Using an LLM**: This step involves utilizing an LLM to translate a math word problem into a structured representation that can be understood by a computer. This is done by:
    
    - **Few-Shot Prompting:** The LLM is provided with a few examples of word problems and their corresponding step-by-step solutions, known as a "prompt." This prompt guides the LLM on how to approach the problem.
    - **The DECLARATIVE Prompt:** The specific prompt used in this approach is called the "DECLARATIVE prompt." It focuses on representing the problem in a declarative manner, meaning it focuses on defining the relationships between variables and equations rather than outlining a specific procedure for solving the problem.
    - **Incremental Formalization:** The word problem is broken down into smaller parts, with each part being formalized as a variable or an equation. This process continues until the entire problem is represented in a structured format.
    - **Principles for Solutions:** The DECLARATIVE prompt follows certain principles to ensure the generated solutions are clear and consistent. These principles include using each sentence to introduce a new variable or equation, ensuring the last sentence identifies the goal variable, using only previously introduced variables in each equation, using a unique variable for each quantity, and utilizing all numbers from the problem.
- **Step 2: Solving Equations Using a Symbolic Solver**: The structured representation generated in Step 1, consisting of variables and equations, is passed to a symbolic solver. This solver is a specialized computer program that can solve mathematical equations algebraically.
    

Let's illustrate this process with a simplified example. Imagine the following word problem:

"John has 3 apples and Mary gives him 2 more. How many apples does John have now?"

**Step 1:**

- The LLM, guided by the DECLARATIVE prompt, might formalize the problem as follows:
    - [John's initial apples] = 3
    - [Apples Mary gives] = 2
    - [John's total apples] = [John's initial apples] + [Apples Mary gives]
    - Goal: [John's total apples]

**Step 2:**

- This set of equations is passed to a symbolic solver, which would compute:
    - [John's total apples] = 3 + 2 = 5

Therefore, the final answer would be that John has 5 apples.

It is important to note that this is a very simplified example. The actual process, especially when dealing with more complex algebraic problems, involves many more steps and considerations. The sources you provided focus on the effectiveness of this approach, especially when dealing with problems that require "declarative reasoning," meaning problems that are better solved by defining relationships rather than following a specific procedure.


