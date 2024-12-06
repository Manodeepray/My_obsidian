# paper

logical deductions between facts
synthetic dataset EDU-RELAT
knowledgebase of family relationships and biographies

connected facts – and very often, the fact that has been unlearnt can be deduced from facts that are already known by the model.

Deep unlearning is formulated by stating a set of facts and logical rules that connect the facts. The fact is deeply unlearnt if the target fact cannot be deduced from the retained facts in the LLM through the given logical rules.

Recall measures to how well an unlearning method unlearns the relevant facts so
that the target fact cannot be deduced; while accuracy measures what extent other irrelevant facts
are retained by the unlearning process.

dataset EDU-RELAT
as a benchmark. The dataset consists of two parts: a synthetic knowledge base and a realistic
logical rule set.

biographical information about a group of people
(e.g., “The birthyear of Sloane Lee is 1908”), as well as family relationships

logical rules are realistic rules describing family relationships

four common unlearning methods (Gradient Ascent, Nega
tive
Preference Optimization, Task Vector, and Who’s Harry Potter) on four popular LLMs (Phi-1.5,
GPT2-XL, Llama2-7b, Llama3-8b).



