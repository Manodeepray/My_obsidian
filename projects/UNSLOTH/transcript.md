# Synthetic Data AI Agents Challenge
### Hackathon Overview & Structure

- Virtual hackathon with AI agent competition format
- Team registration closes at 10pm (Pacific time)
- Submissions due Sunday 2pm sharp - hard deadline
- Tournament bracket style: teams compete head-to-head
  - Team A’s question agent asks 20 questions to Team B’s answer agent
  - Team B’s question agent asks same number to Team A’s answer agent
  - Scoring: correct answers = points for answer agent, incorrect = points for question agent
- 8 finalist teams advance to AMD AI Dev Day (Monday, San Francisco)

### Prizes & Resources

- 1st place: $3000 + 1200 GPU hours
- 2nd place: $1500 + 600 GPU hours
- 3rd place: $900 + 300 GPU hours
- All prizes divisible by 3 for team splits
- Free MI 300 GPU access (192GB vRAM) for competition
- Finalists get full 8-GPU node

### Technical Requirements

- Two topics only: seating arrangements & blood relations/family tree
- Required JSON response format:
  - Question agent: topic, question (multiple choice), choices, answer, explanation
  - Answer agent: answer (letter choice), reasoning
- Code goes in agents folder: answer_model.py & question_model.py
- Time limits: 10 seconds per question, 6 seconds per answer
- English only, no RAG or adversarial approaches allowed
- Max token limits set in YAML configs (don’t modify)

### Development Tips & Tools

- Synthetic Data Kit encouraged for QA pair generation
- Fine-tuning pipeline: ingest data → create QA pairs → curate → train with Unsloth
- Focus on data curation over hyperparameter optimization
- LLM judges prefer explanations - include reasoning in answers
- Consider multiple smaller models for different problem subsets
- Stay within same model family for consistent behavior scaling
- Baseline before/after fine-tuning to verify improvements

---

## Technical Details

- Topics: Only two allowed—seating arrangements and blood relations/family tree. Questions must strictly pertain to these.
- Data: No provided dataset. You must generate/curate your own (synthetic data encouraged). Ensure licensing is compliant for any sourced materials.
- Agents and IO formats:
  - Question agent must output JSON with: topic, question (must be multiple choice), choices, answer (correct letter), explanation (optional).
  - Answer agent must output JSON with: answer (letter), reasoning (optional).
  - Extra JSON keys are allowed, but required keys must be present and valid.
- Code structure:
  - Implement logic only in agents/answer_model.py and agents/question_model.py.
  - Do not modify agents/answer_agent.py and agents/question_agent.py (these are used for evaluation).
  - Store checkpoints within the provided top-level workspace; ensure loading paths resolve on the MI300 instance.
- Infrastructure:
  - Each team gets an MI300 GPU (192GB vRAM). Finalists get an 8-GPU node.
  - Access via JupyterLab at a provided IP URL; default password: AMD hack (will be DM’d to team lead).
  - You may install additional packages; if you break the env, organizers may reset it but won’t handhold custom setups.
- Models:
  - No restriction on model family/size; use what fits within hardware/time limits.
  - Fine-tuning supported (LoRA, QLoRA, full FT) with Unsloth; multi-GPU allowed for finalists.
  - Tip: Iterate on smaller variants within a model family for faster loops; scale later within same family.
- Inference/runtime constraints:
  - Time limits enforced during bracket runs: 10s per generated question; 6s per generated answer.
  - No RAG, no tool use beyond pure generation as implied by token/config limits; English only.
  - Max token limits set in YAML configs—do not change those values.
- Evaluation and scoring:
  - Bracket head-to-head. Team A’s question agent asks N MCQs; Team B’s answer agent answers them, and vice versa.
  - Points = number correct for answer agent; question agent gets points for opponent’s incorrect.
  - Validity: Question agent must include the correct answer in its JSON; organizers verify solvability and correctness offline.
  - LLM-as-a-judge is used for some checks; explanations in answers can improve judged quality (not required but recommended).
- Development workflow:
  - Recommended pipeline: ingest source docs → generate QA pairs (Synthetic Data Kit) → curate/score → fine-tune with Unsloth → validate with provided test scripts.
  - Baseline pre/post fine-tuning; consider ensembles or specialized smaller models per subtopic if within limits.
- Prohibited/allowed tactics:
  - Disallowed: RAG, adversarial prompts, prompt injection into opponent agents, non-English, image/diagram questions.
  - Allowed: Aggressive prompt engineering, synthetic data generation, model distillation from larger to smaller models, optional extra metadata in JSON (non-adversarial).
- Deadlines/logistics:
  - Team registration closes at 10 (Pacific). Submissions hard cutoff Sunday 2pm Pacific. Edits after deadline risk disqualification.
  - Bracket details and finalists’ scoring may differ; finalists invited to AMD AI Dev Day in SF.

## Tips
- Prioritize data curation over hyperparameter tuning; your dataset quality is the main edge.
- Use Synthetic Data Kit to generate/curate QA pairs; tweak prompts/configs for better coverage and difficulty.
- Baseline before/after fine-tuning to verify real gains.
- Include reasoning/explanations in answers; LLM-as-a-judge tends to reward them.
- Iterate on smaller models within a single family for fast loops; scale up later within the same family to keep behavior consistent.
- Consider ensembling/specialized smaller models per subtopic (e.g., separate models for seating vs. blood relations).
- Distill from larger models to produce high-quality training data for smaller, steerable models.
- Keep required JSON keys exact; extra metadata is fine but avoid adversarial hints/injections.
- Store checkpoints locally on the MI300 instance for faster inference.
- Use the provided test scripts to validate output format and runtime limits early.