# LLM post-training learning map

Post-training is optional for the first usable prototype. Establish the safety harness and evaluation baseline first; otherwise you cannot show whether tuning made the system better or less safe.

## 1. Supervised fine-tuning (SFT)

Use a small, licensed instruction dataset of desired **patient-education** behavior: grounded answer style, citation placeholders, boundary statements and escalation language. Learn data formatting, train/validation split, LoRA/QLoRA, loss curves, checkpoint selection and regression evaluation.

## 2. Preference optimization

Create pairs such as “grounded, concise answer with escalation” versus “confident unsupported diagnosis”. Study DPO / ORPO / SimPO and understand that preference data can improve style but does not prove clinical safety.

## 3. Reward modeling and RL

Only after SFT and offline evaluation are stable. Implement decomposed rewards for citation support, refusal correctness, formatting and helpfulness; keep a held-out adversarial set. Learn PPO/GRPO/DAPO conceptually, reward hacking, KL control, on-policy instability and why a scalar reward is not clinical validation.

## 4. Retrieval-aware training

Compare prompt-only RAG against citation-aware SFT. Measure retrieval recall separately from answer faithfulness, because fine-tuning cannot compensate for missing or stale evidence.

## Minimum experiment record

For every run, retain dataset version and license, prompt/model/retriever version, hyperparameters, compute budget, random seed, metrics, failure cases and a decision note. Never claim training on hospital records without an actual authorized data-governance process.

