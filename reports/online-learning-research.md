# Research: learning useful facts while QwenBrain is deployed

Research date: 2026-09-11. Scope: a targeted review of primary papers and author-maintained implementations, including 2026 Doc-to-LoRA work. This is evidence for the next offline experiment, not a claim that continual memory is solved. Published experiments and this project's hypotheses are distinguished below. No training or live adapter changes were made for this research.

## Decision

Keep the current Qwen plus LoRA setup as the experimental platform. First compare raw-message training, diverse restatements, answer-supervised Q&A, and replay under controlled budgets. Measure fresh-context factual answers and retention rather than accepting a lower training loss as evidence of useful memory. Treat knowledge editing and learned test-time memory architectures as later research branches.

The key correction to the earlier explanation is that **next-token training can teach factual knowledge**. What we have not demonstrated is that a tiny amount of training on conversational user text produces reliable answers to differently worded questions. Fitting the observed sentence, extracting its fact under a new prompt, and retaining that fact through later updates are separate empirical questions. The controlled biography experiments support making those distinctions; they do not prove the cause of this project's failure. [Knowledge Storage and Extraction](https://arxiv.org/abs/2309.14316)

## Relevant evidence

### 1. Memorizing text and extracting its facts can differ

Allen-Zhu and Li train on controlled synthetic biographies and then examine factual extraction. Data diversity, including paraphrases and sentence order changes, strongly affects whether facts can be elicited through questions. Their experiments include models that fit training biographies but fail subsequent extraction, alongside conditions where extraction works. This is evidence that plain language modeling can acquire facts and that the form of exposure matters. It is not evidence that all facts require Q&A supervision or that a particular augmentation recipe will work on an instruction-tuned 0.8B hybrid model. **Project implication:** retain raw-text training as a baseline and test restatements separately from Q&A. [Physics of Language Models, Part 3.1: Knowledge Storage and Extraction](https://arxiv.org/abs/2309.14316)

### 2. Training format and access to context deserve separate tests

**New News (2025)** is especially relevant: it compares knowledge incorporation across Qwen2.5-Instruct models from 0.5B to 32B using LoRA. Its self-Q&A protocol improves incorporation relative to naive fine-tuning. It also reports *contextual shadowing*: prefixing training examples with the source information can reduce later use of that information without the prefix. Settings include rank 16, alpha 32, batch size 16, and learning rate 1e-4; this is not our one-step, rank-8 Qwen3.5 experiment. Smaller models also remained weak on some evaluations despite additional compute. **Project hypothesis:** an earlier occurrence of a fact in chat history may make local prediction easier without teaching independent recall. Test this rather than assuming it caused our result. [New News, Sections 4–6 and Appendices B/C](https://arxiv.org/html/2505.01812v1)

**Lampinen et al. (2025)** compare in-context learning and fine-tuning on controlled novel knowledge. In their data-matched studies, context often supports more flexible deductions. Augmenting fine-tuning data with context-generated inferences improves generalization; they also find circumstances where reversed relations can be learned. Main experiments use Gemini 1.5 Flash and hundreds of updates. This supports an explicit context-reading control and varied training examples, not a universal claim that weights cannot support reasoning. [Controlled generalization study](https://arxiv.org/html/2505.00661v2)

**Zhu et al. (EMNLP 2025)** connect QA extraction with predicting fact tokens under unseen paraphrased prefixes. Their experiments include Qwen2-1.5B and Llama2-7B; formatting augmentation and sharpness-aware minimization improve knowledge learning in the tested settings. This gives a reason to compare faithful declarative variation with Q&A, while changing the optimizer only in a later ablation. The reported correlations are not a diagnosis of our checkpoint. [Enhancing LLM Knowledge Learning through Generalization](https://aclanthology.org/2025.findings-emnlp.469/)

**The Reversal Curse (2023/ICLR 2024)** demonstrates failures to generalize learned associations in the opposite direction in GPT-3/Llama experiments. This is evidence that successful training on one formulation need not transfer to another. Our user-versus-assistant identity confusion is not literally the same test, so it would be premature to label our failure a confirmed reversal curse. [The Reversal Curse](https://arxiv.org/html/2309.12288v2)

### 3. New knowledge may learn at a different rate from familiar knowledge

Gekhman et al. use controlled closed-book QA fine-tuning, varying how much training material introduces new facts versus knowledge already available to the model. New factual examples are learned more slowly; in their setting, learning them is also associated with increased hallucination on evaluated questions. This is a warning against assuming one optimization step is a meaningful exposure budget or that stronger fitting monotonically improves an assistant. It does not establish a universal step count, show that our messages were forgotten, or prove that LoRA cannot learn them. **Project implication:** plot fact-answer performance across several exposure counts and keep an unrelated-question check. [Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?](https://arxiv.org/abs/2405.05904)

### 4. Replay is a reasonable baseline, not a guarantee

Ibrahim et al. compare continual pretraining against retraining on combined data. They study 405M models with English-to-English and English-to-German shifts, and a 10B model for the weaker shift, using hundreds of billions of tokens. Learning-rate scheduling plus replay can approach retraining performance. Their replay comparisons keep compute fixed by replacing some new-data tokens with old-data tokens. These results support testing replay and matching budgets. Their chosen replay fractions and learning-rate schedules are not prescriptions for a handful of chat facts or LoRA updates. **Project implication:** compare replay against an equally costly no-replay condition, and report the acquisition cost of allocating updates to old facts. [Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://arxiv.org/html/2403.08763v3)

Zucchet et al. study synthetic biography learning and find that training statistics can improve before precise individual facts are acquired. During later training on new individuals, old-fact performance can decline before new-fact performance improves. Replaying old examples only partly mitigates the problem in their setup. Their controlled task offers a useful counterexample to treating replay as a complete solution, but its learning dynamics cannot diagnose our deployed model. **Project implication:** measure each fact immediately after learning and again after later updates; a final zero alone cannot distinguish never learned from learned then lost. [How do language models learn facts? Dynamics, curricula and hallucinations](https://arxiv.org/html/2503.21676v1)

### 5. Knowledge editing directly targets associations, with different requirements

| Method | Primary result and relevant constraint | Fit for this project |
| --- | --- | --- |
| ROME | Directly edits a selected feed-forward matrix to change a factual association, evaluating efficacy, paraphrases, and neighboring facts. It uses a structured subject/relation/target setup. [ROME paper](https://arxiv.org/abs/2202.05262) | Useful conceptual comparison, but arbitrary first-person chat is not already a clean edit specification. A fact is not established to occupy one exclusive neuron or chat-activated region. |
| MEMIT | Extends direct editing to many associations, demonstrating thousands of edits on GPT-J 6B and GPT-NeoX 20B. [MEMIT paper](https://arxiv.org/abs/2210.07229) | Batch editing results do not establish reliable lifelong, sequential personal memory, nor compatibility with Qwen's hybrid architecture. |
| MEND | Learns auxiliary editor networks that transform fine-tuning gradients, permitting rapid edits from an input-output pair after editor training. [MEND paper](https://arxiv.org/abs/2110.11309) | Its speed at edit time depends on a previously trained editor; it is not simply a better one-step optimizer that we can assume transfers to our checkpoint. |

Across these methods, success on a direct question should not be treated as success on every related implication. RippleEdits explicitly evaluates consequences of edits beyond the edited association and finds limitations in existing editing methods. **Project implication:** keep paraphrase, related-fact, and unrelated-fact evaluations separate, whichever update algorithm is used. [Evaluating the Ripple Effects of Knowledge Editing in Language Models](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00644/120576/Evaluating-the-Ripple-Effects-of-Knowledge-Editing)

### 6. SEAL is close to the long-term idea, but has an extra learning loop

SEAL trains a model to produce useful adaptation data or update instructions, using downstream performance after fine-tuning as the learning signal. In its Qwen2.5-7B base, SQuAD-derived single-passage experiment, reported answer accuracy is 32.7% before adaptation, 33.5% after training on the passage, 39.7% with base-model synthetic additions, and 47.0% with learned self-edits. These figures belong to that evaluation, not a forecast for QwenBrain. The paper also reports forgetting during sequential edits. **Project implication:** data transformation is worth testing before an expensive outer reinforcement-learning loop. We should not market simple template augmentation as a reproduction of SEAL, or trust unsupported self-generated facts as training targets. [Self-Adapting Language Models, Table 2 and limitations](https://arxiv.org/html/2506.10943v1)

### 7. TTT layers and Titans are learned memory architectures

The 2024 TTT-layer paper makes a sequence layer's hidden state a small learned model. Its inner loop updates that state on the sequence; an outer training loop learns the surrounding system and the self-supervised task. Experiments cover 125M–1.3B models. The paper distinguishes sequence-dependent state updates from ordinary model parameters. **Project implication:** this resembles the idea of fast memory that adapts while reading, but is not equivalent to adding LoRA to an existing chat model and running ordinary language-model loss after each message. [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/html/2407.04620v2)

Titans combines attention with a neural contextual memory updated at test time, including learned mechanisms for retaining and forgetting information. Its separately named persistent-memory parameters are input-independent and fixed at test time. These are architectural roles, not a guarantee that an arbitrary chat assistant remembers facts across application restarts. **Project implication:** use the paper for a later architecture experiment; a Qwen retrofit would need integration and training evidence. It does not justify relabeling our saved LoRA adapter as a Titans implementation. [Titans: Learning to Memorize at Test Time, Sections 3–4](https://arxiv.org/html/2501.00663v1)

### 8. Doc-to-LoRA is a newer alternative, with an up-front training cost

**Doc-to-LoRA (2026)** meta-trains a hypernetwork to map a document into LoRA parameters for a target model. Once trained, it can produce an adapter in one forward pass, allowing later questions without the original document. Its Appendix C also reports interference with unrelated knowledge. Main experiments use Gemma-2-2B; the author repository's example loads a pretrained Gemma-specific checkpoint. We have not verified a compatible Qwen3.5-0.8B hypernetwork. This is a potential later benchmark or porting project, not evidence that attaching an arbitrary existing hypernetwork will work. [Doc-to-LoRA paper](https://arxiv.org/html/2602.15902v1), [author implementation](https://github.com/SakanaAI/doc-to-lora)

**Text-to-LoRA (2025)** instead generates adapters from task descriptions, after learning across task data or adapter collections. It addresses task adaptation rather than demonstrating continuous retention of arbitrary new personal facts. Distinguish that from document-to-weight knowledge incorporation. [Text-to-LoRA](https://arxiv.org/html/2506.06105v1)

## What the present result can and cannot tell us

The [verified mechanics](milestone-3.md) establish changed LoRA weights, frozen base weights, rollback, and reload. The [local v16 diagnostic](../runs/live-evaluation-1789130469748434500/summary.md) recorded 0/8 fresh-context personal-fact answers for both original and adapted models: four questions about three facts, each sampled with two seeds. Supplying facts in context helped both, while identity confusion also occurred in the original model. The detailed diagnostic remains a local, Git-ignored artifact because it contains conversation-derived information. Current settings and masking are in [learning.json](../configs/learning.json) and [learning.py](../qwenbrain/learning.py).

- Changed adapter tensors and a lower loss on the same message demonstrate optimization of that example. They do not demonstrate generalization or durable fact storage.
- Zero fresh-context factual answers gives no evidence of useful recall on those questions. It does not locate the missing information internally or prove that all possible probes would fail.
- Content-only training inside a user turn and question answering inside an assistant turn differ in both context and role. This is a plausible transfer difficulty, not an established sole cause.
- Training with prior chat as context may permit local text prediction without making an answer accessible after that context disappears. Removing training history is a separate ablation worth testing.
- Limited exposures, learning rate, rank, target modules, decoder behavior, and fresh-versus-persistent optimizer state remain confounded. Changing all at once would prevent us from learning which mattered.

## Proposed controlled experiment

This is an experimental design, not a published recipe or a promise of improvement.

### Start with stable, auditable facts

Use a small synthetic set of explicit named-subject facts before converting a whole private chat log. Stable names avoid confusing a person with the model's first-person identity. Include comparable first-person forms as a later paired test. Keep each fact's original statement, subject, relation, value, and source provenance. Predefine unrelated questions and unsupported questions too.

Create training forms and held-out evaluation forms before running updates. Q&A answers must follow directly from the source statement; introduce no inferred biography details. Different phrasings of one fact are not independent facts for statistics.

### Separate acquisition from retention

First run isolated acquisition trials from the same clean base and identical initial LoRA tensors, varying the number of exposures (for example 1, 4, 16, and 64) under the existing optimizer settings. Keep the current live snapshot as an additional reference, not the shared training initialization: its prior exposures are already confounded. Compare:

| Arm | Training data | Question it answers |
| --- | --- | --- |
| A | Original statement only | Does more exposure make the present objective useful? |
| B | Faithful declarative restatements | Does variation improve transfer without question supervision? |
| C | Grounded Q&A, assistant-answer loss | Does directly training the requested answer behavior help? |

Report both supervised target-token counts and GPU time; an update is not equal compute when sequences differ. In the main comparison, cap the total processed-token budget and state any residual compute differences. Do not tune using final held-out questions.

Start with a proposed local pilot of 12 synthetic facts: four for choosing settings and eight fresh facts for the final acquisition comparison. Final facts are still taught to each experimental model; their held-out *question phrasings* are not. Freeze the phrasings, scoring rules, and settings before that comparison. Repeat the selected settings over three training seeds and orders. These sizes are practical design choices, not statistical guarantees or paper prescriptions.

For the first format comparison, train on standalone examples. Create a separate history ablation with identical targets and update settings, adding an earlier answer-bearing statement only in one arm. For Q&A, the data-construction stage may see the verified source fact, but the student training input should contain the question without that answer-bearing source. Measure a source-prefixed variant separately. This avoids silently mixing a data-format change with an easier training context.

An example with synthetic data:

```text
Verified source: The user's project is named Cedar.
Training input: What is the user's project called?
Assistant target: The user's project is called Cedar.
Held-out probe: Tell me the name of the project I am building.
```

This is an answer-token SFT experiment inspired by context distillation. Hand-checked templates are sufficient initially; an unreliable self-generated answer would contaminate the test. A sophisticated data-generating teacher is a separate experiment.

Only after at least one arm acquires facts, run a sequential stream with and without replay. Use a small uniform buffer first. Match the overall token or time budget; record how much new-fact exposure replay displaces. Measure new-fact answers immediately, previous-fact answers after each stage, and the final old-fact performance. If no arm acquires facts, replay cannot rescue knowledge that was never usefully learned.

### Keep the test independent of the update

Evaluate with learning disabled, fresh conversation context, and no retrieval. Use deterministic decoding as the primary comparison, plus answer-token conditional log-probability against fixed plausible distractors to detect probability movement that generation misses. Keep prompt and scoring templates fixed and report length normalization. Compare the base, initial adapter, and each experimental adapter. Include a supplied-fact-in-context condition as a separate ability-to-read control; it is not weight memory.

For each fact, separate direct completion, held-out question phrasing, speaker reference, and downstream implications. Score factual meaning and the named subject, not just presence of a name token. Report unrelated-answer stability alongside new-fact accuracy. Save trial configurations, seeds, example counts, and failed outputs. Our gate includes three fixed reference losses, but none directly tests acquisition or retention of the new personal facts; it should not be described as a factual-memory gate. During the offline comparison, report raw candidate results separately from gate decisions, so rejected candidates do not silently disappear from the learning curves.

### Conditions for trying the result in live chat

Proceed only if an offline setting improves held-out fact questions over its matched baseline while retaining previously learned facts across later updates, within a reported latency budget. Then compare its final result across multiple seeds and fact orders. Preserve a frozen snapshot of the current serving adapter throughout research. Every incoming message can still trigger the learning pipeline; the evidence may favor different handling for explicit facts, corrections, greetings, and questions. Any change in that handling should be documented as a deliberate design change rather than silently assigning factual answers to arbitrary messages.

## Open questions for the next decision

1. Does a stable named subject help more than increasing exposures?
2. Do faithful restatements close the gap, or is assistant-answer supervision materially better?
3. Once a fact is acquired, how quickly does it decay under unrelated updates, and how much does replay cost?
4. Does an updated answer generalize to new wording without confusing the user's identity and the assistant's identity?
5. Does the result survive a saved-adapter reload and a fully fresh context?

The recommended immediate artifact is a small reproducible acquisition-and-retention benchmark with controlled adapters. A more elaborate memory architecture is premature until that benchmark tells us what the current system can learn.
