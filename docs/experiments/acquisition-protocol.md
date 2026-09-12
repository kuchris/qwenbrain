# Acquisition comparison v1

This offline experiment implements the first acquisition comparison from
[the research review](../research/online-learning-research.md). It does not change the live
chat objective or load/train/save `adapters/live.safetensors`.

## Frozen design

The fixtures, milestone schedule, and train/test question templates are in
[acquisition.json](../../configs/acquisition.json). All people/project associations
are explicitly synthetic. The four pilot facts and eight confirmation facts have
disjoint subjects and values. Every confirmation fact is taught during its trial;
the held-out material is the evaluation question wording, not the factual answer.

| Format | Input and training targets |
| --- | --- |
| Raw | Repeated original statement in a user turn; loss on its content |
| Varied | Four faithful declarative forms, cycled; loss on user content |
| Q&A | Four question forms, cycled; only the assistant's project-name answer is scored |

No example includes preceding facts or previous chat history. The first exposure
of raw and varied uses the identical statement, providing an internal comparison.
System text and the pinned Qwen chat template are shared. Control/template tokens
and end-of-turn tokens are not prediction targets in any arm.

One exposure means one example of every fact in the phase, in a single minibatch:
four rows for the pilot, eight for confirmation. This differs from the live loop's
one-message batch. Checkpoints follow 1, 4, 16, and 64 exposures per fact. Rank 8,
alpha 16, LR 0.0001, zero dropout/weight decay, clipping 1.0, and a fresh AdamW per
step are held fixed. Each arm starts from identical LoRA tensors for its seed;
the frozen base is never trained. Seed controls initialization and row ordering.

All training rows are padded to the maximum length across all fixed fixtures.
Every arm computes full vocabulary logits, including masked positions, so arms
in a phase have identical batch dimensions and padded-token budgets at each
milestone. Report actual nonpadding tokens, supervised tokens, and GPU time too:
matching padded dimensions is a compute control, not a claim of equal information
content. The objective and number of supervised tokens necessarily differ.

The pilot compares all three arms at seed 42. Choose varied or Q&A by mean greedy
exact-match accuracy over the four milestones; tie-break by mean candidate-rank
accuracy, then prefer varied. The confirmation compares that selected format
against raw at all four milestones with seeds 42, 43, and 44. This selection rule
is coded before pilot outcomes and does not inspect confirmation results.

## Evaluation

Each fact gets two question wordings absent from the training templates. Every
question starts a fresh system-only conversation; evaluation never updates the
weights. Greedy generation uses at most 24 new tokens and asks for just the name.
Exact match ignores case, whitespace, punctuation, and formatting characters; it
does not give credit merely for mentioning the expected name in a longer answer.
Consequently, exact match also measures format compliance. Raw answers must be
retained and inspected when interpreting recall.

Additionally score each possible project name from the phase by mean answer-token
log-probability, under the same question, without putting answer choices in the
prompt. Record the gold score and whether it beats every distractor. Ties count
as incorrect. Candidate-rank accuracy is a constrained discrimination measure,
not free recall; its uniform-choice reference is 25% for the pilot and 12.5% for
confirmation. Different phrasings/seeds are not independent facts.

Evaluate frozen/no-op baselines without facts and with the corresponding source
fact explicitly supplied. The latter measures reading ability, not weight memory.
Four unrelated arithmetic/geography questions are checked at each checkpoint.
This small control set cannot establish preservation of general capabilities.

Record raw checkpoint performance regardless of the live candidate gate. The
three fixed reference-loss gates are not applied to this experiment; accepted
versus rejected live updates would confound the acquisition curves.

## Integrity and storage

The harness checks exact content-target decoding, masked next-token loss against
an independent FP64 calculation, and padded versus unpadded loss. Adapter hashes
before and after every evaluation must match. A full base-parameter checksum must
match before and after the complete experiment. All final arm adapters, raw
answers, per-step losses, timing, and token counts go to `runs/acquisition-*/`.
Those large artifacts are ignored by Git. The final report records source hashes.

Activation checkpointing is enabled. SDPA's math backend must enclose both forward
and backward: the first attempted pilot raised a recomputation metadata mismatch
before its first optimizer step when backward left the forced backend context.
The experiment wrapper now holds that context through the entire update. Live
training code was not modified as part of this experiment.

Run [compare-learning.bat](../../scripts/compare-learning.bat), or from the project root:

```powershell
.venv\Scripts\python.exe -X utf8 -m qwenbrain.experiment_acquisition
```

This run measures acquisition. Replay, sequential forgetting, first-person chat,
history ablations, and promotion into the live model are separate experiments.

After the main comparison, a supplementary probe continues each original user
statement immediately before its project name, using the final 64-exposure
checkpoints. It records the first completed clause and scores the expected name.
This post-hoc diagnostic is not used for pilot selection or confirmation settings;
it helps distinguish completion of the trained statement from answering a new
assistant question. `compare-learning.bat` runs it after a successful comparison.

## Follow-on retention pilot

After the acquisition comparison selected Q&A and demonstrated acquisition, a
separate one-seed experiment was defined. It starts both arms from the same
seed-42 final 64-exposure confirmation adapter, using the first configured seed
rather than selecting the highest-scoring checkpoint. Eight new synthetic
person/project associations and the existing question templates are fixed in
`qwenbrain/experiment_retention.py` before its training starts.

Both arms run 64 updates of eight rows, padded to the same length. No replay
trains all eight new facts per step. The 50% replay arm trains four new and four
old facts, alternating halves for equal coverage every two steps. Every fact
cycles through all four training question forms. At steps 16 and 64, evaluate
old and new facts separately with fresh context and the same scoring rules.

Replay therefore gives each new fact 8/32 exposures at steps 16/64, while no
replay gives it 16/64. Report the retention/new-learning tradeoff at equal total
batch/step budget; this is not an equal-new-exposure comparison. Keep the initial
checkpoint and base weights unchanged, and save each final adapter separately.
This follow-on design was chosen after seeing acquisition results and is not a
multi-seed confirmation of replay's effectiveness.

Run [compare-retention.bat](../../scripts/compare-retention.bat) after the acquisition run.
See [acquisition results](acquisition-results.md) and
[retention results](retention-results.md) for the measured outcomes.
