# Learning format experiment: results

Q&A supervision made synthetic facts accessible through new question wordings.
Repeating raw statements taught their completion but did not transfer to the
tested assistant questions. This supports changing the training objective before
increasing the complexity of the memory architecture.

These are local GPU measurements, not a claim that arbitrary live conversations
now produce reliable memory. The live adapter and chat implementation were not
changed by this experiment.

## Acquisition

The [fixed protocol](acquisition-protocol.md) compared raw statements, four
declarative forms, and four grounded Q&A forms on four pilot facts. Q&A won the
predefined pilot selection rule. Confirmation compared Q&A against raw on eight
fresh synthetic facts with seeds 42, 43, and 44. Each fact was taught; only the
two evaluation question wordings were held out. Tests used fresh context with
no source facts, conversation history, retrieval, or updates during evaluation.

| Training format | 1 exposure | 4 exposures | 16 exposures | 64 exposures |
| --- | ---: | ---: | ---: | ---: |
| Pilot: raw, seed 42 | 0/8 | 0/8 | 0/8 | 0/8 |
| Pilot: varied, seed 42 | 0/8 | 0/8 | 1/8 | 2/8 |
| Pilot: Q&A, seed 42 | 0/8 | 1/8 | 4/8 | 6/8 |
| Confirmation: raw, all three seeds | 0/48 | 0/48 | 0/48 | 0/48 |
| Confirmation: Q&A, all three seeds | 0/48 | 0/48 | **43/48** | **41/48** |

Scores are greedy exact answers, ignoring punctuation/case/whitespace but not
extra words. The confirmation total is eight facts x two question wordings x
three seeds, **not 48 independent facts**. One exposure is a minibatch containing
every fact, not one ordinary chat message. A single exposure gave no improvement
on these questions.

| Q&A confirmation seed | 16 exposures | 64 exposures |
| --- | ---: | ---: |
| 42 | 15/16 | 14/16 |
| 43 | 12/16 | 14/16 |
| 44 | 16/16 | 13/16 |

At 64 exposures, the correct project name ranked above all seven distractors for
48/48 questions, using mean answer-token log-probability. This is constrained
discrimination, not free generation. For example, outputs such as `Maple Sleet`
fail exact match even when `Maple` ranks first. All arms excluded end-of-turn
tokens from training loss; an explicit stopping-target ablation is still needed
before attributing these extensions to that choice. More training did not
consistently improve the primary score.

The untrained model answered 0/16 confirmation questions without facts and 16/16
with the corresponding fact supplied in context. It could read the facts, while
answering them from updated weights depended strongly on training format.

## What the raw model actually learned

A post-hoc diagnostic loaded each final adapter and asked it to continue the
original user statement just before its project name. This is deliberately a
different task from answering an assistant question; it was not used to choose
the winning format.

| Final confirmation checkpoint | Correct statement completions | Held-out Q&A |
| --- | ---: | ---: |
| Raw, three seeds | **24/24** | 0/48 |
| Q&A, three seeds | 0/24 | **41/48** |
| Untrained base, one evaluation | 0/8 | 0/16 |

Raw training completed `Tova Renn's project is named` with `Maple.`. Its pilot
training loss fell from 6.55 to 0.14, yet pilot Q&A stayed at 0/8. Q&A adapters
produced an empty decoded continuation on this unfinished-user-turn diagnostic;
that failure does not establish that the association was absent internally.

The supported conclusion is format-dependent retrieval of learned associations.
It would be inaccurate to say the raw objective never learned anything or that
Q&A learned a general-purpose, prompt-independent memory.

## Unrelated answers and limitations

The four arithmetic/geography controls all passed for Q&A confirmation at 16
exposures. At 64, seed 44 answered `Beacon` for France's capital. The pilot was
less stable: at 16, three control answers were wrong; at 64, France was still
`Cobourg`. Its `3 + 5 = 8` response was factually correct but failed the strict
number-only format. Raw confirmation also answered `Lyon` for France at 64;
seed 43's `Tokio` failed literal spelling, not necessarily factual knowledge.

These four questions are a small regression probe, not evidence that general
capabilities are preserved. The live reference-loss gate was deliberately not
applied here, so these results describe experimental candidates, not accepted
live updates. First-person identity, corrections, unsupported questions, other
fact relations, realistic message streams, and multiple fact orders remain
untested. Confirmation repeats the same eight associations across three seeds.

## Compute and verification

- Qwen/Qwen3.5-0.8B, pinned revision
  `2fc06364715b967f1860aea9cf38778875588b17`, text-only BF16; RTX 5070 Ti.
- Rank-8 LoRA, alpha 16, learning rate 0.0001, zero dropout/weight decay,
  clipping 1.0, fresh AdamW for each step. Only adapter parameters were trained.
- Nine trials x 64 steps = 576 updates; measured update time totaled **407.65
  seconds**, excluding loading, evaluation, hashing, and adapter I/O. Peak
  PyTorch allocated memory during these updates was **2,540.98 MiB**; this is
  not total device memory usage or a live-chat latency measurement.
- All rows padded to 45 tokens. Within each phase, compared arms had identical
  initial adapter tensors, batch dimensions, step counts, and padded-token
  budgets. Actual input/target counts and time are retained in the evidence;
  equal padding does not mean equal supervision.
- Target-span decoding, independent FP64 next-token loss, and padded/unpadded
  loss checks passed. Adapter checksums stayed unchanged during evaluation.
  Full base-parameter checksums matched before/after training.
- All nine final adapters were reloaded in a fresh process; one held-out Q&A
  response per adapter matched its recorded response. This is a sampled reload
  check, not a rerun of every answer after restart.

The first attempted batched update hit a checkpoint-recomputation backend mismatch
before its first optimizer step. The experimental wrapper now keeps math SDPA
active throughout forward and backward; the completed comparison uses that fix.

## Evidence and rerun

- [Complete acquisition results and raw answers](acquisition-results.json).
- [Direct-completion and reload probes](acquisition-completions.json).
- Original local run and adapters:
  `runs/acquisition-1789131725328701100/`.
- Double-click [compare-learning.bat](../compare-learning.bat) to repeat the
  comparison and completion probes. Each run creates a separate directory.
- [Research and rationale](online-learning-research.md).

The follow-on retention experiment and its implications are recorded in
[retention-results.md](retention-results.md).
