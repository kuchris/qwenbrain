# End-of-response supervision: results

Adding the assistant end token improved exact fact answers on this fresh synthetic
set. At the predefined primary checkpoint of 16 exposures, answer-plus-end training
scored **48/48**, compared with **44/48** for answer-only training. Both passed the
four unrelated controls in all three seeds at that checkpoint.

At 64 exposures, end-token training still answered every fact correctly but failed
both capital questions in every seed. It improved answer behavior on the trained
facts without protecting unrelated knowledge under longer training. These adapters
were not promoted to live chat.

## Controlled comparison

The [protocol](stopping-protocol.md) and [fixtures](../configs/stopping.json) were
fixed before training. Eight synthetic person/project associations were fresh
relative to both preceding experiments. Each arm used the same clean base and
identical starting LoRA tensors within seeds 42, 43, and 44. Input IDs, padding,
question forms, batch size, ordering, optimizer settings, and generation settings
were unchanged across arms. The only label difference was supervising the actual
assistant `<|im_end|>` token immediately after the answer.

The standard mean loss includes that additional token, so this also changes the
relative weight assigned to answer tokens. We did not separately isolate end-token
learning from that weighting effect. The token already occurred in the rendered
training inputs of both arms; it was masked from the loss in the answer-only arm.

Each exposure is an eight-example minibatch containing all eight facts. Four
training question forms are cycled, and two different wordings are evaluated from
fresh system-only context. No facts or retrieval results are supplied at test.
The 48 results are eight facts x two wordings x three seeds, not 48 independent
facts. There was no checkpoint selection or tuning using these results.

## Fact answers and unrelated controls

| Exposures per fact | Answer-only exact answers | Answer + end exact answers | Answer-only controls | Answer + end controls |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0/48 | 0/48 | 12/12 | 12/12 |
| 4 | 0/48 | 0/48 | 12/12 | 12/12 |
| **16 (primary)** | **44/48** | **48/48** | **12/12** | **12/12** |
| 64 (secondary) | 43/48 | 48/48 | 11/12 | 6/12 |

The control total repeats the same four arithmetic/geography questions across
three seeds. It is not a broad capability benchmark. At 64, the end-token arm
answered `Basalt` for France's capital and `Hazel` for Japan's in all three seeds.
The answer-only arm answered France with `Basalt` in seed 42; its other controls
passed. This is a substantive regression, not a punctuation-scoring issue.

| Seed | Answer-only at 16 | Answer + end at 16 | Answer-only at 64 | Answer + end at 64 |
| --- | ---: | ---: | ---: | ---: |
| 42 | 14/16 | 16/16 | 15/16 | 16/16 |
| 43 | 14/16 | 16/16 | 13/16 | 16/16 |
| 44 | 16/16 | 16/16 | 15/16 | 16/16 |

The untrained baseline answered 0/16 without facts and 16/16 when the corresponding
source fact was supplied in context. Neither training arm demonstrated useful
one-exposure recall on these questions.

## Did stopping actually improve?

At 64 exposures, answer-only training generated five name extensions across the
48 questions, including `Hazel Creek` instead of `Hazel` and `Garnet Valley` instead
of `Garnet`. Answer-plus-end training generated zero such extensions and gave
48 exact answers followed by the end token. At 16, the extension counts were two
versus zero; not all answer-only failures were extensions.

All 48 answer-only outputs also eventually ended with the end token at 64. The
problem was therefore stopping **after the correct name**, rather than an inability
to terminate at all. At 16, answer-only produced one output-limit hit; the end-token
arm produced none. Exact-answer-and-stopped counts matched exact-answer counts
at both 16 and 64 for both arms. Candidate ranking scored answer tokens only and
was 48/48 at both checkpoints for both arms, despite the free-generation differences.

This supports answer-plus-end supervision as the next candidate objective for
this narrow Q&A task. It does not establish robust memory for arbitrary facts,
speaker identity, corrections, or ordinary chat. The older retention stream was
not rerun here, so no improved retention claim follows from this experiment.

## Runtime and verification

The pinned text-only Qwen3.5-0.8B ran in BF16 on the RTX 5070 Ti. LoRA rank 8,
alpha 16, learning rate 0.0001, zero dropout/weight decay, clipping 1.0, and a fresh
AdamW per step matched the earlier experiment. All training batches were padded
to 45 tokens and used the same math attention backend through forward/backward.

Six trials x 64 updates = 384 updates. Measured update time totaled **271.81
seconds**, excluding evaluation, loading, hashing, and checkpoint I/O. Peak PyTorch
allocated memory during updates was **2,540.98 MiB**, not total device memory usage.
Each arm processed 23,040 padded tokens per seed by step 64. Answer-only supervised
1,216 tokens per seed; answer-plus-end supervised 1,728, reflecting one extra
target per training example. Full counters and per-step timing are in the JSON.

All six runs completed. Target checks confirmed exactly one changed label per
example at the real assistant end token. Independent shifted FP64 loss and
padded/unpadded loss checks passed for both objectives. Evaluation preserved
adapter checksums. The full frozen-base checksum matched before/after training.
Both arms shared the recorded initial adapter checksum within each seed.

Each final adapter was saved, deliberately reset, then reloaded in the same
process. All 16 final answer texts per adapter matched their pre-save outputs:
96/96 across six reloads. This checks adapter reload after reset, not a separate
process restart. Source and checkpoint file hashes also passed verification.

## Evidence and next step

- [Full raw results, generated token IDs, counters, and hashes](stopping-results.json).
- Local run and six adapter files: `runs/stopping-1789133726536020600/`.
- Double-click [compare-stopping.bat](../compare-stopping.bat) to repeat in a new
  output directory. The Python module was executed for this run.

Next, compare fact-only Q&A against the same end-token objective with a small
general-instruction replay set. Evaluate unrelated questions whose answers were
not simply copied into that replay set, alongside acquisition and old-fact
retention. Keep this a separate controlled experiment before integrating automatic
fact extraction and Q&A generation into live chat.
