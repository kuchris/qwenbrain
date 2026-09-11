# Answer stopping comparison v1

Compare answer-only Q&A (`qa`) against the same Q&A with the assistant's actual
`<|im_end|>` token also supervised (`qa_end`). This follows the name-extension
errors in [acquisition](acquisition-results.md) and [retention](retention-results.md).
It tests a training objective; it does not change live chat or train general replay.

## Fixed comparison

[The config](../configs/stopping.json) defines eight fresh synthetic person/project
associations, disjoint from acquisition and retention subjects and values. Both
arms start from the same clean-base LoRA tensors within each seed (42, 43, 44).
Each step contains one example per fact. Four training question forms are cycled;
the two held-out forms match the earlier protocol. No fact-bearing context is
supplied during Q&A training or evaluation. The primary outcome is normalized
greedy exact answer accuracy at **16 exposures per fact**. Checkpoints at 1, 4,
and 64 are secondary; there is no tuning or winner-dependent confirmation stage.

Model revision, rank, learning rate, fresh optimizer per step, clipping, decoder,
batch dimensions, input IDs, padding, ordering, and all original answer labels
are identical across arms. Only one extra label per example changes: the actual
end-of-turn token immediately following the answer in the rendered chat template.
Question, system, and other template tokens remain masked. No text token spelling
is substituted for the special token. Standard mean loss means adding that token
also changes the relative weighting of answer tokens; this is part of the tested
objective, not an equal-answer-gradient comparison.

## Measurements and checks

Retain raw answers and generated token IDs, exact match, correct name followed by
extra text, termination with the end token, output-limit hits, and exact answers
that also terminate. Greedy decoding remains limited to 24 tokens. Candidate
log-probabilities score answer tokens only for both arms, as in acquisition.
Four unchanged arithmetic/geography questions provide a small regression check;
this cannot establish general quality preservation. Base and supplied-fact-in-
context controls run before training. There is no retrieval or learning at test.

Check that paired input IDs are unchanged and labels differ at exactly the
assistant end token. Verify independent shifted FP64 loss and padded/unpadded
loss for both objectives. Hash adapters around evaluation and the full base
before/after training. At each final checkpoint, reset the adapter, reload the
saved file, and repeat every held-out answer. Record counters, times, source and
checkpoint hashes. The completed result must include all seeds and failures.

All artifacts go to a new `runs/stopping-*/` directory. Live adapter files are not
loaded or written. Double-click [compare-stopping.bat](../compare-stopping.bat).
The 48 query results per arm/milestone are eight facts x two wordings x three
seeds, not 48 independent facts. This is acquisition only, not a repeat of the
sequential retention test or proof of autonomous conversational learning.
