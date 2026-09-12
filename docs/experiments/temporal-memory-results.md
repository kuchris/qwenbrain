# Temporal memory and unknown-answer results

Versioned SQLite retrieval selected the correct source for every answerable query
in this pilot. The remaining errors occurred when the frozen model interpreted
the evidence or decided whether to abstain. Explicit lookup-result presentation
improved the measured behavior, but did not make it completely reliable.

## Setup and completed runs

Eight fresh synthetic identities, 20 append-only project records and three
observation stages: initial names, changed names, then corrections. The correction
stage fixes two current names and two historical names without replacing the
newer current state. Each stage uses fresh context for every question. Three
arms compare no evidence (F), gold relevant evidence (O), and SQLite lookup (R).
No model parameters or live adapter were updated.

- Raw source run: `runs/temporal-memory-1789144120764647100/`.
- Explicit presentation follow-up: `runs/temporal-memory-1789144281143662500/`.
- Both completed on 2026-09-12 Asia/Tokyo using the pinned local model and greedy
  generation, with 24 output tokens and the actual assistant end token.

The raw run already instructs the model to use memory for personal questions and
say UNKNOWN when unsupported. The follow-up additionally labels empty lookup
results, makes the source's effective/received dates visible, and explains that
project states persist until superseded and date selection has already been
performed. It does not supply expected answers or bypass generation for UNKNOWN.
See the [fixed protocol and declared follow-up](temporal-memory-protocol.md).

## Retrieval arm by capability

| R configuration | Current answers | Historical answers | Correct UNKNOWN | Unrelated controls |
| --- | ---: | ---: | ---: | ---: |
| Raw source sentences | 33/48 | 5/16 | 22/36 | 12/12 |
| Explicit lookup result | 48/48 | 13/16 | 35/36 | 12/12 |

These counts aggregate repeated stages and question forms. They do not represent
48 independent current facts, 36 independent unknown cases, or 12 independent
general-knowledge tasks. There are eight fact identities and four unique controls.

| Explicit R stage | Current | History | Correct UNKNOWN |
| --- | ---: | ---: | ---: |
| Initial | 16/16 | Not queried | 12/12 |
| Changed | 16/16 | 6/8 | 12/12 |
| Corrected | 16/16 | 7/8 | 11/12 |

All correct answers in these category counts also ended with the proper end token.
Current queries after the late historical corrections still used the newer state.
The corrected historical names were available for retrospective questions.

## Full comparison and diagnostic interpretation

| Presentation | Arm | Known-answer accuracy | Correct UNKNOWN |
| --- | --- | ---: | ---: |
| Raw | F | 0/64 | 21/36 |
| Raw | O | 38/64 | 21/36 |
| Raw | R | 38/64 | 22/36 |
| Explicit | F | 0/64 | 33/36 |
| Explicit | O | 61/64 | 33/36 |
| Explicit | R | 61/64 | 35/36 |

R selected the gold source on 64/64 answerable queries in each run. O and R had
the same known-answer accuracy, so the observed known-answer errors are not
attributable to selecting the wrong database row in these tests. Raw-v1 also had
a presentation ambiguity: `As of DATE` did not explicitly explain the store's
persistent-state semantics. The follow-up changes several presentation details
together; it cannot identify the contribution of each individual sentence.

O has no relevant source for favorite-color questions. R retrieves a project
record for that person, which is irrelevant to the requested attribute. Explicit
R's higher UNKNOWN score than O in this small sample does not mean retrieval is
better than an oracle at finding relevant evidence: the additional project-only
context also changes the reader's behavior.

The explicit format was designed after seeing the raw results and evaluated on
the same data. Treat the improvement as development evidence, not independent
confirmation. The shared UNKNOWN instruction also differs from memory-v1, so
these numbers are not a direct prompt-controlled comparison with the prior LoRA
experiment.

## Remaining failures

Explicit R produced three false abstentions despite correct evidence:

- Aster Rook's project on 2026-01-15: expected Cobalt, got UNKNOWN after both
  the changed and corrected stages.
- Cato Finch's project on 2026-01-15: expected Opal, got UNKNOWN after the
  changed stage. After the historical correction, the Laurel answer was correct.

It also fabricated one unsupported answer: after the corrected stage, it answered
RED for Aster Rook's favorite color, although the retrieved source only gave the
project name Vesper. This was a substantive unsupported claim, not just a failure
to use the exact UNKNOWN spelling. All four erroneous outputs ended normally.

The explicit run correctly abstained on all 12 unknown-person queries, six
before-first-record queries, six future queries and six ambiguous queries. It
abstained on five of six unsupported-attribute queries. These are repetitions
across stages; the store supports only full names and explicit ISO dates.

## Verification and artifacts

- Seven CPU test methods passed: two existing retrieval tests and five temporal
  tests. Coverage includes late historical corrections, same-effective-date
  corrections, observation cutoffs, future dates, ambiguous identities, invalid
  batches, and reopen persistence.
- For each run, a separate Python process reopened the final 8,192-byte SQLite
  database and reproduced all 112 R retrieval results. Those include early-stage
  queries executed with the earlier observation cutoff against the final store.
  Later observations did not leak into earlier results. The child process did
  not reload the neural model or repeat generation.
- SQLite hashes matched before/after every evaluation. Full frozen-base hashes,
  source/config hashes and the live-checkpoint hash checks all passed.
- Generation outputs, presented memory, source ids, dates, token ids and timing
  details are retained in the explicit report. The raw report predates the added
  `presented_memory` diagnostic field; its source evidence and configuration are
  still recorded.

[Raw results](../../reports/temporal-memory-raw-results.json) and
[explicit results](../../reports/temporal-memory-explicit-results.json) contain the complete
measurements. Run [compare-temporal-memory.bat](../../scripts/compare-temporal-memory.bat)
for raw sources; pass `--evidence-format explicit` for the follow-up.

The next engineering boundary is enforcing agreement between the answer and a
typed, applicable source, then validating on fresh held-out facts. Increasing
memory capacity or adding consolidation is not justified by these particular
errors: the correct record is already available, but the reader can ignore it
or invent an unsupported attribute. Automatic conversational encoding, arbitrary
date expressions, multiple relations and unresolved source conflicts remain
outside this experiment.
