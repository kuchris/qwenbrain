# Typed-source answer checker: results

A conservative checker now validates personal-project answers against the
applicable typed value in SQLite. On three fixed fresh-data worlds it preserved
all exact-format correct answers and did not improve exact-match accuracy. The raw
model already abstained on all unknown questions. The checker retained 14 false
abstentions and converted one correct-but-verbose historical answer into UNKNOWN. A
separate regression replay confirmed that the prior unsupported RED answer is
blocked without filling in the model's other UNKNOWN answers.

## Fresh-data paired results

Three data seeds (73, 101, 137), 24 distinct known people, 60 total versioned
records, and three observation stages per world. Each query generated exactly
one raw answer; the checker then evaluated that same answer. No model parameters
were updated. The expected answer and gold source id were not checker inputs.

| Capability | Raw model | Checked output | Correct raw answers blocked |
| --- | ---: | ---: | ---: |
| Current project, including corrections | 144/144 | 144/144 | 0 |
| Historical project | 33/48 | 33/48 | 0 |
| UNKNOWN when unsupported | 108/108 | 108/108 | 0 |
| Arithmetic/capital controls | 36/36 | 36/36 | 0 |

These are repeated queries/stages, not 144 independent current facts or 36
independent general-knowledge tasks. Names and values are fresh; templates and
scenario structure are reused. Seeds randomize data assignments, not stochastic
model training. The four unique controls bypass verification and must not be
described as source-verified answers.

| Data seed | Current, raw/checked | History, raw/checked | UNKNOWN, raw/checked |
| --- | ---: | ---: | ---: |
| 73 | 48/48 | 11/16 | 36/36 |
| 101 | 48/48 | 9/16 | 36/36 |
| 137 | 48/48 | 13/16 | 36/36 |

All 192 answerable source selections were correct. Fourteen historical outputs
were already UNKNOWN. One other output was `Kora Jade's project was named Aviary.`
The source value was indeed Aviary, but the checker rejected the full sentence
because the protocol requires only the field value. It therefore increased false
abstentions from 14 to 15 while leaving strict exact-match accuracy unchanged.

The zero in the table means zero *exact-format* correct answers blocked; one
factually correct verbose answer was blocked. This is a real availability cost
of strict matching. There was no observed invented personal fact in the fresh
raw answers, so this run does not demonstrate hallucination reduction; do not
attribute the perfect unknown score to the checker.

## Prior-error regression replay

The CPU regression test replays all 112 recorded R answers from the previous
explicit temporal experiment against a typed copy of its source stream, using
each original observation cutoff. This is previously seen development evidence,
separate from the fresh GPU evaluation.

- The actual RED answer to Aster Rook's favorite-color question becomes UNKNOWN.
- All three previous false abstentions remain UNKNOWN. The checker does not
  substitute the source's project name for an abstaining model.
- No previously correct answer becomes incorrect.

## What the checker enforces

Source records now persist a typed project value alongside source text and dates.
For the supported personal-project grammar, the checker independently retrieves
the applicable row. It allows a complete value match after case/whitespace
normalization, returns the canonical stored value and source id, and blocks
missing/untyped sources, unsupported question forms, wrong attributes, obsolete
answers and extra text. Meaningful punctuation is preserved: C, C++ and C# are
different values. A rejected answer becomes application-generated UNKNOWN.

This is a source-consistency constraint conditional on a correct typed source and
query interpretation. It does not establish that the source itself is true.
Typed values are supplied by the fixture encoder; automatic extraction from
arbitrary conversation is not implemented. Simple arithmetic/capital grammar
bypasses the checker, is labeled outside_scope, and carries no verification claim.
Other unsupported query forms fail closed, so this is not a general chat router.

Raw model end-token details remain recorded. An UNKNOWN inserted by the checker
is not a model-generated EOS success or evidence that model weights improved.
Original run JSON used an overly broad `unsupported_personal_claims` counter name
for nonexact/nonabstaining outputs. The consolidated summary and future runner
use the corrected name; the one raw case is the correct-but-verbose Aviary answer.
This reporting correction does not change generation, decisions or paired scores.

## Verification

- Fourteen CPU test methods passed, including the previous actual-error replay,
  wrong attributes/entities, stale/future answers, extra text, punctuation
  collisions, retained abstentions, untyped records and the earlier temporal
  persistence tests.
- Each new typed database was reopened in a separate Python process; all 336
  retrieval results matched, including earlier observation cutoffs against each
  final database.
- Full base hashes, source/config hashes and live-checkpoint checks passed in
  all three runs. Database hashes matched around every evaluation.
- Paired scores were independently recounted from the saved raw/checked outputs.

Completed runs on 2026-09-12 Asia/Tokyo:

- `runs/grounded-memory-73-1789144865647412000/`
- `runs/grounded-memory-101-1789144888404735300/`
- `runs/grounded-memory-137-1789144914440547700/`

[Complete paired outputs, source records and checksums](../../reports/grounded-memory-results.json)
and [fixed protocol](grounded-memory-protocol.md) are saved with this report.
Use [compare-grounded-memory.bat](../../scripts/compare-grounded-memory.bat) to repeat all
worlds, or pass `--seed 73` for one. Live chat is unchanged.

The remaining measured limitation is historical answer availability: the source
is present but the model sometimes says UNKNOWN. A verifier can prevent an
unsupported assertion while leaving that problem intact. A separately labeled
extractive source-answer path would address a different requirement and must not
be presented as a model-memory improvement.
