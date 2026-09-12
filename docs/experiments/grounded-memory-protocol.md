# Typed-source answer checking: fixed fresh-data pilot

Protocol fixed before generating answers on the three new fixtures. Data seeds
73, 101 and 137 define 24 distinct known person identities and randomized project
values. Each world has 20 source records, the same three-stage change/correction
timeline as temporal-memory-v1, and 112 queries. Question templates and scenario
structure are reused; this is new-identity/value validation, not held-out language
or arbitrary query understanding. Model decoding remains greedy; these are data
seeds, not three stochastic model-training runs.

## Pipeline and paired comparison

Use the frozen base and explicit evidence presentation from the preceding
temporal experiment. Generate once for each query, then run the checker on that
exact answer. Report raw and checked results side by side. Expected answers,
gold source ids and benchmark categories are available only to evaluation; the
checker receives question text, observation date, candidate answer and the store.

ProjectMemoryStore adds a typed project value keyed by source id to the existing
append-only temporal store. Source text and value commit together. Fixture
encoding supplies trusted structured values; no learned extraction is claimed.
The checker resolves the applicable source again, using the same explicit-name
and ISO-date lookup. A narrow full-question grammar identifies project queries.
Only case and whitespace are normalized when matching an answer to the stored
value. Meaningful punctuation remains significant. An accepted answer returns
the canonical stored value and source id.

Missing/untyped evidence, unsupported questions or a mismatching answer produce
UNKNOWN. Model UNKNOWN is retained even if a correct source exists: this is a
conservative verifier, not extractive answer replacement. Simple arithmetic and
capital questions matching an explicit grammar bypass checking and are marked
outside_scope, never verified. Everything else fails closed. Do not describe
this narrow grammar as a general semantic entailment checker or a live-chat router.

The application can therefore suppress unsupported outputs without improving
the neural model. Report source consistency separately from knowledge accuracy
and count false abstentions and correct answers blocked by the checker. There
is no generated EOS measurement for an application-replaced UNKNOWN; retain the
raw model EOS and exact-plus-EOS fields separately.

## Metrics and checks

Per seed/stage, report current/history accuracy, correct UNKNOWN, unrelated
controls, raw nonexact/nonabstaining personal answers, checked nonexact/nonabstaining personal answers,
false abstentions, decisions, source ids and raw/checked answer pairs. A claim
is counted as nonexact here when it fails the strict field-value format; a verbose
but factually correct sentence can also fail. This is not an entailment judgment. Existing provenance,
database no-write, full frozen-base and live-checkpoint checks also run. A
separate process reopens each final typed database and repeats all 112 retrievals.

CPU tests must reject wrong entities, obsolete answers, unsupported attributes,
future/missing sources, extra answer text and punctuation collisions; preserve
valid answers; retain UNKNOWN; explicitly mark general bypasses; and reject
untyped records. Seven earlier retrieval/temporal tests remain required.

Run `compare-grounded-memory.bat` for all three fixed worlds, or pass `--seed 73`
to run one. All outputs go to new `runs/grounded-memory-SEED-*/` folders. No chat
adapter, optimizer or live-chat code is changed. Preserve the first completed
results on these fixtures; any later tuning needs a separately labeled study.

Reporting clarification after the first completed runs: rename the original
`raw_unsupported_personal_claims`/`checked_unsupported_personal_claims` counters to
nonexact/nonabstaining-answer counters. Inspection found a correct value embedded
in a full sentence; failing exact match must not be called a fabricated fact.
Generation, acceptance rules and paired scores are unchanged. Original run files
retain their legacy counter names; the consolidated summary uses corrected names.
