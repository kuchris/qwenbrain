# Frozen, oracle, retrieval, and parameter memory pilot

This diagnostic reuses the retention experiment's 16 synthetic person/project
facts, two held-out question forms, and four unrelated controls. It is not a
fresh-data confirmation. Fix this protocol before running; do not select settings
from the new comparison's answers.

## Four arms

- F: frozen base, no memory evidence.
- O: frozen base with exactly one correct original statement supplied by its gold
  source id. Controls have no memory evidence.
- R: frozen base with a SQLite lookup using only the question. Full stored names
  must match on normalized word boundaries. Zero or multiple matching names
  return no evidence. Return one original statement; never index test questions.
- W: reload the existing acquisition seed42 QA64 checkpoint, then repeat 16 steps
  of the previously tested 50% old/new replay. Each step has four old and four new
  examples, cycling the original four training QA forms. Answer-only labels and
  all optimizer settings match retention-v1. This setting was chosen using the
  previous retention results, not claimed to be an independently selected optimum.

All arms use the same pinned base revision, original system prompt, greedy
generation, 24 generated-token budget, and actual assistant end token. Every
query builds a fresh prompt/cache. O/R insert the statement as a preceding user
turn followed by `Understood.`, exactly as in acquisition's context control.
No evaluation question causes training or memory writes. Training and experiment
outputs stay under a new `runs/memory-*/` directory. The live adapter is not loaded
or changed.

Measure exact match, correct answer plus EOS, EOS alone, output-budget truncation,
old/new recall separately, control correct-to-wrong changes, retrieval Recall@1,
and generation/query latency. Preserve raw answers, token ids and evidence. Exact
match uses the original acquisition normalization; it is not a semantic judge.
Report SQLite write/storage costs separately from replay update and adapter IO
costs. The existing acquisition checkpoint is a sunk historical training cost;
incremental replay time must not be described as total parameter-memory cost.

## Checks and limits

CPU tests cover reopen persistence, absent facts, punctuation/case, name prefix
collisions, ambiguous names and non-memory queries. GPU checks compare full base
hashes, adapter hashes before/after evaluation, source/checkpoint hashes, and the
SQLite hash before/after evaluation. Reload the saved W checkpoint after replacing
its in-memory adapter with the initial state, then verify the complete adapter
checksum. This is an in-process reload; it is not a separate-process restart test.

Run F/O/R before attaching W. No lower-level parameter switches are needed for
frozen arms. Warm up generation once before measured evaluations. GPU timings
are one local run, not isolated performance benchmarks; arm order is fixed.

This pilot has 16 facts, not 32 independent facts. Two question forms are repeated
measurements; there is only one LoRA seed and no fresh random fact assignment.
Explicit-name retrieval is deliberately easy and says nothing about pronouns,
aliases, semantic search, conflicting updates, deletion or large memory banks.
Four unrelated controls can expose a regression but cannot establish general
capability preservation. W's pre-replay old-fact result provides a retention
reference; the main comparison evaluates the same final 16-fact state for all arms.

## Reproduce

Run `scripts\compare-memory.bat --retention-report runs/<retention-run>/results.json`,
or pass the same argument to `.venv\Scripts\python.exe -X utf8 -m
qwenbrain.experiment_memory`. The explicitly selected report must reference an
available acquisition checkpoint with a matching checksum. The
[full reproduction pipeline](../reproduction.md) rebuilds all required training
inputs. CPU tests: `.venv\Scripts\python.exe -m unittest discover -s tests -v`.
