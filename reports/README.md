# Saved experiment evidence

JSON files here preserve measured outputs, configurations, checksums and original
run references. Human-readable protocols and interpretation now live in `docs/`.
The organization change did not rewrite these JSON files. Recorded source hashes
describe the code used by that run, not necessarily a later checkout.

| Experiment | Read first | Raw evidence |
| --- | --- | --- |
| Acquisition formats | [Results](../docs/experiments/acquisition-results.md), [protocol](../docs/experiments/acquisition-protocol.md) | [Acquisition](acquisition-results.json), [completion probes](acquisition-completions.json) |
| Replay and retention | [Results](../docs/experiments/retention-results.md) | [Retention](retention-results.json) |
| End-token supervision | [Results](../docs/experiments/stopping-results.md), [protocol](../docs/experiments/stopping-protocol.md) | [Stopping](stopping-results.json) |
| Four-arm memory comparison | [Results](../docs/experiments/memory-results.md), [protocol](../docs/experiments/memory-protocol.md) | [Memory comparison](memory-results.json) |
| Temporal retrieval | [Results](../docs/experiments/temporal-memory-results.md), [protocol](../docs/experiments/temporal-memory-protocol.md) | [Raw sources](temporal-memory-raw-results.json), [explicit evidence](temporal-memory-explicit-results.json) |
| Typed-source checker | [Results](../docs/experiments/grounded-memory-results.md), [protocol](../docs/experiments/grounded-memory-protocol.md) | [Paired outputs and aggregate](grounded-memory-results.json) |

Implementation evidence is documented separately:

- [Milestone 1: local chat](../docs/history/milestone-1.md) links the model manifest,
  chat checks and original launcher checks.
- [Milestone 2: one-message update](../docs/history/milestone-2.md) links masking,
  gradients, frozen-base and checkpoint-reload checks.
- [Milestone 3: automatic adaptation](../docs/history/milestone-3.md) links the
  online learner and CLI checks.
- [Repetition diagnostic](../docs/history/repetition-fix.md) links before/after
  sampling and launcher measurements.

Some historical diagnostics/checkpoints are local files under Git-ignored `runs/`
and `adapters/`. Their availability on this machine is not evidence that a clean
checkout includes them. Read the [command prerequisites](../scripts/README.md)
before repeating an experiment. New runs do not overwrite these summaries unless
a check's documented output explicitly targets `reports/`.
