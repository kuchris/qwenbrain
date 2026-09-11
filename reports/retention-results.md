# Retention after learning new facts

In this one-seed follow-on test, replay preserved more exact old-fact answers
than training only new facts at the same batch/step budget. It did not fully
preserve the starting behavior, and both arms developed unrelated answer errors.
The result supports further offline work; neither adapter was promoted to chat.

## Setup

Both arms started from the same final seed-42 Q&A adapter from
[the acquisition comparison](acquisition-results.md), chosen by a fixed
first-seed rule rather than its score. It initially answered **14/16** old-fact
questions and **0/16** new-fact questions correctly. The old and new sets each
contain eight synthetic person/project associations, tested with two held-out
question wordings and no history or retrieval.

Both arms ran 64 updates of eight examples, padded to 46 tokens. No replay used
eight new examples per step; 50% replay used four new and four old, alternating
halves. Thus replay halves new-fact exposure at equal total batch/step budget.
All four training question forms are cycled. The learning settings and evaluation
rules match acquisition. This design was defined after acquisition succeeded;
it is a one-seed pilot, not a second multi-seed confirmation.

## Results

| Condition | Update steps | Exposures per new fact | Old answers | New answers | Unrelated controls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Starting adapter | 0 | 0 | 14/16 | 0/16 | 4/4 |
| No replay | 16 | 16 | 8/16 | 12/16 | 3/4 |
| 50% replay | 16 | 8 | 12/16 | 13/16 | 3/4 |
| No replay | 64 | 64 | 10/16 | 11/16 | 2/4 |
| 50% replay | 64 | 32 | 12/16 | 11/16 | 3/4 |

Replay improved old-answer exact match by two questions at step 64 while both
arms scored 11/16 on new answers. The 16-step result also favored replay in this
run despite fewer new exposures. These are observations from one initialization
and one old/new partition; they do not establish a general effect size or an
optimal replay ratio.

## Inspecting the failures

Several failures were project-name extensions: `Maple Saffron`, `Falcon 2`, and
`Nimbus 2`. No replay also answered `Tundra` when the old fact required `Orchid`;
Tundra was a newly taught project. That is a concrete interference example.
Replay avoided that substitution at the measured checkpoints, but still extended
other names. Its final Silver answer included an unexpected thinking delimiter.

All old-answer candidate ranks remained correct throughout both arms, and final
new-answer ranks were also 16/16. Candidate lists were scored separately within
each eight-fact set: an old-fact ranking **does not include the eight new values**.
Consequently this measure can miss cross-set interference such as Orchid/Tundra.
It also does not measure stopping or whole-response correctness. The drop in
exact answers should be described as degraded recall behavior, not proof that
the underlying association was erased from the weights.

The control errors were substantive: no replay ended with `Jun 2025` for France's
capital and `Juniper` for Japan's; replay ended with `Falcone` for France's.
Neither method maintained all four unrelated answers. These controls are far
too small to assess overall model quality, but their failures already rule out
claiming this setting preserves general chat behavior.

## Verification and evidence

Both arms began with identical adapter checksums. Each processed 23,552 padded
tokens by step 64. Replay processed 256 old plus 256 new examples; no replay
processed 512 new. Actual nonpadding and supervised token counts are recorded.
Measured update time was 45.05 seconds without replay and 44.27 seconds with it,
excluding evaluation and I/O. Peak PyTorch allocation was 2,561.14 MiB; this is
not total VRAM usage or per-message serving latency.

Both 64-step runs completed with finite trainable weights. Evaluation preserved
adapter checksums, the full frozen-base checksum matched before/after, and the
source acquisition checkpoint file remained byte-for-byte unchanged. Saved
retention adapters were not independently reloaded for a second evaluation;
the acquisition adapter reload checks are a separate result.

- [Full retention results, raw answers, counters, and checksums](retention-results.json).
- Local outputs: `runs/retention-1789132854343756800/`.
- [Protocol](acquisition-protocol.md#follow-on-retention-pilot).
- Double-click [compare-retention.bat](../compare-retention.bat) to repeat using
  the latest completed Q&A acquisition run's seed-42 final checkpoint.

## Next bounded experiment

Keep grounded Q&A as the candidate training format. Compare answer-only training
with answer-plus-end-of-turn supervision to test whether stopping errors improve;
keep all other settings fixed and use fresh pilot/confirmation facts. Include
all old and new project names in retention distractors, retain greedy answers as
the main measure, and broaden unrelated and unsupported-question controls.

Then repeat retention across seeds and fact orders, testing a small amount of
general instruction replay alongside fact replay. Only after that should a
separate first-person chat experiment test trustworthy fact extraction,
corrections, identity, one-message latency, and the live acceptance gate.
Every message can trigger the pipeline, but this experiment does not demonstrate
that one gradient update per message is sufficient or that every message contains
a teachable factual answer.
