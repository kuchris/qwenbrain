# Four-way memory comparison

On the existing 16 synthetic facts, both oracle source evidence and a SQLite
full-name lookup produced 32/32 exact answers with a normal end token. The
reproduced Q&A plus 50% replay setting produced 25/32 and changed France's capital
from Paris to Beacon. This supports external lookup for this narrow task; it
does not establish that retrieval is universally better than parameter memory.

## Measured results

Run: `runs/memory-1789143609186993800/`, 2026-09-12 Asia/Tokyo. Same pinned local
Qwen3.5-0.8B revision, BF16 frozen base, original question forms and system prompt,
greedy decoding, 24-token maximum, and fresh context for every question.

| Arm | Old facts | New facts | All exact answers | Exact and stopped | Unrelated controls | Mean query time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F: frozen, no evidence | 0/16 | 0/16 | 0/32 | 0/32 | 4/4 | 0.515 s |
| O: correct source supplied | 16/16 | 16/16 | 32/32 | 32/32 | 4/4 | 0.167 s |
| R: SQLite full-name lookup | 16/16 | 16/16 | 32/32 | 32/32 | 4/4 | 0.167 s |
| W: Q&A + 50% replay, 16 steps | 12/16 | 13/16 | 25/32 | 25/32 | 3/4 | 0.250 s |

There are 16 independent fact identities with two question wordings each, not 32
independent facts. W uses one seed (42). It starts from the existing QA64
acquisition checkpoint and reproduces the earlier retention protocol's 16 replay
steps, including answer-only supervision. Its pre-replay old-fact score was
14/16 with 4/4 controls; after replay that became 12/16 and 3/4. The old/new
post-replay exact scores match the previous retention run.

R retrieved the correct source on 32/32 fact questions (Recall@1 = 1.0), injected
no memory on the four unrelated questions, and produced exactly the same token
sequences as O on all 36 questions. This is an intentionally easy explicit-name
lookup. The store never receives a gold id, expected answer or test-question
template; its inputs are the source records and the live query string.

## Errors and stopping

W's seven fact failures were:

- Maple became `Maple Saffron` and `Maple Squirrel`.
- Falcon became `Falcon 2` in both question forms.
- Granite became `Sequoia` in one form.
- Nimbus became `Nimbus 2` in one form.
- Silver became `Silver Sequoia` in one form.

W answered `Beacon` for France's capital. Every W fact answer ended with the
correct end token, including the wrong ones: ending eventually is not the same
as ending immediately after the correct name. R/O produced the exact name and
end token in all 32 cases. F ended in 20/32 cases and hit the output budget in 12.

## Costs

- SQLite: 16 raw source statements; 8,192 bytes; create/write/commit/close took
  0.0105 s, excluding model loading and any learned fact extraction.
- W: the 16 replay updates took 11.629 s of measured update compute, or 12.539 s
  including the update loop's packing and checks. Saving the 21,694,944-byte
  adapter took another 0.0353 s. This excludes historical acquisition training,
  evaluation, base-model loading and checkpoint reload.
- Mean prompt length across the 36 questions was 45.56 tokens for F/W and 65.83
  for O/R. Query times in the table include lookup, prompt preparation and
  generation; mean generation-only times were 0.5143/0.1663/0.1661/0.2489 s for
  F/O/R/W respectively. Loading and one warm-up query are excluded.

Do not interpret these times as a general retrieval speedup. F often emits long
incorrect answers while O/R emit short names; W also executes a LoRA adapter.
This was one fixed-order run on the local RTX 5070 Ti with other desktop/GPU
applications present, not an isolated latency benchmark.

## Verification

- Two CPU test methods passed, covering reopen persistence, absent-until-written
  sources, punctuation/case/whitespace, name-prefix collisions, ambiguous queries,
  unrelated questions and SQL-like query text.
- Saved W adapter values were discarded, reloaded, and matched by full adapter
  checksum before final evaluation. This was an in-process reload, not a
  separate-process restart test.
- Adapter hashes were unchanged during every evaluation. The full frozen-base
  checksum matched before and after the complete run.
- SQLite was closed and reopened before retrieval. Its bytes/hash stayed
  unchanged throughout evaluation, so queries did not write new memories.
- The source acquisition checkpoint, source/config hashes, and live checkpoint
  remained unchanged. No experiment adapter was promoted to chat.

## Interpretation and next boundary

The model can read a supplied source and return its project name; on this task
the remaining lookup mechanism need not be a learned neural module. The result
justifies testing external memory further before adding consolidation training.

These facts and the W setting were reused from earlier experiments. The W
configuration was selected using earlier observations, and this pilot does not
compare all plausible parameter-learning methods. Only four unrelated controls
were checked. No claims are made about general chat quality, first-person fact
extraction, pronouns, aliases, semantic search, updates, forgetting/deletion or
large stores. Next, use fresh facts and test ambiguous identities, explicit
corrections, historical questions and unknown-answer behavior.

- [Fixed protocol](memory-protocol.md)
- [Complete raw outputs, evidence, timings and checksums](../../reports/memory-results.json)
- [Repeat the experiment](../../scripts/compare-memory.bat)
