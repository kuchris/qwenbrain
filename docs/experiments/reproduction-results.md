# Four-arm GPU pipeline reproduction

All four stages completed from newly trained adapters on 2026-09-12. The original
headline scores were reproduced: statement completion **24/24** versus QA
**0/48**, supervised QA **43/48** at 16 exposures and **41/48** at 64, and
SQLite retrieval **32/32** versus the selected QA/replay setting's **25/32**.

This verifies the new execution path for the existing synthetic experiment. It
does not add independent data or establish that retrieval always beats weights.

## What was rebuilt

The staged source was exported into a separate directory containing no model,
run, adapter, conversation, cache or virtual-environment directories. A fresh
environment was installed with `uv sync --locked --link-mode copy`. Only the
pinned pretrained model files were copied from the existing project; all 12
files were checked against the saved download manifest. No old trained adapters
or experiment outputs were copied into this checkout.

The run then executed:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m qwenbrain.reproduce run --output runs/full-reproduction
```

Acquisition rebuilt nine adapters: three pilot formats and raw/selected-QA
confirmation at seeds 42, 43 and 44. Completion probes reloaded those files.
Retention used the new seed-42 QA64 adapter. The four-arm comparison used the
explicit new retention report and repeated the previously selected 16-step
50% replay setting from that QA64 adapter. The live chat adapter was never used.

Environment: Windows, Python 3.11.16, RTX 5070 Ti, PyTorch 2.11.0+cu128,
Transformers 5.16.1, PEFT 0.20.0, Qwen3.5-0.8B in BF16 at revision
`2fc06364715b967f1860aea9cf38778875588b17`.

## Recounted results

| Arm | Old answers | New answers | All facts | Unrelated controls |
| --- | ---: | ---: | ---: | ---: |
| Frozen, no evidence | 0/16 | 0/16 | 0/32 | 4/4 |
| Oracle source | 16/16 | 16/16 | 32/32 | 4/4 |
| SQLite full-name lookup | 16/16 | 16/16 | 32/32 | 4/4 |
| QA + 50% replay, 16 steps | 12/16 | 13/16 | 25/32 | 3/4 |

There are 16 fact identities with two question forms each. Oracle and retrieval
each selected the correct source on 32/32 fact queries. W again answered
`Beacon` for France's capital. Correct-and-stopped counts equaled exact-answer
counts for these arms; an eventual end token did not prevent incorrect answers.

Acquisition's QA confirmation counts were 15/16, 12/16 and 16/16 at 16
exposures, and 14/16, 14/16 and 13/16 at 64. These use eight shared identities
and two query forms across three training seeds. The newly trained seed-42 QA64
file also matched the historical checkpoint's SHA-256.

At 64 retention steps, no replay preserved 10/16 old answers and 50% replay
preserved 12/16; both produced 11/16 new answers. Neither retained all four
unrelated controls. See the saved raw outputs for every checkpoint and failure.

| Stage | Wall time in this run |
| --- | ---: |
| Acquisition | 906.03 s |
| Completion and reload probes | 46.58 s |
| Retention | 185.13 s |
| Four-arm comparison | 66.06 s |

The stages totaled about 20 minutes, excluding environment setup and the model
preflight. Stage times include model loading and evaluation. This was the same
desktop GPU with other applications present, not an isolated speed benchmark.

## Verification and evidence

- Every stage exited successfully, with a completed report and hashed inputs.
  The pipeline checked that source/config files stayed unchanged during the run.
- Full frozen-base hashes matched before/after and across training stages. The
  source project's live adapter and pre-existing diagnostic file stayed unchanged.
- Run source bytes matched the recorded SHA-256 values. Staged Git content
  matched that source after CRLF-to-LF normalization only; the Windows export and
  Git blobs have different newline bytes.
- Re-running the summary with `python -S` succeeded without third-party packages.
  Summary JSON/Markdown were byte-identical, and no raw report was rewritten.
  Scores are counted from saved answers rather than trusted cached score fields.
- CPU artifact tests cover incomplete/tampered/mixed inputs, relocation and
  output preservation. The repository has 26 standard-library test methods.

[Saved evidence](../../reports/reproduction-results.json) contains the export
procedure, environment, pipeline manifest, recount verification, summary and all
four raw report payloads. Original report-file hashes are retained in the
manifest; the historical experiment JSON files were not replaced. Model weights,
adapters and local logs remain outside Git.

The complete local bundle is under
`runs/reproduction-verification-1789208372895007300/checkout/runs/full-reproduction/`.
Follow the [reproduction guide](../reproduction.md) to create your own bundle.

The stopping, temporal-memory and grounded-memory experiments were not rerun in
this pipeline. Independent question/scenario templates, broader controls and a
public showcase remain separate portfolio work.
