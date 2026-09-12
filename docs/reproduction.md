# Reproduce the four-arm GPU memory experiment

This pipeline rebuilds the acquisition checkpoint and runs the original
acquisition, completion, retention and four-arm memory protocols in order.
It uses explicit report paths and checksum-verified checkpoints. It never
selects a run by modification time or imports the live chat adapter.

The target is reproducibility of this existing synthetic experiment. A new run
is not new-data confirmation, a broader comparison of learning algorithms, or
proof that all LLM facts should live outside weights.

## One command after setup

On the measured Windows / RTX 5070 Ti environment, from the project root:

```powershell
.\setup.bat
.\scripts\reproduce-memory.bat --output runs/my-reproduction
```

`setup.bat` installs the locked Python 3.11 dependencies and downloads the pinned
Qwen3.5-0.8B snapshot. The runner checks package versions and hashes every model
file against the project's pinned manifest before launching any experiment.
GPU stages then load local files only. Pretrained model weights are required;
previously trained adapters, old run directories and conversation data are not.

The equivalent direct command is:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m qwenbrain.reproduce run --output runs/my-reproduction
```

Omit `--output` for a unique `runs/reproduction-*/` directory. Explicit output
directories must be new and inside `runs/`; an existing directory is refused.
The pipeline performs real GPU training and evaluation, including nine
acquisition training runs. It is not a quick startup smoke test.

## Inputs and outputs

The default inputs are `configs/acquisition.json`, `configs/learning.json`,
`configs/chat.json`, `uv.lock`, the pinned model manifest and tracked source code.
Use `--config` and `--learning-config` to select explicit project-local files.
The pipeline keeps acquisition-v1's four pilot facts, eight confirmation facts,
three confirmation seeds, four training forms, and 1/4/16/64 exposure schedule.
Retention-v1 requires QA to win the original selection rule and requires its
seed-42 final 64-exposure checkpoint. It stops if those prerequisites are unmet.

```text
runs/my-reproduction/
  inputs/acquisition.json       Copied experiment input
  inputs/learning.json          Copied learning input
  acquisition/results.json     Scores and checkpoint hashes
  acquisition/*.safetensors    Newly trained adapters
  completions.json              Direct-completion and reload probes
  retention/results.json       Replay/no-replay measurements and input hashes
  retention/*.safetensors      Separate retention adapters
  memory/results.json          Frozen/oracle/retrieval/weight comparison
  memory/memory.sqlite          Sources used for retrieval
  memory/qa-replay16.safetensors
  acquisition.log, completions.log, retention.log, memory.log
  pipeline.json                 Commands, stage status, report hashes, environment
  summary.json, summary.md      Automatically recounted results
```

Each stage runs in a fresh Python process. Stage wall time includes model
loading and evaluation; raw reports also retain their narrower training/query
timings. Those are different measurements and should not be compared as though
they had identical boundaries.

New input references are relative to their containing report, so the complete
bundle can be moved together under a project's `runs/` directory. Original
project-relative paths in diagnostic fields remain historical provenance.

## Rebuild only the summary

This command uses the Python standard library and does not load a model:

```powershell
python -m qwenbrain.reproduce summarize --run-dir runs/my-reproduction
```

It requires a completed pipeline, verifies the stage/input checksums and their
connections, and recounts the saved generated text using the original scoring
normalization. It refuses incomplete, altered or mixed-run inputs. It regenerates
`summary.json` and `summary.md`; raw results are not rewritten. Stage logs and
model files are not needed for recounting, but the bundle's input configs and
referenced acquisition checkpoint are checked.

## Run a stage explicitly

For debugging or repeating a particular stage, use separate output directories:

```powershell
.\scripts\compare-learning.bat --output runs/acquisition-example
.\.venv\Scripts\python.exe -m qwenbrain.probe_acquisition_completions --acquisition-report runs/acquisition-example/results.json
.\scripts\compare-retention.bat --acquisition-report runs/acquisition-example/results.json --output runs/retention-example
.\scripts\compare-memory.bat --retention-report runs/retention-example/results.json --output runs/memory-example
```

`compare-learning.bat` now runs acquisition only; completion probes receive the
exact acquisition report separately. `compare-retention.bat` and
`compare-memory.bat` require the shown report argument. A historical retention
report can still be passed explicitly when its recorded checkpoint exists and
matches its hash. Old acquisition reports without checkpoint hashes must be
regenerated before being used by the current retention runner.

On failure, inspect `pipeline.json` and the failed stage's log. A partial run is
marked failed and does not receive a completed summary. Fix the cause and use a
new output directory; this version does not automatically resume partial runs.

## Verification boundary

CPU tests cover missing/partial artifacts, checksum mismatches, relocation,
output preservation, QA64 prerequisites, mixed-run rejection, and independent
score recounting. GPU execution is verified separately on the local NVIDIA
machine. The stopping, temporal-memory and grounded-memory experiments remain
their own entry points and are not stages of this four-arm pipeline.

The measured reproduction procedure and outcomes are recorded in
[the verification report](experiments/reproduction-results.md).
