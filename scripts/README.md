# Experiment entry points

Run commands from the project root in PowerShell. The launchers also resolve the
project directory when double-clicked or called from another working directory.
`setup.bat`, `chat.bat`, `memory-demo.bat`, and `project-env.bat` remain at the root.

For the CPU-only interactive demo, use `.\memory-demo.bat`; use
`.\memory-demo.bat walkthrough` for the checked process-restart scenario.
See [memory demo usage](../docs/memory-demo.md). It also runs with a standalone
Python 3.11 interpreter and does not require the GPU setup below.

All commands below use the project `.venv` and local caches. Experiment/check
adapters are separate from the live chat adapter. GPU commands run sequentially;
do not launch several training experiments at once.

| Command | Purpose | Prerequisite beyond setup |
| --- | --- | --- |
| `.\scripts\reproduce-memory.bat --output runs/my-reproduction` | Full acquisition/completion/retention/four-arm pipeline | None; creates every training checkpoint in a new output directory |
| `.\scripts\compare-grounded-memory.bat --seed 73` | One fresh world, paired raw/checked answers | None; omit `--seed` for all three worlds |
| `.\scripts\compare-temporal-memory.bat` | Frozen/oracle/retrieval with raw sources | None |
| `.\scripts\compare-temporal-memory.bat --evidence-format explicit` | Explicit date and lookup-result presentation | None |
| `.\scripts\compare-stopping.bat` | Answer-only vs answer-plus-end-token supervision | None |
| `.\scripts\compare-learning.bat --output runs/acquisition-example` | Raw, varied and QA acquisition | None; creates training checkpoints |
| `.\scripts\compare-retention.bat --acquisition-report runs/acquisition-example/results.json` | New facts with/without replay | Explicit completed acquisition report with QA selected and verified QA64 checkpoint |
| `.\scripts\compare-memory.bat --retention-report runs/retention-example/results.json` | Frozen/oracle/SQLite/QA-replay comparison | Explicit completed retention report and its verified acquisition checkpoint |
| `.\scripts\check-update.bat` | One-message update, masking, reload and recall checks | None; writes `reports/update-checks.json` |
| `.\scripts\check-learning.bat` | Automatic-update acceptance, rejection and persistence checks | None; writes `reports/online-checks.json` |

The fresh-world experiments are the easiest GPU entry points for a new reader.
Experiment runners write new directories under `runs/`; they do not automatically
replace the consolidated result JSON or narrative reports in `reports/` and
`docs/experiments/`. The four-stage pipeline automatically generates its own
summary; `python -m qwenbrain.reproduce summarize --run-dir runs/my-reproduction`
recounts it without loading a model.

The full pipeline removes the need for a historical training checkpoint by
rebuilding acquisition first. Individual stages use explicit input reports;
they do not search for the latest run. See the [reproduction guide](../docs/reproduction.md)
for commands, output layout and artifact checks. Historical published reports
remain unchanged.

For automation without the batch files' final `pause`, invoke the Python module:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m qwenbrain.experiment_grounded_memory --seed 73
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

See the [experiment index](../reports/README.md) for protocols and saved results,
and [usage](../docs/usage.md) for chat behavior and storage settings.
