# Experiment entry points

Run commands from the project root in PowerShell. The launchers also resolve the
project directory when double-clicked or called from another working directory.
`setup.bat`, `chat.bat`, and `project-env.bat` remain at the root.

All commands below use the project `.venv` and local caches. Experiment/check
adapters are separate from the live chat adapter. GPU commands run sequentially;
do not launch several training experiments at once.

| Command | Purpose | Prerequisite beyond setup |
| --- | --- | --- |
| `.\scripts\compare-grounded-memory.bat --seed 73` | One fresh world, paired raw/checked answers | None; omit `--seed` for all three worlds |
| `.\scripts\compare-temporal-memory.bat` | Frozen/oracle/retrieval with raw sources | None |
| `.\scripts\compare-temporal-memory.bat --evidence-format explicit` | Explicit date and lookup-result presentation | None |
| `.\scripts\compare-stopping.bat` | Answer-only vs answer-plus-end-token supervision | None |
| `.\scripts\compare-learning.bat` | Raw, varied and QA acquisition; completion probes | None; creates training checkpoints |
| `.\scripts\compare-retention.bat` | New facts with/without replay | A completed acquisition run with QA selected; uses the latest completed run |
| `.\scripts\compare-memory.bat` | Frozen/oracle/SQLite/QA-replay comparison | The exact acquisition checkpoint and SHA-256 referenced in `reports/retention-results.json` |
| `.\scripts\check-update.bat` | One-message update, masking, reload and recall checks | None; writes `reports/update-checks.json` |
| `.\scripts\check-learning.bat` | Automatic-update acceptance, rejection and persistence checks | None; writes `reports/online-checks.json` |

The fresh-world experiments are the easiest GPU entry points for a new reader.
Experiment runners write new directories under `runs/`; they do not automatically
replace the consolidated result JSON or narrative reports in `reports/` and
`docs/experiments/`. A reproducible aggregation command remains planned work.

The four-arm comparison has a historical-artifact dependency. Re-running
acquisition creates a new path, which the saved retention report will not select
automatically. Do not edit a published report or its checksum to make a new run
appear identical. Explicit input selection and new report generation are listed
in the [portfolio roadmap](../docs/portfolio-roadmap.md).

For automation without the batch files' final `pause`, invoke the Python module:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m qwenbrain.experiment_grounded_memory --seed 73
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

See the [experiment index](../reports/README.md) for protocols and saved results,
and [usage](../docs/usage.md) for chat behavior and storage settings.
