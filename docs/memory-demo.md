# Persistent memory demo

This CPU-only application turns explicit project claims into persistent, versioned
records. It answers two question templates directly from the applicable source.
It never loads a model, updates weights, or describes a lookup as model recall.

## Run the walkthrough

From the project root with Python 3.11, without installing packages:

```powershell
python -m qwenbrain.memory_demo walkthrough
```

The Windows launcher uses this project's existing `.venv` interpreter:

```powershell
.\memory-demo.bat walkthrough
```

The walkthrough creates a new `runs/memory-demo-*/` folder containing
`memory.sqlite` and `transcript.json`. It runs every command in a new process,
checks the expected value/source ID or abstention reason, and checks database
hashes around each query. A failed scenario returns a nonzero exit code and
retains the transcript. `walkthrough --output PATH` accepts a new directory only;
it refuses to replace existing artifacts.

The 13 scenario steps demonstrate:

1. Write Cobalt, exit the writer, and read it in a new process.
2. Record Vesper as the new February name; January still returns Cobalt.
3. Receive a late correction to the January name: Laurel.
4. Keep Vesper as the current name while historical queries return Laurel.
5. Apply an earlier observation cutoff to recover the previously known Cobalt.
6. Return UNKNOWN for a missing person, unsupported attribute, date before the
   first effective record, or future query.

These are deterministic scenario assertions, not independent benchmark examples
or a neural accuracy result. The [saved validation transcript](../reports/memory-demo-validation.json)
records one completed run.

## Enter your own commands

Double-click [memory-demo.bat](../memory-demo.bat), or run:

```powershell
python -m qwenbrain.memory_demo
```

At the `memory>` prompt, enter:

```text
remember "Aster Rook" project_name Cobalt --valid-from 2026-01-01
ask "What is Aster Rook's project called?"
remember "Aster Rook" project_name Vesper --valid-from 2026-02-01
ask "What was Aster Rook's project called on 2026-01-15?"
quit
```

Reopen the application and ask the current-name question again. It returns Vesper
with the second source ID. Each successful `remember` commits immediately; each
command closes its database connection. `help` lists commands, `remember --help`
shows write options, and `quit`/EOF exits. Invalid input leaves the shell usable.

Default interactive storage is `data/memory-demo.sqlite` under the project,
ignored by Git. The walkthrough uses its own fresh database. For a different
database, put `--db PATH` before the subcommand:

```powershell
python -m qwenbrain.memory_demo --db data/example.sqlite remember "Aster Rook" project_name Cobalt --valid-from 2026-01-01
python -m qwenbrain.memory_demo --db data/example.sqlite ask "What is Aster Rook's project called?"
```

Relative custom database paths are resolved from your working directory. The
Windows launcher switches to the project root first. The default path always
uses the project root, regardless of the invocation directory.

## Dates and provenance

`--valid-from` is the date a claim takes effect. `--recorded-at` is the date it
was observed; it defaults to the machine's local calendar date. The optional
override supports controlled replay, so this field is caller-supplied metadata,
not an authenticated audit timestamp. Dates use `YYYY-MM-DD`, with day resolution.

`ask --known-at DATE` limits the observations available to the answer, defaulting
to today. A historical date selects the effective state at that date. Selection
uses the latest applicable effective date, then the latest observation date,
then source ID to break same-day ties. Later historical corrections therefore
do not overwrite a newer current state. Future questions return UNKNOWN.

Successful answers contain `mode=source_value`, `source_id`, `valid_from`,
`recorded_at`, `known_at`, `as_of`, and `source_input`. Interactive writes retain
the entered command; one-shot writes retain a canonical command with resolved
dates, excluding the local database path. Original version rows remain present.

An UNKNOWN answer has `mode=abstention`, a null source ID, and a reason:
`unsupported_query`, `invalid_query`, `future_query`, or `no_applicable_source`.
The last reason includes absent identities, unavailable dates and untyped rows;
it does not claim to diagnose which one occurred.

## Supported scope

- Relation: `project_name` only. Values can contain spaces and meaningful
  punctuation, such as `North Star`, `C++` and `C#`. UNKNOWN is reserved.
- Names: Unicode letters/numbers, underscores and spaces; case and repeated
  whitespace are normalized. Name punctuation, aliases and pronouns are outside
  this first interface. Exact identity lookup separates name prefixes.
- Questions: `What is NAME's project called?` and
  `What was NAME's project called on YYYY-MM-DD?`. Other forms abstain, including
  general-knowledge questions. The demo is not a general chat router.
- Encoding: explicit commands with validated fields; no automatic extraction
  from conversational text and no independent verification that a claim is true.

The model-generated and checked outputs in the earlier experiments remain
separate measurements. Directly returning a source value must not be counted as
an improvement in those models' answers.

## Verification

Run `python -m unittest discover -s tests -v`. The suite includes six demo
integration tests: process restarts and corrections, read-only/UNKNOWN behavior,
invalid writes, interactive recovery and Unicode provenance, concurrent ID
allocation, and walkthrough replay/output preservation. Existing retrieval and
checker regressions remain part of the suite. CI runs on Linux and Windows.
