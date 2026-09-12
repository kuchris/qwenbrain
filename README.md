# QwenBrain

**Local LLM memory experiments: LoRA adaptation, versioned retrieval, and answer verification.**

Does learning from a message make an LLM remember its facts after the conversation
is cleared? This project separates weight updates, source retrieval, and answer
generation, then measures where each succeeds or fails on controlled synthetic data.

Built with Python, PyTorch, Hugging Face Transformers, PEFT/LoRA, SQLite and `uv`.
Measured locally on an RTX 5070 Ti using a pinned Qwen3.5-0.8B model in BF16.

[Source repository](https://github.com/kuchris/qwenbrain) ·
[CPU test runs](https://github.com/kuchris/qwenbrain/actions/workflows/tests.yml)

**Status:** a local research and engineering prototype. The original live chat
attempts per-message LoRA updates; retrieval and verification run in separate
offline experiments. An integrated memory assistant is not implemented.

## Results worth reading

| Question | Measured result | Evidence and boundary |
| --- | --- | --- |
| Does fitting a statement produce useful QA recall? | Raw-statement completion: **24/24**; held-out QA: **0/48**. Supervised QA at 16 exposures: **43/48**. | [Acquisition](docs/experiments/acquisition-results.md). Eight confirmation facts, repeated forms/seeds; completion and QA use different probes. |
| How does explicit lookup compare with the tested weight update? | Oracle and SQLite: **32/32**; QA + replay: **25/32**, with one unrelated control regression. | [Four-arm comparison](docs/experiments/memory-results.md). Sixteen reused facts, two forms; one previously selected LoRA setting. |
| Does correct retrieval guarantee a correct answer? | On 24 fresh synthetic identities, source selection: **192/192**; generated current answers: **144/144**, historical: **33/48**. | [Fresh-data evaluation](docs/experiments/grounded-memory-results.md). Three data seeds and repeated stages; exact full-name lookup, not semantic search. |
| Does a strict source checker solve the remaining errors? | Exact-match scores did not improve. It retained 14 false abstentions and rejected one factually correct verbose answer. | [Paired checker evaluation](docs/experiments/grounded-memory-results.md). A prior unsupported answer was blocked in a separate development regression. |

These pilots support explicit, versioned storage for the tested personal-fact task.
They do not prove that weights cannot store facts, that retrieval always wins,
or that general model quality is preserved. Four unique general-knowledge controls
are too few for that last claim.

## What is implemented

- **Candidate LoRA updates:** frozen base, explicit target masking, regression
  checks, rejection/rollback, atomic checkpoint replacement, and reload checks.
- **Versioned SQLite memory:** original source records, effective and observation
  dates, current/historical lookup, late corrections, and restart persistence.
- **Typed-source checking:** exact project-value checks with source IDs and
  explicit abstention. Typed fields come from fixtures; free-form extraction is
  not implemented.
- **Evaluation and evidence:** fresh context per query, frozen/oracle/retrieval/
  weight baselines, saved raw outputs, checksums, and 14 CPU test methods.

The code has two separate execution paths:

```text
Live adaptation experiment
  user message -> reply -> candidate LoRA update -> checks -> accept or rollback

Offline memory experiments
  synthetic records -> versioned SQLite -> query/date lookup -> frozen Qwen
                                                        -> optional source check
```

## Try it locally

The measured GPU environment is Windows, Python 3.11, and an RTX 5070 Ti.
With `uv` installed, run these PowerShell commands from the project root:

```powershell
.\setup.bat
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\scripts\compare-grounded-memory.bat --seed 73
```

Setup installs locked dependencies and downloads the pinned model. The grounded
experiment starts from checked-in fixtures and does not require an old training
adapter. Results are written to a new `runs/grounded-memory-73-*/` directory.
These steps have been exercised in the existing environment; a clean-machine
reproduction has not yet been verified.

The CPU tests use only the standard library and checked-in fixtures/reports. They
can also run with an existing Python 3.11 interpreter, without downloading weights:

```powershell
python -m unittest discover -s tests -v
```

See [all experiment commands and prerequisites](scripts/README.md) and the
[local usage guide](docs/usage.md). The four-arm weight comparison currently
requires the historical acquisition checkpoint referenced by its report.

To run the original chat experiment, launch [chat.bat](chat.bat). It attempts a
weight update after each message and saves accepted changes locally. `/reset`
clears conversation context; it does not erase the adapter. This is not the
retrieval-memory demo.

## Repository map

| Location | Contents |
| --- | --- |
| [qwenbrain/](qwenbrain/) | Chat, LoRA learning, memory stores, checker, experiment runners |
| [configs/](configs/) | Model/learning settings and fixed synthetic worlds |
| [tests/](tests/) | CPU memory, temporal and grounding regression tests |
| [scripts/](scripts/) | Experiment and GPU-check launchers |
| [docs/experiments/](docs/experiments/) | Protocols, result interpretation and limitations |
| [docs/research/](docs/research/) | Literature review and rationale |
| [docs/history/](docs/history/) | Original plan, implementation milestones and diagnostics |
| [reports/](reports/README.md) | Saved JSON measurements, outputs and checksums |
| `runs/`, `models/`, `adapters/`, `data/`, `.cache/` | Local runtime artifacts; ignored by Git |

Source modules, configs, raw results, model paths and checkpoints retain their
locations so that experiment references and recorded evidence remain usable.
Historical JSON may name launchers at their former root location; current
commands are in `scripts/`. GitHub Actions runs the CPU tests on Python 3.11;
GPU experiments are run separately on the local machine.

## Portfolio next steps

The strongest current contribution is a measured diagnosis: lower training loss,
successful retrieval, and reliable answers are distinct outcomes. The next
deliverables are a small integrated memory demo, reproducible experiment
selection/reporting, and evaluation beyond the current templates.

See the [portfolio roadmap and draft CV wording](docs/portfolio-roadmap.md).
