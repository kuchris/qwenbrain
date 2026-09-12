# Running QwenBrain locally

For explicit writes, historical lookup and restart persistence without a model,
start [memory-demo.bat](../memory-demo.bat). See the [memory demo guide](memory-demo.md).

The chat entry point is the original per-message LoRA experiment. Retrieval and
typed-source checking are separate offline experiments; they are not integrated
into chat. Run shell commands from the project root unless stated otherwise.

## Run

Double-click [chat.bat](../chat.bat), wait for the model to load, and enter messages.
The model stays in VRAM until the process exits. Wait for the learning result
after each reply before entering the next message. `/reset` clears the conversation
without reloading the model or erasing learned weights. A blank line or `/quit`
closes the chat. Commands do not trigger training.

Setup has already been run on this machine. On a fresh checkout with `uv` available,
run [setup.bat](../setup.bat) once to install the pinned environment and download the
pinned model into this folder. Chat itself loads local files only and works offline.

Messages and replies are saved as UTF-8 JSONL files in `data/conversations/`.
They are ignored by Git. Each launch starts a fresh conversation; saved transcripts
are not automatically restored or replayed for training. Only the newly submitted
message is a prediction target; older conversation turns provide context.

The initial chat budget is 2,048 total tokens, reserving 256 for a reply. Old
complete turns are removed from the prompt when necessary. A new message that
cannot fit on its own is rejected with an explanation. Thinking is disabled.
These are local prototype limits, not Qwen's maximum advertised context length.

Generation uses a response-only presence penalty to reduce repeated-sentence loops.
After a code or sampling-config update, exit with `/quit` and reopen `chat.bat`;
`/reset` clears context but does not reload code or configuration. See
[the repetition fix](history/repetition-fix.md) for the reproduced failure and checks.

See [the milestone evidence](history/milestone-1.md) for measurements and limits.

Double-click [check-update.bat](../scripts/check-update.bat) to repeat the separate one-message
training check using `configs/update-check.json`. It trains on the fixed example,
writes a new adapter under `adapters/one-message-*/`, and reports its before/after
loss and fresh-context answers. It does not change the adapter used by `chat.bat`.
The first update lowered message loss but did not establish recall of the supplied
project name. See [milestone 2 results](history/milestone-2.md).

Double-click [compare-learning.bat](../scripts/compare-learning.bat) for the separate offline
acquisition experiment: raw statements, varied statements, and grounded Q&A at
1/4/16/64 exposures, followed by confirmation on fresh synthetic facts. It writes
its own adapters and results under `runs/acquisition-*/` and never trains the live
adapter. See [the fixed protocol](experiments/acquisition-protocol.md) and
[the research behind it](research/online-learning-research.md).

The [completed acquisition comparison](experiments/acquisition-results.md) found
43/48 held-out answers correct after 16 Q&A exposures, versus 0/48 for raw
statements. These are eight synthetic facts, two question wordings, and three
seeds. Raw training did learn statement completion; transfer to chat questions
was the gap. The experiment does not change live learning.

After a completed Q&A acquisition comparison, [compare-retention.bat](../scripts/compare-retention.bat)
tests learning eight new facts with and without replay of the old eight. It uses
the latest completed comparison's seed-42 final Q&A adapter and creates separate
outputs under `runs/retention-*/`. See [the retention results](experiments/retention-results.md).

Double-click [compare-stopping.bat](../scripts/compare-stopping.bat) to compare answer-only
Q&A training against supervising the answer plus its end-of-response token. It
uses eight fresh synthetic facts and three seeds, writes separate adapters under
`runs/stopping-*/`, and leaves live learning unchanged. See
[the fixed stopping protocol](experiments/stopping-protocol.md).
The [completed stopping comparison](experiments/stopping-results.md) scored 48/48
with end-token supervision versus 44/48 without it at 16 exposures. Longer
training still caused unrelated answer regressions; live learning is unchanged.

Double-click [compare-memory.bat](../scripts/compare-memory.bat) to compare four methods on
the retention experiment's same 16 facts: frozen base, oracle source evidence,
SQLite retrieval, and Q&A with 50% replay. The weight arm reproduces the earlier
16-step replay setting; retrieval matches full person names and returns their
original statements. Every question uses a fresh context, and evaluation never
writes memories or trains. Outputs go to `runs/memory-*/`. See the
[fixed protocol](experiments/memory-protocol.md) and [results](experiments/memory-results.md).
This is a diagnostic on reused facts, not a semantic-search or general long-term
memory benchmark. CPU retrieval checks run with
`.venv\Scripts\python.exe -m unittest discover -s tests -v`.

[compare-temporal-memory.bat](../scripts/compare-temporal-memory.bat) tests current names,
historical names, late corrections and UNKNOWN responses on eight fresh people.
The SQLite store keeps both effective and observation dates; the model stays
frozen. The default uses raw source sentences. Pass `--evidence-format explicit`
to repeat the follow-up with labeled lookup results and explicit date semantics.
See the [protocol](experiments/temporal-memory-protocol.md) and
[results](experiments/temporal-memory-results.md). These are offline experiments with
structured dates and exact names, not automatic extraction in live chat.

[compare-grounded-memory.bat](../scripts/compare-grounded-memory.bat) evaluates a conservative
typed-source answer checker on three fixed fresh-data worlds. It compares each
raw model answer with the checked output: an applicable project value must match
in full, otherwise the application returns UNKNOWN. Existing model abstentions
are retained, and simple arithmetic/capital questions are explicitly outside the
checker's scope. See the [protocol](experiments/grounded-memory-protocol.md) and
[results](experiments/grounded-memory-results.md). Use `--seed 73` to repeat one world.

## Automatic learning

Each reply uses the last accepted adapter. After the reply, the app copies that
adapter into a candidate, trains one step on your new message's content, checks
the candidate, and either saves/promotes it or keeps the previous adapter.
Generation and learning run sequentially on one shared base. Both small LoRA
adapters stay in GPU memory. Each generation builds fresh KV/recurrent state from
the bounded conversation, so no cached state survives an adapter change.

Settings are in [configs/learning.json](../configs/learning.json): rank 8, alpha 16,
learning rate 0.0001, gradient clipping 1.0, and a fresh AdamW optimizer per message.
A candidate needs finite changed weights and lower new-message loss. Its loss on
each of three fixed reference texts must stay within **0.15 nats/token** of the
original baseline, which persists across restarts. This provisional threshold is
a small regression check; passing it does not establish general quality or recall.
Questions train their wording, not a verified answer. Generated replies are not
training targets. Rejected updates still count as attempts and are logged.

Training recomputes intermediate activations during backward to reduce GPU memory.
It projects vocabulary logits only for the positions predicting the new content,
while the full bounded history still conditions the transformer. The explicitly
aligned targets are checked against the original full-logit masked loss.

The single live adapter (`adapters/live.safetensors`, created locally) contains the latest accepted
weights, version number, model/config identity, and original guard baselines.
It is atomically replaced after acceptance. Rejected candidates and save failures
leave the previous serving weights and checkpoint intact. This is rollback of a
failed candidate, not a historical archive of all accepted versions. A Windows
file lock prevents two learning chats from writing the same checkpoint. Changed
model/learning configuration is rejected when loading an incompatible checkpoint.

The GPU checks use separate checkpoints under `runs/`; synthetic test facts never
enter your live adapter. Run [check-learning.bat](../scripts/check-learning.bat) to repeat them.
See [milestone 3 evidence](history/milestone-3.md) for results and limitations.

## Project-local storage

All model and experiment data belongs under this folder:

```text
qwenbrain/
├── models/hub/         Downloaded model weights, tokenizer, and model config
├── adapters/           Saved LoRA adapter versions
├── checkpoints/        Resumable training checkpoints
├── data/conversations/ Conversation records used by the experiment
├── data/cache/         Prepared dataset cache
├── runs/               Experiment outputs
├── logs/               Runtime logs
└── .cache/             Hugging Face metadata/Xet, Torch cache, temporary files
```

[project-env.bat](../project-env.bat) configures these cache locations before Python
starts, overriding inherited Hugging Face cache paths in that process. Root-level launchers call the environment script as follows:

```bat
call "%~dp0project-env.bat"
if errorlevel 1 exit /b 1
```

Launchers in `scripts/` instead call `"%~dp0..\project-env.bat"` and switch
to the parent project directory first. The environment script resolves paths
relative to its own location, so launching from another working
directory still uses this project. It does not change Windows user/system
environment settings. The Python chat/download entry points also call
`qwenbrain.paths.configure_storage()` before importing the model libraries,
so direct module invocation uses the same project-local paths. New Python entry
points must do the same; merely opening this directory does not configure caches.
Model loaders must use this configured Hub cache or an explicit path inside
`models/`, and save operations must target this project's data directories.

The paths use Hugging Face's documented
[cache environment variables](https://huggingface.co/docs/huggingface_hub/main/package_reference/environment_variables).
Large weights, adapters, conversation data, checkpoints, logs, and caches are
ignored by Git. The pinned model files (1.65 GiB) have been downloaded here;
their manifest is in `models/download-manifest.json`, with a tracked copy in
`reports/download-manifest.json`. No model data from another project is required.
