# QwenBrain

A local experiment in a conversational model that learns from each user message.
The question is whether small automatic weight updates produce useful retention
and adaptation while keeping chat responsive.

**Status:** `chat.bat` now replies and then automatically attempts one LoRA update
for each submitted message. The base stays frozen in VRAM. A separate candidate
adapter is checked before replacing the serving adapter. Accepted updates are
saved locally and loaded at the next launch. Useful recall remains an experiment,
not a consequence of an accepted update.

## Run

Double-click [chat.bat](chat.bat), wait for the model to load, and enter messages.
The model stays in VRAM until the process exits. Wait for the learning result
after each reply before entering the next message. `/reset` clears the conversation
without reloading the model or erasing learned weights. A blank line or `/quit`
closes the chat. Commands do not trigger training.

Setup has already been run on this machine. On a fresh checkout with `uv` available,
run [setup.bat](setup.bat) once to install the pinned environment and download the
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
[the repetition fix](reports/repetition-fix.md) for the reproduced failure and checks.

See [the milestone evidence](reports/milestone-1.md) for measurements and limits.

Double-click [check-update.bat](check-update.bat) to repeat the separate one-message
training check using `configs/update-check.json`. It trains on the fixed example,
writes a new adapter under `adapters/one-message-*/`, and reports its before/after
loss and fresh-context answers. It does not change the adapter used by `chat.bat`.
The first update lowered message loss but did not establish recall of the supplied
project name. See [milestone 2 results](reports/milestone-2.md).

Double-click [compare-learning.bat](compare-learning.bat) for the separate offline
acquisition experiment: raw statements, varied statements, and grounded Q&A at
1/4/16/64 exposures, followed by confirmation on fresh synthetic facts. It writes
its own adapters and results under `runs/acquisition-*/` and never trains the live
adapter. See [the fixed protocol](reports/acquisition-protocol.md) and
[the research behind it](reports/online-learning-research.md).

The [completed acquisition comparison](reports/acquisition-results.md) found
43/48 held-out answers correct after 16 Q&A exposures, versus 0/48 for raw
statements. These are eight synthetic facts, two question wordings, and three
seeds. Raw training did learn statement completion; transfer to chat questions
was the gap. The experiment does not change live learning.

After a completed Q&A acquisition comparison, [compare-retention.bat](compare-retention.bat)
tests learning eight new facts with and without replay of the old eight. It uses
the latest completed comparison's seed-42 final Q&A adapter and creates separate
outputs under `runs/retention-*/`. See [the retention results](reports/retention-results.md).

## Automatic learning

Each reply uses the last accepted adapter. After the reply, the app copies that
adapter into a candidate, trains one step on your new message's content, checks
the candidate, and either saves/promotes it or keeps the previous adapter.
Generation and learning run sequentially on one shared base. Both small LoRA
adapters stay in GPU memory. Each generation builds fresh KV/recurrent state from
the bounded conversation, so no cached state survives an adapter change.

Settings are in [configs/learning.json](configs/learning.json): rank 8, alpha 16,
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

The single [live adapter](adapters/live.safetensors) contains the latest accepted
weights, version number, model/config identity, and original guard baselines.
It is atomically replaced after acceptance. Rejected candidates and save failures
leave the previous serving weights and checkpoint intact. This is rollback of a
failed candidate, not a historical archive of all accepted versions. A Windows
file lock prevents two learning chats from writing the same checkpoint. Changed
model/learning configuration is rejected when loading an incompatible checkpoint.

The GPU checks use separate checkpoints under `runs/`; synthetic test facts never
enter your live adapter. Run [check-learning.bat](check-learning.bat) to repeat them.
See [milestone 3 evidence](reports/milestone-3.md) for results and limitations.

## Confirmed direction

- Use Qwen3.5-0.8B as the pretrained starting model.
- Keep the model loaded in VRAM across messages.
- Every submitted chat message automatically triggers a learning update attempt;
  learning does not require an explicit correction.
- Preserve learned updates across restarts.
- Store all model files, adapters, conversation/training data, and model-related
  caches inside this project folder, with no links to another project's data.
- Build in small explained milestones for a hands-on, job-seeking portfolio.
- Work locally on the RTX 5070 Ti. Paid compute needs a separately agreed budget.

The implementation uses one frozen base model with serving and candidate LoRA
adapters, trained by next-token prediction on new user-message content. Training
and generation take turns on the shared model. A failed candidate can be rejected
automatically; an update attempt does not guarantee acceptance or improvement.
See [PLAN.md](PLAN.md) for the objective, protocol, and measurements.

"Growing brain" is an analogy for changing learned behavior. This experiment
does not add neurons, demonstrate consciousness, or assume that every observation
improves the model.

## First implementation milestones

1. Pin a compatible environment and model revision. Load the text-only model in
   BF16, enter several chat messages in one process, and measure VRAM and latency.
2. Attach LoRA and prove one user message produces correctly masked loss,
   gradients, and an adapter weight change while base weights remain unchanged.
3. Add ordered per-message updates, candidate evaluation, adapter versioning,
   cache rebuilding, and checkpoint reload checks.
4. Compare a frozen control and an adapting model on the same message stream.
   Test retention with the original messages removed from the prompt, unrelated
   task regressions, reply latency, memory, and restart persistence.

The automatic update loop implements milestone 3. Milestone 4 expands retention
and regression evaluation beyond the initial three-fact probe.

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

[project-env.bat](project-env.bat) configures these cache locations before Python
starts, overriding inherited Hugging Face cache paths in that process. Future
chat, download, and training launchers must begin with:

```bat
call "%~dp0project-env.bat"
if errorlevel 1 exit /b 1
```

It resolves paths relative to the batch file, so launching from another working
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

## Existing work

- `../llmtrain`: from-scratch MHA/MLA models, training pilots, cache checks, and
  the original discussion that led to this project. Keep its results intact.
- `../TinyReasonRL`: an existing Qwen3.5-0.8B-Base text loader and local LoRA/RL
  experiment. Inspect its compatible components before duplicating setup work;
  its dependencies and results still need verification for this project.

No GitHub remote, publishing, or scheduled background training is configured.
