# QwenBrain

A local experiment in a conversational model that learns from each user message.
The question is whether small automatic weight updates produce useful retention
and adaptation while keeping chat responsive.

**Status:** persistent local chat and the first one-message LoRA update check pass.
The update changes adapter weights, keeps original Qwen weights unchanged, and
survives saving/reloading. **Automatic per-message learning is not connected to
chat yet.** `chat.bat` continues to perform inference with frozen base weights.

## Run

Double-click [chat.bat](chat.bat), wait for the model to load, and enter messages.
The model stays in VRAM until the process exits. `/reset` clears the conversation
without reloading the model. A blank line or `/quit` closes the chat.

Setup has already been run on this machine. On a fresh checkout with `uv` available,
run [setup.bat](setup.bat) once to install the pinned environment and download the
pinned model into this folder. Chat itself loads local files only and works offline.

Messages and replies are saved as UTF-8 JSONL files in `data/conversations/`.
They are ignored by Git. Each launch starts a fresh conversation; saved transcripts
are not automatically restored or used for training at this milestone.

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

The initial proposal is one frozen base model with serving and candidate LoRA
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

The first two milestones now pass: local chat and one correctly masked LoRA
update. Next, implement adapter versioning and automatic candidate checks before
connecting update attempts to every chat message.

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
