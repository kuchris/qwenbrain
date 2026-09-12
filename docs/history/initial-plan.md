# QwenBrain: per-message adaptation experiment

> Historical plan for the original per-message learner. For the current project scope and results, see [the project overview](../../README.md).

Status: persistent local GPU chat and the sequential automatic adaptation loop
are implemented. See [milestone 1 evidence](milestone-1.md) and
[milestone 3 evidence](milestone-3.md).
The separate [one-message LoRA check](milestone-2.md) also passes: the
adapter changes, base weights remain bit-identical, and disk reload reproduces
the updated model. The fresh-context probes do not establish project-name recall.
The user requires **every submitted chat message to
automatically trigger an update**, replacing the earlier correction-only proposal.
The current chat reports learning ON, candidate outcomes, and accepted versions.
The completed from-scratch MHA/MLA work remains in the sibling llmtrain project.

Storage requirement: all model weights/tokenizers/configs, adapters, checkpoints,
conversation and training data, and model-related caches stay under
`C:\Users\kyost\Desktop\git\qwenbrain`. Use `project-env.bat` before Python
starts; all future launchers and save paths must follow the layout in README.md.
Do not depend on weights or data in another project's folder or the user's
default Hugging Face cache. Large/private runtime artifacts stay ignored by Git.

## Requirement and proposed objective

Confirmed requirement: ordinary conversation and questions trigger learning,
without a correction button or per-message approval. Each accepted non-empty
user message queues one update attempt; interface commands and blank-line exit
are not chat messages. Updates are processed in message order.

Proposed first objective: causal next-token cross-entropy on the **new user
message's content tokens**, conditioned on a bounded preceding conversation and
the correct user-role prefix. Previous turns and template/control tokens are
context, with their training labels masked. Do not relabel the user's text as an
ideal assistant answer. The role prefix provides context even for a one-token
message. Preserve at least one scored content token when applying length limits.

This makes observations themselves the training targets. A question such as
"What is the capital of France?" supplies its wording, not a correct answer.
The model's generated reply is not automatically treated as verified training
data. Improvement in factual accuracy, instruction following, or memory is a
hypothesis to test, not a consequence of performing an optimizer step.

Loss masking must be explicit: a conventional assistant-only SFT mask would omit
the user tokens that this experiment intends to learn from. See the
[TRL loss-mask documentation](https://huggingface.co/docs/trl/sft_trainer).

## Implemented first execution loop

1. Keep the text-only Qwen3.5-0.8B base weights frozen in VRAM. Use one base
   instance with separate serving and candidate LoRA adapters.
2. Generate a reply using the accepted adapter, then copy it into the candidate and perform
   one small optimizer step on the message objective. Initially use a fresh
   optimizer per attempt so a rejected attempt leaves no optimizer history.
3. Record scored tokens, loss, finite gradients, parameter delta, elapsed time,
   and peak allocated VRAM. An attempted update is not proof of changed weights;
   a failed or zero-delta attempt must be reported accurately.
4. Automatically check the candidate for numerical validity and regressions on
   fixed probes. Accept or reject it using recorded thresholds established before
   the live experiment. Every message still triggers a learning attempt even if
   the candidate is rejected. No manual per-message confirmation is required.
5. Return to the input prompt after learning finishes, showing the reply's adapter
   version and the accepted version for the next turn. The synchronous terminal
   loop has no application-level background queue or simultaneous training and generation.
6. Rebuild the bounded conversation's KV/recurrent state after an adapter change;
   do not mix old cached state with new weights. Save accepted adapter versions
   locally so updates survive process restart. One atomically replaced checkpoint
   stores the latest accepted version; rejected candidates leave it unchanged.

The initial experimental settings are recorded in `configs/learning.json`.
Candidates must reduce new-message loss, have finite changed weights, and keep
each fixed reference-text loss within 0.15 nats/token of its original baseline.
That baseline is persisted to prevent the allowed drift from accumulating at
every turn or restart. This provisional threshold is not calibrated as a general
quality guarantee. Wider calibration and useful-retention evaluation remain work
for milestone 4.

## Evidence required

- Each submitted message produces exactly one recorded update attempt.
- Base weights remain unchanged; only candidate adapter parameters can change.
- Serving parameters stay fixed throughout a response and during candidate training.
- Loss includes user content and excludes previous turns, control tokens, and
  generated answers as targets.
- Fresh-context probes assess retention and paraphrase generalization without
  repeating the original message in the evaluation prompt.
- Compare a frozen-model control and an adapting model on the same message stream;
  report unrelated-task regressions, rejected updates, reply latency, and memory.
- After restart, loading the last accepted adapter reproduces its test behavior.

Per-message adaptation can reinforce noise or reduce previously useful behavior.
The experiment should report these outcomes directly; lower loss on an observed
message alone does not establish a better assistant.
