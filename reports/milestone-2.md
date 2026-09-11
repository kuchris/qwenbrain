# Milestone 2: one message changes LoRA weights

The first real one-message update passed. The sentence **"My project is called
Atlas."** changed adapter parameters while every original Qwen parameter kept the
same SHA-256 checksum. Saving and reloading the adapter into an independently
loaded base model reproduced identical logits and seeded probe answers.

**Fresh-context recall did not succeed in this check.** The direct name question
received `No` both before and after training; a paraphrased question still received
a request for more information. Lower loss on the training message establishes
improved prediction of that text, not that the assistant can now answer questions
about it. The unrelated `2 + 2` probe remained `4`; one probe is insufficient to
establish that general capabilities were preserved.

## Configuration and result

| Item | Recorded value |
| --- | --- |
| Base | Qwen3.5-0.8B, revision `2fc06364715b967f1860aea9cf38778875588b17` |
| Original parameters | 752,393,024, frozen BF16 |
| LoRA | Rank 8, alpha 16, all-linear targets, dropout 0, no bias training |
| Adapter parameters | 5,411,328, FP32 |
| Training targets | Six tokens spelling `My project is called Atlas.` |
| Optimizer | One fresh AdamW step, LR 0.0001, weight decay 0, gradient clipping 1.0 |
| Loss before / after | 5.371546 / 4.498805 |
| Changed tensors | 186 of 372 adapter tensors |
| Largest parameter change | Approximately 0.0001 |
| Frozen-base checksum comparison | Exactly equal before and after |
| Reload maximum logit difference | 0.0 |
| Measured update step | 0.657 seconds |
| Peak allocated memory during update | 1,844.74 MiB |

The time covers forward/backward, clipping, and the optimizer step, with CUDA
synchronization. It excludes loading, checksum scans, evaluation, saving, and
reload checks. Peak allocated GPU memory excludes other processes, CUDA overhead,
and allocator-reserved memory. A second base instance is loaded only for the
independent reload verification, after the update measurement; this is not the
proposed live serving architecture.

PEFT initializes LoRA's A matrices randomly and B matrices to zero. Consequently
the initial adapter is a no-op, and only B changes in this first step with zero
weight decay. A's first-step zero gradients are expected. The initial adapter's
logits matched the original model exactly. See the
[PEFT initialization reference](https://huggingface.co/docs/peft/v0.20.0/package_reference/lora).

## What the masking and integrity checks establish

- The actual pinned chat template locates the final user-content span using a
  unique placeholder; repeated text in prior messages is not confused with it.
- Only new content-token labels are retained. Older turns, assistant answers,
  role markers, and other control tokens have label `-100`.
- Labels are not manually shifted before calling the causal LM; its internal
  loss performs the next-token shift. An independent FP64 log-sum-exp calculation
  over the selected predecessor logits agrees within `7.36e-7` (limit `1e-4`).
- English, Japanese, a one-token message, marker-like text, and multiline content
  retain their expected targets. Empty, control-only, and over-budget inputs
  are rejected. This check does not silently truncate messages.
- All trainable parameters are LoRA parameters. Base parameters remain frozen,
  receive no gradients, and have the same full byte-level checksum after training.
- Adapter parameters are finite and changed; the training message's loss falls.
- The saved adapter tensors reload exactly, as do resulting logits and fixed-seed
  answers in conversations that do not contain the training statement.

The model uses BF16 autocast for this training forward pass and FP32 adapter
parameters. The update creates its own optimizer, uses no inference cache, and
returns the model to evaluation mode. Presence penalties apply to generated
answers through the existing chat path; they are not part of the training loss.

## Files and reproduction

- [Configuration](../configs/update-check.json)
- [Full checks, gradients, hashes, and probe answers](update-checks.json)
- [Learning code](../qwenbrain/learning.py)
- [Verification code](../qwenbrain/check_update.py)

Double-click `check-update.bat`, or run from the repository root:

```powershell
.venv\Scripts\python.exe -X utf8 -m qwenbrain.check_update
```

PEFT 0.20.0 and its dependencies are included in the updated lockfile. Run
`setup.bat` after updating a separate checkout to install them.
The check uses Hub offline mode. Each successful run saves a new local
`adapters/one-message-*/` directory, including about 20.69 MiB of adapter weights,
adapter configuration, and a copy of its experiment report. The exact directory
and content hashes are recorded in `reports/update-checks.json`.

This adapter is diagnostic and is not automatically promoted into `chat.bat`.
Live per-message updates, serving/candidate isolation, acceptance criteria,
rollback, and meaningful retention evaluation are the next milestone.
