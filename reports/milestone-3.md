# Milestone 3: automatic chat updates

`chat.bat` now responds and then attempts exactly one candidate update for every
submitted chat message. One frozen Qwen base remains loaded in VRAM, with separate
serving and candidate LoRA adapters. Generation and learning take turns. An
accepted candidate becomes the next response's adapter and is saved locally.

**Mechanics passed; useful fresh-context recall was not demonstrated.** Three
facts produced three accepted updates, but none of the three names/values was
recalled after clearing the conversation. The unrelated arithmetic probe remained
correct. This is a small experiment, not evidence of general capability retention.

## Measured GPU results

Recorded on the local RTX 5070 Ti with the pinned BF16 Qwen3.5-0.8B text model.

| New user message | Loss before | Loss after | Learning, checks, and save | Peak allocated during step |
| --- | ---: | ---: | ---: | ---: |
| My project is called Atlas. | 5.3715 | 4.5336 | 1.65 s | 1,599 MiB |
| My favorite color is teal. | 3.2365 | 2.3271 | 1.71 s | 1,600 MiB |
| My cat is named Mochi. | 2.3893 | 1.5496 | 1.86 s | 1,658 MiB |

These are sequential conversation turns with growing history. The timing excludes
response generation and initial model loading. Allocated memory excludes other
processes, CUDA overhead, and reserved allocator memory. The model's original
752,393,024 parameters remained frozen; each adapter has 5,411,328 parameters.

A separate stress case fit the chat budget with 1,700 new content tokens and
1,722 training input tokens. Its update completed in 2.72 seconds with 6,439 MiB
peak allocated memory and was accepted. Repeated `hello` tokens are a memory
stress case, not a representative quality benchmark or worst-case latency claim.

## Protocol and acceptance

1. Generate with the current serving version and record the response.
2. Record an update attempt. Copy serving weights into the candidate adapter.
3. Train one fresh AdamW step on the new user-content tokens, conditioned on the
   bounded conversation used for the response. Previous turns, the generated
   answer, and role/control tokens are not training targets.
4. Require finite changed weights and reduced new-message loss. Require each of
   three fixed reference-text losses to stay within 0.15 nats/token of the
   original baseline. The baseline is stored with the checkpoint and remains
   fixed across turns and restarts. This is a provisional regression threshold.
5. If checks pass, replace serving weights and atomically save a single checkpoint
   containing weights, version, base revision, learning configuration, and guard
   baselines. Restore the previous serving weights if saving fails.
6. Select the frozen serving adapter, return to evaluation mode, and accept the
   next input. Every generation prefills from conversation text again; no
   previous KV/recurrent state is passed into a new generation.

Rank is 8, alpha 16, learning rate 0.0001, weight decay zero, clipping norm 1.0,
and dropout zero. There is no persistent optimizer state. Training uses BF16
autocast and FP32 LoRA weights, non-reentrant gradient checkpointing, and selected
vocabulary projections to keep memory bounded. Only predictor positions for
unmasked content are projected. Explicit `shift_labels` align those logits with
their next-token targets once. Against full masked evaluation, selected logits
were bit-identical and loss agreed within 1e-4.

The implementation follows the installed PEFT 0.20.0 adapter state APIs and
Transformers 5.16.1 `Qwen3_5ForCausalLM.forward` / `ForCausalLMLoss` behavior.
This single-file checkpoint format is loaded by QwenBrain; it is not a standalone
PEFT adapter directory for `PeftModel.from_pretrained`.

## Verification performed

- Three calls through the real `OnlineLearner.turn` path generated exactly three
  response/update-start/update event sequences and accepted versions 1, 2, and 3.
- During each optimizer step, only candidate LoRA parameters were trainable.
  Serving tensors stayed exactly equal; all original base parameters retained
  the same full SHA-256 checksum across updates and rejection tests.
- Injected guard failure, nonfinite candidate weights, and atomic replacement
  failure kept both serving weights and the existing checkpoint unchanged.
  Failed replacement also removed its temporary file.
- Control-only content produced a recorded rejection without an optimizer step.
  Empty and over-budget inputs were rejected before a chat turn was generated.
- `/reset` cleared conversation context while preserving weights and version.
  A second writer could not acquire the checkpoint's Windows file lock.
- An independently loaded base plus saved adapter reproduced tensors, logits,
  and seeded fresh-context answers exactly. Resumed learning advanced version
  3 to 4 and saved it; the subsequent stress case advanced it to 5.
- `check-learning.bat` completed successfully. `chat.bat` was run with `/reset`
  and `/quit`. The real CLI main was also run with `hi`, `/reset`, `/quit`, using
  a test checkpoint: it printed one accepted update, saved version 1, reset, and
  exited. Test facts never entered `adapters/live.safetensors`.

Raw results: [GPU checks](online-checks.json) and
[CLI result](cli-learning-checks.json). Model files, test adapters, and transcripts
remain under this project. Test checkpoints are under `runs/online-check-*/` or
`runs/cli-check-*/`; the live checkpoint starts fresh on the user's first chat.

## Recall result and limits

Fresh probes did not include the teaching statements or any previous turns.
The frozen baseline and the adapted model both scored 0/3 on the supplied project
name, favorite color, and cat name. After adaptation, the model requested a
project name, suggested blue instead of teal, and said it did not know the cat's
name. The `2 + 2` answer was `4` before and after. Reloading produced the same
adapted responses with the same seed.

An accepted update means this candidate passed the recorded numerical and small
reference-loss checks. It does not establish learned facts, reliable recall,
better answers, or general resistance to forgetting. Fixed reference texts can
miss regressions. A user question supplies wording, not its correct answer.

The live checkpoint stores only the latest accepted version. Rejected candidates
can be rolled back automatically; there is no UI for restoring arbitrary earlier
accepted versions. A crash after the response but before an update completes may
leave an `update_start` without a result; pending attempts are not replayed on
restart. The synchronous terminal has no background training queue.

Next: a broader frozen-versus-adapting evaluation with multiple facts, paraphrased
questions, and unrelated tasks, then compare learning objectives or replay
strategies against this baseline. Lower message loss alone is not success.
