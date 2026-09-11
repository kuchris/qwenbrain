# QwenBrain

A local experiment in a conversational model that learns from each user message.
The question is whether small automatic weight updates produce useful retention
and adaptation while keeping chat responsive.

**Status:** project initialized; design transferred from `llmtrain`. No model has
been loaded and no training or chat service has been implemented in this project.

## Confirmed direction

- Use Qwen3.5-0.8B as the pretrained starting model.
- Keep the model loaded in VRAM across messages.
- Every submitted chat message automatically triggers a learning update attempt;
  learning does not require an explicit correction.
- Preserve learned updates across restarts.
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

The first milestone is model loading and chat only; it should pass before the
automatic learning loop is implemented. Provide a Windows `.bat` launcher when
the chat entry point exists so repeated testing needs no long command.

## Existing work

- `../llmtrain`: from-scratch MHA/MLA models, training pilots, cache checks, and
  the original discussion that led to this project. Keep its results intact.
- `../TinyReasonRL`: an existing Qwen3.5-0.8B-Base text loader and local LoRA/RL
  experiment. Inspect its compatible components before duplicating setup work;
  its dependencies and results still need verification for this project.

No GitHub remote, publishing, or scheduled background training is configured.
