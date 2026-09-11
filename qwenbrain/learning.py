"""Content-only next-token targets and one bounded LoRA optimizer step."""

import json
import time

from .paths import configure_storage

configure_storage()

import torch
from peft import LoraConfig, get_peft_model
from torch.nn.attention import SDPBackend, sdpa_kernel


def build_message_batch(tokenizer, history, message, max_tokens):
    """Label only new user content, using offsets in the actual chat template.

    Render a unique placeholder to locate the new content without confusing it
    with identical text in history or a substring of a role/control marker.
    HF's causal LM loss shifts these labels internally; do not shift them twice.
    """
    message = message.strip()
    if not message:
        raise ValueError("A training message must contain text.")
    marker = "QWENBRAIN_CONTENT_MARKER"
    while marker in json.dumps(history, ensure_ascii=False) + message:
        marker += "_X"

    def render(content):
        return tokenizer.apply_chat_template(
            history + [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )

    prefix, suffix = render(marker).split(marker)
    text = render(message)
    if text != prefix + message + suffix:
        raise ValueError("Chat template transformed user content; cannot safely locate training targets.")
    start, end = len(prefix), len(prefix) + len(message)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
    offsets = encoded.pop("offset_mapping")[0]
    if encoded.input_ids.shape[1] > max_tokens:
        raise ValueError("Training input exceeds the configured context budget.")
    mask = (offsets[:, 0] >= start) & (offsets[:, 1] <= end) & (offsets[:, 1] > offsets[:, 0])
    # Control tokens are never prediction targets, including if supplied literally.
    mask &= ~torch.isin(encoded.input_ids[0], torch.tensor(tokenizer.all_special_ids))
    mask[0] = False  # There is no previous position to predict token zero.
    if not mask.any():
        raise ValueError("Message contains no scorable content tokens.")
    labels = encoded.input_ids.clone()
    labels[:, ~mask] = -100
    return {**encoded, "labels": labels}, {
        "input_tokens": encoded.input_ids.shape[1], "scored_tokens": mask.sum().item(),
        "scored_text": tokenizer.decode(encoded.input_ids[0, mask], skip_special_tokens=False),
        "scored_positions": mask.nonzero().flatten().tolist(),
    }


def attach_adapter(model, config, adapter_name="default"):
    return get_peft_model(model, LoraConfig(
        r=config["rank"], lora_alpha=config["alpha"], target_modules=config["target_modules"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    ), adapter_name=adapter_name)


def select_scored_logits(batch):
    """Project only predecessor positions needed for a single message's loss.

    Explicit shift_labels overrides HF's usual shift; labels are aligned once.
    The full history still goes through the transformer as conditioning context.
    """
    positions = (batch["labels"][0] != -100).nonzero().flatten()
    return {**batch, "logits_to_keep": positions - 1, "shift_labels": batch["labels"][:, positions]}


def update_once(model, batch, config):
    """One bounded AdamW step; original weights must already be frozen by PEFT."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"], weight_decay=0.0)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**batch, use_cache=False).loss
    loss.backward()
    gradients = {name: p.grad.detach().float().norm().item()
                 for name, p in model.named_parameters() if p.requires_grad and p.grad is not None}
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, config["gradient_clip"], error_if_nonfinite=True)
    if not torch.isfinite(loss) or not gradients or gradient_norm.item() == 0:
        raise RuntimeError("No finite, nonzero learning signal for this message.")
    optimizer.step()
    torch.cuda.synchronize()
    metrics = {"steps": 1, "loss_before": loss.item(), "seconds": time.perf_counter() - started,
               "gradient_norm_before_clip": gradient_norm.item(), "adapter_gradient_norms": gradients,
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    return metrics
