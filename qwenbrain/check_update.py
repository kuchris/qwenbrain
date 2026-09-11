"""Prove one real LoRA update, target masking, frozen base weights, and disk reload."""

import hashlib
import json
import os
import time
import warnings

os.environ["HF_HUB_OFFLINE"] = "1"

from .chat import ChatSession, ROOT
from .learning import attach_adapter, build_message_batch, update_once

import peft
import torch
from peft import PeftModel, get_peft_model_state_dict
from torch.nn.attention import SDPBackend, sdpa_kernel


def parameter_hash(parameters):
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode())
        digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def measure(model, batch):
    with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(**batch, use_cache=False)
    return result.loss.item(), result.logits.float().cpu()


def probes(session, questions):
    results = []
    for question in questions:
        session.reset()  # The training statement is absent from each probe context.
        torch.manual_seed(42)
        answer, _ = session.reply(question)
        results.append({"question": question, "answer": answer})
    session.reset()
    return results


def main():
    config = json.loads((ROOT / "configs/update-check.json").read_text())
    session = ChatSession()
    torch.manual_seed(config["seed"])
    history = list(session.messages)
    # Repeat the new text in earlier turns to ensure only the final instance is labeled.
    repeated_history = history + [{"role": "user", "content": config["message"]},
                                  {"role": "assistant", "content": "Previous answer must not be a target."}]
    masks = []
    for text in [config["message"], "hi", "end", "こんにちは", "QWENBRAIN_CONTENT_MARKER", "line one\nline two"]:
        masked, trace = build_message_batch(session.tokenizer, repeated_history, text, 2048)
        assert trace["scored_text"] == text, trace
        assert trace["scored_tokens"] > 0
        assert masked["labels"][0, -1].item() == -100
        masks.append({"message": text, **trace})
    for text in ["", "<|im_end|>", "too long " * 5000]:
        try:
            build_message_batch(session.tokenizer, history, text, 2048)
        except ValueError:
            pass
        else:
            raise AssertionError("Empty/control-only/overlong message accepted")

    batch_cpu, trace = build_message_batch(session.tokenizer, history, config["message"], 2048)
    batch = {k: v.cuda() for k, v in batch_cpu.items()}
    base_parameters = list(session.model.named_parameters())
    base_hash_before = parameter_hash(base_parameters)
    base_loss, base_logits = measure(session.model, batch)
    baseline_probes = probes(session, config["fresh_context_probes"])
    torch.manual_seed(config["seed"])
    session.model = attach_adapter(session.model, config)
    session.model.eval()
    trainable = {n: p for n, p in session.model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    assert all(p.dtype == torch.float32 for p in trainable.values())
    assert all(not p.requires_grad for _, p in base_parameters)
    before = {n: p.detach().cpu().clone() for n, p in trainable.items()}
    initial_loss, initial_logits = measure(session.model, batch)
    torch.testing.assert_close(initial_logits, base_logits, atol=0, rtol=0)
    # Independently check HF's internal next-token shift and content-only loss.
    positions = torch.tensor(trace["scored_positions"])
    selected = initial_logits[0, positions - 1].double()
    targets = batch_cpu["labels"][0, positions]
    manual_loss = (selected.logsumexp(-1) - selected[torch.arange(len(positions)), targets]).mean()
    # FP64 CPU reference vs FP32 CUDA reduction over a 248k-token vocabulary.
    loss_reference_error = abs(manual_loss.item() - initial_loss)
    assert loss_reference_error < 1e-4, loss_reference_error
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        metrics = update_once(session.model, batch, config)
    after_loss, after_logits = measure(session.model, batch)
    deltas = {n: (p.detach().cpu() - before[n]).abs().max().item() for n, p in trainable.items()}
    assert any(value > 0 for value in deltas.values())
    assert all(torch.isfinite(p).all() for p in trainable.values())
    assert all(value == 0 for n, value in deltas.items() if ".lora_A." in n)
    assert any(value > 0 for n, value in deltas.items() if ".lora_B." in n)
    assert after_loss < initial_loss, (initial_loss, after_loss)
    base_hash_after = parameter_hash(base_parameters)
    assert base_hash_before == base_hash_after
    assert all(p.grad is None for _, p in base_parameters)
    after_probes = probes(session, config["fresh_context_probes"])

    adapter_dir = ROOT / "adapters" / f"one-message-{time.time_ns()}"
    session.model.save_pretrained(adapter_dir, safe_serialization=True)
    saved_adapter = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(session.model).items()}
    # Independently load the original checkpoint and the adapter from disk.
    fresh = ChatSession()
    fresh.model = PeftModel.from_pretrained(fresh.model, adapter_dir, is_trainable=False, local_files_only=True)
    fresh.model.eval()
    loaded_adapter = get_peft_model_state_dict(fresh.model)
    for name, value in saved_adapter.items():
        torch.testing.assert_close(value, loaded_adapter[name].cpu(), atol=0, rtol=0)
    reload_loss, reload_logits = measure(fresh.model, batch)
    torch.testing.assert_close(reload_logits, after_logits, atol=0, rtol=0)
    reload_probes = probes(fresh, config["fresh_context_probes"])
    assert reload_probes == after_probes
    report = {
        "status": "passed", "model_id": session.config["model_id"], "revision": session.config["model_revision"],
        "peft": peft.__version__, "config": config, "mask_checks": masks, "training_targets": trace,
        "base_parameters": sum(p.numel() for _, p in base_parameters),
        "adapter_parameters": sum(p.numel() for p in trainable.values()),
        "adapter_tensors": len(trainable), "changed_adapter_tensors": sum(d > 0 for d in deltas.values()),
        "max_adapter_change": max(deltas.values()),
        "base_sha256_before": base_hash_before, "base_sha256_after": base_hash_after,
        "base_loss": base_loss, "initial_adapter_loss": initial_loss, "after_loss": after_loss,
        "fp64_masked_loss_reference": manual_loss.item(), "loss_reference_error": loss_reference_error,
        "loss_reference_atol": 1e-4,
        "reload_loss": reload_loss, "reload_max_logit_difference": (reload_logits - after_logits).abs().max().item(),
        "update": metrics, "warnings": list(dict.fromkeys(str(w.message) for w in captured)),
        "fresh_context_before": baseline_probes, "fresh_context_after": after_probes,
        "fresh_context_reload": reload_probes, "adapter_directory": adapter_dir.relative_to(ROOT).as_posix(),
        "saved_files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                        for p in sorted(adapter_dir.iterdir()) if p.is_file()],
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [ROOT / "qwenbrain/learning.py", ROOT / "qwenbrain/check_update.py", ROOT / "uv.lock"]},
        "scope": "One-message mechanics check. No live chat updates, candidate acceptance gate, or general retention/quality improvement established.",
    }
    (ROOT / "reports/update-checks.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (adapter_dir / "experiment.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["status", "adapter_parameters", "changed_adapter_tensors", "base_loss",
                                           "after_loss", "reload_max_logit_difference", "fresh_context_before",
                                           "fresh_context_after", "adapter_directory"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
