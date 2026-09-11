"""Check actual GPU chat, local storage, bounded context, and inference-only state."""

import json
import os
from pathlib import Path

# Must take effect before importing the model libraries.
os.environ["HF_HUB_OFFLINE"] = "1"

from .chat import ChatSession, ROOT

import torch
from huggingface_hub import constants


def main():
    session = ChatSession()
    model_id = id(session.model)
    versions = [p._version for p in session.model.parameters()]
    assert not session.model.training
    assert all(not p.requires_grad and p.device.type == "cuda" for p in session.model.parameters())
    assert all(p.dtype == torch.bfloat16 for p in session.model.parameters())
    for directory in (constants.HF_HOME, constants.HF_HUB_CACHE, constants.HF_XET_CACHE):
        assert Path(directory).resolve().is_relative_to(ROOT)

    turns = []
    for message in ["My project is called Atlas. Reply with OK.",
                    "What is my project called? Reply with only its name."]:
        answer, metrics = session.reply(message)
        assert answer.strip() and metrics["generated_tokens"] > 0
        assert id(session.model) == model_id
        assert metrics["prompt_tokens"] + session.config["max_new_tokens"] <= session.config["context_tokens"]
        turns.append({"user": message, "assistant": answer, "metrics": metrics})
    assert [item["role"] for item in session.messages] == ["system", "user", "assistant", "user", "assistant"]
    # Checking retained input context is distinct from claiming learned weights.
    assert "Atlas" in session.messages[1]["content"]
    session.reset()
    assert len(session.messages) == 1 and id(session.model) == model_id
    messages, inputs, omitted = session.prepare("Hello")
    assert "Atlas" not in session.tokenizer.decode(inputs.input_ids[0]) and omitted == 0

    # Drop old complete turns rather than silently cutting a new message.
    session.messages += [{"role": "user", "content": "old context " * 2000},
                         {"role": "assistant", "content": "Old answer."}]
    messages, inputs, omitted = session.prepare("こんにちは")
    assert omitted == 1 and len(messages) == 2 and messages[-1]["content"] == "こんにちは"
    saved = list(session.messages)
    try:
        session.prepare("oversized " * 5000)
    except ValueError:
        pass
    else:
        raise AssertionError("Overlong current message was accepted")
    assert session.messages == saved
    assert versions == [p._version for p in session.model.parameters()]
    assert all(p.grad is None for p in session.model.parameters())

    report = {"status": "passed", "metadata": session.metadata, "turns": turns,
              "checks": ["offline local model loading", "all parameters frozen/BF16/on CUDA",
                         "same model object reused across turns", "conversation history retained",
                         "reset removes old context without model reload", "complete old turns omitted at budget",
                         "overlong new message rejected without changing history", "parameter versions unchanged and no gradients"],
              "scope": "Inference-only milestone. Recall uses prompt context; no LoRA or per-message training yet."}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/chat-checks.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "parameters": session.metadata["parameters"],
                      "load_seconds": session.metadata["load_seconds"],
                      "allocated_after_load_mib": session.metadata["allocated_after_load_mib"],
                      "turns": turns}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
