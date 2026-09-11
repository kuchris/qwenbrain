"""Real-GPU checks of chat updates, isolation, rejection, persistence, and recall."""

import hashlib
import gc
import json
import os
import time
from unittest.mock import patch

os.environ["HF_HUB_OFFLINE"] = "1"

from .chat import ChatSession, ROOT
from .check_update import measure, parameter_hash, probes
from .learning import build_message_batch, select_scored_logits, update_once
from .online import OnlineLearner

import torch


def equal_state(left, right):
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)


def main():
    directory = ROOT / "runs" / f"online-check-{time.time_ns()}"
    checkpoint = directory / "live.safetensors"
    session = ChatSession()
    base_parameters = list(session.model.named_parameters())
    base_before = parameter_hash(base_parameters)
    base_ids = [id(p) for _, p in base_parameters]
    questions = ["What is my project called? Reply with only its name.",
                 "What is my favorite color?", "What is my cat's name?", "What is 2 + 2? Reply with only the number."]
    baseline = probes(session, questions)
    learner = OnlineLearner(session, checkpoint)
    masked, trace = build_message_batch(session.tokenizer, list(session.messages), "My project is called Atlas.", 2048)
    masked = {k: v.cuda() for k, v in masked.items()}
    full_loss, full_logits = measure(session.model, masked)
    selected_loss, selected_logits = measure(session.model, select_scored_logits(masked))
    positions = torch.tensor(trace["scored_positions"]) - 1
    torch.testing.assert_close(selected_logits, full_logits[:, positions], atol=0, rtol=0)
    assert abs(selected_loss - full_loss) < 1e-4
    del full_logits, selected_logits
    try:
        OnlineLearner(session, checkpoint)
    except RuntimeError as error:
        assert "another learning chat" in str(error)
    else:
        raise AssertionError("Two writers acquired the same checkpoint")
    events = []
    isolation_checks = []

    def isolated_update(model, batch, config):
        serving_before = learner.state()
        assert model.active_adapter == "candidate"
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        assert trainable and all(".candidate." in name and "lora_" in name for name in trainable)
        assert all(not p.requires_grad for _, p in base_parameters)
        result = update_once(model, batch, config)
        equal_state(serving_before, learner.state())
        isolation_checks.append(True)
        return result

    turns = []
    for message in ["My project is called Atlas.", "My favorite color is teal.", "My cat is named Mochi."]:
        with patch("qwenbrain.online.update_once", side_effect=isolated_update):
            answer, metrics, learning = learner.turn(message, record=events.append)
        turns.append({"message": message, "answer": answer, "generation": metrics, "learning": learning})
        assert learner.model.active_adapter == "serving"
        assert not any(p.requires_grad for p in learner.model.parameters())
        assert all(p.grad is None for p in learner.model.parameters())
        assert metrics["adapter_version"] == learning["version_before"]
    assert [e["event"] for e in events] == ["turn", "update_start", "update"] * 3
    assert len(isolation_checks) == 3
    assert learner.version > 0, turns
    assert learner.version == sum(t["learning"]["weight_updates"] for t in turns)
    preserved = learner.state()
    version = learner.version
    disk_before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    failures = []

    def check_rejected(result, reason):
        assert result["status"] == "rejected" and reason in result["reason"], result
        assert result["weight_updates"] == 0 and learner.version == version
        assert learner.model.active_adapter == "serving" and not learner.model.training
        assert not any(p.requires_grad for p in learner.model.parameters())
        equal_state(preserved, learner.state())
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == disk_before
        failures.append(result)

    session.reset()
    equal_state(preserved, learner.state())
    assert learner.version == version and len(session.messages) == 1
    message = "My project is called Atlas."
    history = list(session.messages)
    real_loss = learner.loss

    def failed_guard(batch):
        if any(batch is guard for guard in learner.guards):
            return 1000.0
        return real_loss(batch)

    with patch.object(learner, "loss", side_effect=failed_guard):
        check_rejected(learner.learn(history, message), "guard loss threshold")

    def corrupt_candidate(model, batch, config):
        result = update_once(model, batch, config)
        with torch.no_grad():
            next(p for p in model.parameters() if p.requires_grad).fill_(float("nan"))
        return result

    with patch("qwenbrain.online.update_once", side_effect=corrupt_candidate):
        check_rejected(learner.learn(history, message), "nonfinite")
    with patch("qwenbrain.online.os.replace", side_effect=OSError("injected disk failure")):
        check_rejected(learner.learn(history, message), "injected disk failure")
    assert not checkpoint.with_suffix(".tmp").exists()
    check_rejected(learner.learn(history, "<|im_end|>"), "no scorable")
    # Invalid input is rejected before generation; commands are handled by the CLI.
    for text in ["", "too long " * 5000]:
        try:
            learner.turn(text, record=events.append)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input became a chat turn")
    assert len(events) == 9
    after = probes(session, questions)  # Evaluation never calls learner.turn().
    batch, _ = build_message_batch(session.tokenizer, history, message, 2048)
    batch = {k: v.cuda() for k, v in batch.items()}
    _, logits = measure(session.model, batch)
    assert base_ids == [id(p) for _, p in base_parameters]
    base_after = parameter_hash(base_parameters)
    assert base_before == base_after
    learner.close()

    # New loader, original disk base, saved adapter; no original conversation supplied.
    fresh = ChatSession()
    reloaded = OnlineLearner(fresh, checkpoint)
    assert reloaded.version == version
    equal_state(preserved, reloaded.state())
    _, reload_logits = measure(fresh.model, batch)
    torch.testing.assert_close(logits, reload_logits, atol=0, rtol=0)
    reload_probes = probes(fresh, questions)
    assert after == reload_probes
    # Resumed updates continue version numbering and persist after a restart.
    resumed = reloaded.learn(list(fresh.messages), "My project is called Atlas.")
    assert resumed["status"] == "accepted" and reloaded.version == version + 1, resumed
    model_metadata = dict(session.metadata)
    learning_config = dict(learner.config)
    # Release the first model before checking the supported context budget.
    del real_loss, learner, session, base_parameters
    gc.collect()
    torch.cuda.empty_cache()
    fresh.reset()
    long_message = "hello " * 1700
    prepared, inputs, _ = fresh.prepare(long_message)
    assert inputs.input_ids.shape[1] > 1700
    long_update = reloaded.learn(prepared[:-1], long_message)
    assert long_update["optimizer_steps"] == 1, long_update
    assert long_update.get("loss_after") is not None, long_update
    reloaded.close()
    report = {
        "status": "passed", "config": learning_config, "model": model_metadata,
        "turns": turns, "events": events, "serving_isolation_checks": len(isolation_checks),
        "rejections": failures, "base_sha256_before": base_before, "base_sha256_after": base_after,
        "reload_max_logit_difference": (logits - reload_logits).abs().max().item(),
        "selected_logits_max_difference": 0.0, "selected_loss_difference": abs(selected_loss - full_loss),
        "version_before_reload": version, "resumed_update": resumed,
        "near_context_limit_update": long_update,
        "fresh_context_before": baseline, "fresh_context_after": after, "fresh_context_reload": reload_probes,
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "scope": "Mechanics and three-fact recall probe; small fixed loss guards are not a general quality guarantee.",
    }
    (ROOT / "reports/online-checks.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
