"""One-seed, matched-batch retention pilot after successful Q&A acquisition."""

import hashlib
import json
import random
import time

from .experiment_acquisition import Experiment, encode_target, pack
from .chat import ChatSession, ROOT
from .check_update import parameter_hash
from .learning import attach_adapter, update_once

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.nn.attention import SDPBackend, sdpa_kernel


class RetentionExperiment(Experiment):
    """Reuse the frozen acquisition evaluation/scoring protocol, with a new stream."""

    def __init__(self):
        candidates = []
        for path in (ROOT / "runs").glob("acquisition-*/results.json"):
            report = json.loads(path.read_text())
            if report.get("status") == "completed":
                candidates.append((path, report))
        path, acquisition = max(candidates, key=lambda pair: pair[0].stat().st_mtime_ns)
        assert acquisition["selected_arm"] == "qa", "This pilot requires a completed Q&A confirmation."
        self.config = acquisition["protocol"]
        self.learning = acquisition["learning"]
        self.out = ROOT / "runs" / f"retention-{time.time_ns()}"
        self.out.mkdir()
        self.old = self.config["confirmation"]
        self.new = [dict(subject=s, value=v) for s, v in [
            ("Milo Hart", "Juniper"), ("Zara Penn", "Granite"), ("Oren Fisk", "Saffron"),
            ("Vera Lark", "Tundra"), ("Enzo Reed", "Nimbus"), ("Alma Kent", "Sequoia"),
            ("Nico Ward", "Lagoon"), ("Rhea Pike", "Silver"),
        ]]
        self.session = ChatSession()
        self.base_parameters = list(self.session.model.named_parameters())
        self.base_hash = parameter_hash(self.base_parameters)
        self.system = list(self.session.messages)
        self.session.model = attach_adapter(self.session.model, self.learning)
        self.model = self.session.model
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.eval()
        self.initial_path = path.parent / "confirmation-qa-42.safetensors"
        self.initial = load_file(self.initial_path)
        self.rows = {}
        for name, facts in (("old", self.old), ("new", self.new)):
            self.rows[name] = [[encode_target(
                self.session.tokenizer,
                self.system + [{"role": "user", "content": template.format(**fact)}],
                "assistant", fact["value"],
            ) for fact in facts] for template in self.config["training_questions"]]
        self.length = max(len(row["input_ids"]) for variants in self.rows.values() for rows in variants for row in rows)
        self.report = {"protocol": "retention-v1", "seed": 42, "old_facts": self.old, "new_facts": self.new,
                       "initial_checkpoint": self.initial_path.relative_to(ROOT).as_posix(),
                       "initial_checkpoint_sha256": hashlib.sha256(self.initial_path.read_bytes()).hexdigest(),
                       "learning": self.learning, "evaluation_config": self.config,
                       "padded_length": self.length, "batch_size": 8, "milestones": [16, 64],
                       "base_sha256_before": self.base_hash, "runs": [],
                       "scope": "Follow-on one-seed pilot; seed42 final64-exposure QA adapter chosen by fixed first-seed rule, not best confirmation score. Equal total batch/step budget; replay halves new-fact exposures."}
        set_peft_model_state_dict(self.model, self.initial)
        self.report["before"] = {"old": self.evaluate(self.old), "new": self.evaluate(self.new)}
        self.write()

    def run(self):
        for replay in (False, True):
            set_peft_model_state_dict(self.model, self.initial)
            self.model.set_adapter("default")
            assert all(not p.requires_grad for _, p in self.base_parameters)
            run = {"arm": "replay_50_percent" if replay else "no_replay", "checkpoints": [], "steps": [],
                   "initial_adapter_sha256": parameter_hash(list((n, p) for n, p in self.model.named_parameters() if "lora_" in n))}
            self.report["runs"].append(run)
            counters = {"old_examples": 0, "new_examples": 0, "padded_tokens": 0,
                        "nonpadding_tokens": 0, "target_tokens": 0, "training_seconds": 0.0}
            for step in range(1, 65):
                # Each fact cycles through all four forms; selection alternates halves.
                variant = ((step - 1) // 2) % 4
                if replay:
                    offset = ((step - 1) % 2) * 4
                    rows = self.rows["new"][variant][offset:offset + 4] + self.rows["old"][variant][offset:offset + 4]
                    counters["old_examples"] += 4
                    counters["new_examples"] += 4
                else:
                    rows = list(self.rows["new"][variant])
                    counters["new_examples"] += 8
                random.Random(42000 + step).shuffle(rows)
                batch = pack(rows, self.session.tokenizer, self.length)
                with sdpa_kernel(SDPBackend.MATH):
                    metrics = update_once(self.model, batch, self.learning)
                assert all(torch.isfinite(p).all() for p in self.model.parameters() if p.requires_grad)
                counters["padded_tokens"] += batch["input_ids"].numel()
                counters["nonpadding_tokens"] += batch["attention_mask"].sum().item()
                counters["target_tokens"] += (batch["labels"] != -100).sum().item()
                counters["training_seconds"] += metrics["seconds"]
                run["steps"].append({"step": step, "loss": metrics["loss_before"], "seconds": metrics["seconds"],
                                     "peak_allocated_mib": metrics["peak_allocated_mib"]})
                if step in self.report["milestones"]:
                    result = {"steps": step, **counters, "old": self.evaluate(self.old), "new": self.evaluate(self.new)}
                    run["checkpoints"].append(result)
                    self.write()
                    print(run["arm"], step, 'old=', result["old"]["exact_match"], 'new=', result["new"]["exact_match"], flush=True)
            save_file(self.state(), self.out / (run["arm"] + ".safetensors"))
        self.report["base_sha256_after"] = parameter_hash(self.base_parameters)
        assert self.report["base_sha256_after"] == self.base_hash
        assert hashlib.sha256(self.initial_path.read_bytes()).hexdigest() == self.report["initial_checkpoint_sha256"]
        self.report["source_sha256"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in (ROOT / "qwenbrain/experiment_retention.py", ROOT / "qwenbrain/experiment_acquisition.py", ROOT / "qwenbrain/learning.py")}
        self.report["status"] = "completed"
        self.write()
        print("RESULTS", self.out, flush=True)


if __name__ == "__main__":
    RetentionExperiment().run()
