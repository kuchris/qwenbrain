"""Offline, preregistered format/exposure comparison; never writes the live adapter."""

import argparse
import json
import os
import random

os.environ["HF_HUB_OFFLINE"] = "1"

from .chat import ChatSession, ROOT
from .check_update import parameter_hash
from .learning import attach_adapter, update_once
from .artifacts import file_hash, input_record, new_run, normalize, project_file

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn.attention import SDPBackend, sdpa_kernel


def encode_target(tokenizer, history, role, target):
    """Locate only the final content span; no question, template, or EOS labels."""
    marker = "ACQUISITION_TARGET_MARKER"
    assert marker not in json.dumps(history) + target
    def render(content):
        return tokenizer.apply_chat_template(history + [{"role": role, "content": content}],
                                             tokenize=False, add_generation_prompt=False, enable_thinking=False)
    prefix, suffix = render(marker).split(marker)
    text = render(target)
    assert text == prefix + target + suffix
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    start, end = len(prefix), len(prefix) + len(target)
    labels = [token if i > 0 and a >= start and b <= end and b > a
              and token not in tokenizer.all_special_ids else -100
              for i, (token, (a, b)) in enumerate(zip(encoded.input_ids, encoded.offset_mapping))]
    scored = [t for t in labels if t != -100]
    assert scored and tokenizer.decode(scored) == target, (role, target, scored)
    return {"input_ids": encoded.input_ids, "labels": labels, "target": target}


def pack(rows, tokenizer, length=None):
    length = length or max(len(row["input_ids"]) for row in rows)
    assert all(len(row["input_ids"]) <= length for row in rows)
    ids = torch.full((len(rows), length), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, row in enumerate(rows):
        n = len(row["input_ids"])
        ids[i, :n] = torch.tensor(row["input_ids"])
        mask[i, :n] = 1
        labels[i, :n] = torch.tensor(row["labels"])
    return {"input_ids": ids.cuda(), "attention_mask": mask.cuda(), "labels": labels.cuda()}


class Experiment:
    def __init__(self, config_path="configs/acquisition.json", learning_path="configs/learning.json", output=None):
        config_path, learning_path = project_file(config_path), project_file(learning_path)
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.learning = json.loads(learning_path.read_text(encoding="utf-8"))
        self.out = new_run("acquisition", output)
        self.live_path = ROOT / "adapters/live.safetensors"
        self.live_before = file_hash(self.live_path) if self.live_path.exists() else None
        self.session = ChatSession()
        self.base_parameters = list(self.session.model.named_parameters())
        self.base_hash = parameter_hash(self.base_parameters)
        self.system = list(self.session.messages)
        self.session.model = attach_adapter(self.session.model, self.learning)
        self.model = self.session.model
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.eval()
        self.fixtures = {phase: self.prepare_fixtures(self.config[phase]) for phase in ("pilot", "confirmation")}
        self.length = max(len(row["input_ids"]) for formats in self.fixtures.values()
                          for variants in formats.values() for rows in variants for row in rows)
        self.report = {"protocol": self.config, "learning": self.learning, "model": self.session.metadata,
                       "status": "running", "checkpoint_sha256": {},
                       "inputs": {"config": input_record(config_path, self.out), "learning": input_record(learning_path, self.out)},
                       "padded_training_length": self.length, "runs": [], "controls": {},
                       "base_sha256_before": self.base_hash, "directory": self.out.relative_to(ROOT).as_posix(),
                       "budget": "Equal full-logit padded batch dimensions, steps, and fact exposures within each phase; unpadded and supervised tokens plus measured time also reported.",
                       "selection_rule": "Choose varied or qa by mean pilot exact-match accuracy over all four milestones; tie-break by mean candidate-rank accuracy; final tie favors varied. Confirm selected arm against raw with fixed settings.",
                       "scope": "Synthetic project-name acquisition; no live promotion or retention/replay experiment."}
        paths = [config_path, learning_path, ROOT / "configs/chat.json", ROOT / "uv.lock",
                 ROOT / "qwenbrain/experiment_acquisition.py", ROOT / "qwenbrain/learning.py",
                 ROOT / "qwenbrain/chat.py", ROOT / "qwenbrain/artifacts.py"]
        self.report["source_sha256"] = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}
        self.write()

    def write(self):
        (self.out / "results.json").write_text(json.dumps(self.report, indent=2), encoding="utf-8")

    def prepare_fixtures(self, facts):
        result = {}
        for arm in ("raw", "varied", "qa"):
            variants = []
            for variant in range(4):
                rows = []
                for fact in facts:
                    if arm == "qa":
                        q = self.config["training_questions"][variant].format(**fact)
                        history = self.system + [{"role": "user", "content": q}]
                        row = encode_target(self.session.tokenizer, history, "assistant", fact["value"])
                    else:
                        statement = self.config["statements"][variant if arm == "varied" else 0].format(**fact)
                        row = encode_target(self.session.tokenizer, self.system, "user", statement)
                    rows.append(row)
                variants.append(rows)
            result[arm] = variants
        return result

    def state(self):
        return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(
            self.model, save_embedding_layers=False).items()}

    def initialize(self, seed):
        torch.manual_seed(seed)
        for module in self.model.modules():
            if hasattr(module, "reset_lora_parameters"):
                module.reset_lora_parameters("default", True)
        self.model.set_adapter("default")
        self.model.eval()
        return self.state()

    @torch.inference_mode()
    def generate(self, question, context=None):
        messages = list(self.system)
        if context:
            messages += [{"role": "user", "content": context}, {"role": "assistant", "content": "Understood."}]
        messages += [{"role": "user", "content": question}]
        text = self.session.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                          enable_thinking=False)
        inputs = self.session.tokenizer(text, add_special_tokens=False, return_tensors="pt").to("cuda")
        with sdpa_kernel(SDPBackend.MATH):
            output = self.model.generate(**inputs, max_new_tokens=self.config["max_new_tokens"], do_sample=False,
                                         eos_token_id=self.session.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                                         pad_token_id=self.session.tokenizer.pad_token_id, use_cache=True)
        return self.session.tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    @torch.inference_mode()
    def candidate_scores(self, question, values):
        rows = [encode_target(self.session.tokenizer, self.system + [{"role": "user", "content": question}],
                              "assistant", value) for value in values]
        batch = pack(rows, self.session.tokenizer)
        with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
        positions = batch["labels"][:, 1:] != -100
        labels = batch["labels"][:, 1:][positions]
        selected = logits[:, :-1][positions].float()
        logps = selected.gather(1, labels[:, None]).squeeze(1) - selected.logsumexp(-1)
        counts = positions.sum(-1).tolist()
        return {value: part.mean().item() for value, part in zip(values, logps.split(counts))}

    def evaluate(self, facts, with_context=False):
        self.model.eval()
        before = parameter_hash(list((n, p) for n, p in self.model.named_parameters() if "lora_" in n))
        answers = []
        for fact in facts:
            for template in self.config["test_questions"]:
                question = template.format(**fact)
                context = self.config["statements"][0].format(**fact) if with_context else None
                answer = self.generate(question, context)
                entry = {"subject": fact["subject"], "question": question, "expected": fact["value"],
                         "answer": answer, "exact_match": normalize(answer) == normalize(fact["value"])}
                if not with_context:
                    scores = self.candidate_scores(question, [x["value"] for x in facts])
                    entry.update(candidate_logp=scores, gold_logp=scores[fact["value"]],
                                 rank_correct=scores[fact["value"]] > max(v for k, v in scores.items() if k != fact["value"]))
                answers.append(entry)
        controls = [{**item, "output": self.generate(item["question"])} for item in self.config["controls"]]
        for item in controls:
            item["exact_match"] = normalize(item["answer"]) == normalize(item["output"])
        after = parameter_hash(list((n, p) for n, p in self.model.named_parameters() if "lora_" in n))
        assert before == after
        result = {"answers": answers, "exact_match": sum(x["exact_match"] for x in answers) / len(answers),
                  "controls": controls, "control_accuracy": sum(x["exact_match"] for x in controls) / len(controls)}
        if not with_context:
            result.update(rank_accuracy=sum(x["rank_correct"] for x in answers) / len(answers),
                          mean_gold_logp=sum(x["gold_logp"] for x in answers) / len(answers))
        return result

    def validate_masking(self):
        # Variable-length right padding must leave loss unchanged for the same example.
        for arm in ("raw", "qa"):
            row = self.fixtures["pilot"][arm][0][0]
            losses = []
            for length in (len(row["input_ids"]), self.length):
                batch = pack([row], self.session.tokenizer, length)
                with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16):
                    result = self.model(**batch, use_cache=False)
                    pos = (batch["labels"][0] != -100).nonzero().flatten()
                    selected = result.logits[0, pos - 1].double()
                    targets = batch["labels"][0, pos]
                    reference = (selected.logsumexp(-1) - selected.gather(1, targets[:, None]).squeeze(1)).mean()
                assert abs(reference.item() - result.loss.item()) < 1e-4
                losses.append(result.loss.item())
            assert abs(losses[0] - losses[1]) < 0.02, losses
        self.report["masking_and_padding_checks"] = "passed; target text exact; independent next-token loss reference <1e-4; padded/unpadded loss difference <0.02"

    def train_arm(self, phase, seed, arm, initial):
        set_peft_model_state_dict(self.model, initial)
        self.model.set_adapter("default")
        assert all(not p.requires_grad for _, p in self.base_parameters)
        assert all("lora_" in n for n, p in self.model.named_parameters() if p.requires_grad)
        facts = self.config[phase]
        run = {"phase": phase, "seed": seed, "arm": arm, "checkpoints": [], "steps": [],
               "initial_adapter_sha256": parameter_hash(list((n, p) for n, p in self.model.named_parameters() if "lora_" in n))}
        self.report["runs"].append(run)
        counters = {"padded_tokens": 0, "nonpadding_tokens": 0, "target_tokens": 0, "training_seconds": 0.0}
        for exposure in range(1, max(self.config["milestones"]) + 1):
            rows = list(self.fixtures[phase][arm][(exposure - 1) % 4])
            random.Random(seed * 1000 + exposure).shuffle(rows)
            batch = pack(rows, self.session.tokenizer, self.length)
            # Checkpoint recomputation happens during backward, after update_once's
            # inner forward context ends. Keep the backend fixed for both passes.
            with sdpa_kernel(SDPBackend.MATH):
                metrics = update_once(self.model, batch, self.learning)
            assert all(torch.isfinite(p).all() for p in self.model.parameters() if p.requires_grad)
            counters["padded_tokens"] += batch["input_ids"].numel()
            counters["nonpadding_tokens"] += batch["attention_mask"].sum().item()
            counters["target_tokens"] += (batch["labels"] != -100).sum().item()
            counters["training_seconds"] += metrics["seconds"]
            run["steps"].append({"exposure": exposure, "loss": metrics["loss_before"], "seconds": metrics["seconds"],
                                 "peak_allocated_mib": metrics["peak_allocated_mib"]})
            if exposure in self.config["milestones"]:
                result = self.evaluate(facts)
                run["checkpoints"].append({"exposures_per_fact": exposure, **counters, **result})
                print(f"{phase} seed={seed} {arm} exposures={exposure}: exact={result['exact_match']:.3f} rank={result['rank_accuracy']:.3f} controls={result['control_accuracy']:.2f}", flush=True)
                self.write()
        checkpoint = self.out / f"{phase}-{arm}-{seed}.safetensors"
        save_file(self.state(), checkpoint)
        self.report.setdefault("checkpoint_sha256", {})[checkpoint.name] = file_hash(checkpoint)
        self.write()

    def run(self):
        initial = self.initialize(42)
        self.validate_masking()
        self.report["controls"]["pilot_base"] = self.evaluate(self.config["pilot"])
        self.report["controls"]["pilot_context"] = self.evaluate(self.config["pilot"], with_context=True)
        for arm in ("raw", "varied", "qa"):
            self.train_arm("pilot", 42, arm, initial)
        candidates = [r for r in self.report["runs"] if r["arm"] in ("varied", "qa")]
        winner = max(candidates, key=lambda r: (sum(c["exact_match"] for c in r["checkpoints"]),
                                               sum(c["rank_accuracy"] for c in r["checkpoints"]), r["arm"] == "varied"))["arm"]
        self.report["selected_arm"] = winner
        print("Selected from pilot:", winner, flush=True)
        self.initialize(42)
        self.report["controls"]["confirmation_base"] = self.evaluate(self.config["confirmation"])
        self.report["controls"]["confirmation_context"] = self.evaluate(self.config["confirmation"], with_context=True)
        for seed in self.config["confirmation_seeds"]:
            initial = self.initialize(seed)
            for arm in ("raw", winner):
                self.train_arm("confirmation", seed, arm, initial)
        self.report["base_sha256_after"] = parameter_hash(self.base_parameters)
        assert self.report["base_sha256_after"] == self.base_hash
        assert (file_hash(self.live_path) if self.live_path.exists() else None) == self.live_before
        self.report["live_checkpoint_unchanged"] = True
        for path, digest in self.report["source_sha256"].items():
            assert file_hash(ROOT / path) == digest
        self.report["status"] = "completed"
        self.write()
        print("RESULTS", self.out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/acquisition.json")
    parser.add_argument("--learning-config", default="configs/learning.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    Experiment(args.config, args.learning_config, args.output).run()
