"""Controlled answer-only versus answer-and-end-token LoRA comparison."""

import hashlib
import json
import time

from .experiment_acquisition import Experiment, encode_target, normalize, pack
from .chat import ChatSession, ROOT
from .check_update import parameter_hash
from .learning import attach_adapter

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel


def with_end_target(row, tokenizer):
    """Only add the template's actual assistant end token to the existing labels."""
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    last = max(i for i, label in enumerate(row["labels"]) if label != -100)
    assert row["input_ids"][last + 1] == end_id
    assert row["labels"][last + 1] == -100
    labels = list(row["labels"])
    labels[last + 1] = end_id
    assert sum(a != b for a, b in zip(labels, row["labels"])) == 1
    return {**row, "labels": labels}


class StoppingExperiment(Experiment):
    def __init__(self):
        self.config = json.loads((ROOT / "configs/stopping.json").read_text())
        self.learning = json.loads((ROOT / "configs/learning.json").read_text())
        self.out = ROOT / "runs" / f"stopping-{time.time_ns()}"
        self.out.mkdir()
        self.session = ChatSession()
        self.base_parameters = list(self.session.model.named_parameters())
        self.base_hash = parameter_hash(self.base_parameters)
        self.system = list(self.session.messages)
        self.session.model = attach_adapter(self.session.model, self.learning)
        self.model = self.session.model
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.eval()
        self.generation_details = {}
        qa = [[encode_target(self.session.tokenizer,
                             self.system + [{"role": "user", "content": question.format(**fact)}],
                             "assistant", fact["value"])
               for fact in self.config["confirmation"]] for question in self.config["training_questions"]]
        qa_end = [[with_end_target(row, self.session.tokenizer) for row in rows] for rows in qa]
        self.fixtures = {"confirmation": {"qa": qa, "qa_end": qa_end}}
        self.length = max(len(row["input_ids"]) for rows in qa for row in rows)
        self.report = {"protocol": self.config, "learning": self.learning, "model": self.session.metadata,
                       "padded_training_length": self.length, "runs": [], "controls": {},
                       "base_sha256_before": self.base_hash, "directory": self.out.relative_to(ROOT).as_posix(),
                       "scope": "Offline fresh synthetic facts; only added end-token labels differ between arms. No live adapter access.",
                       "loss_reduction": "Standard mean over supervised tokens; the added end token changes both supervision and relative answer-token weighting.",
                       "primary_metric": "Greedy exact match at 16 exposures per fact; 64 is secondary. No checkpoint or setting selection."}
        paths = ["configs/stopping.json", "configs/chat.json", "configs/learning.json",
                 "qwenbrain/experiment_stopping.py", "qwenbrain/experiment_acquisition.py",
                 "qwenbrain/chat.py", "qwenbrain/learning.py", "uv.lock"]
        self.report["source_sha256"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
        self.write()

    @torch.inference_mode()
    def generate(self, question, context=None):
        messages = list(self.system)
        if context:
            messages += [{"role": "user", "content": context}, {"role": "assistant", "content": "Understood."}]
        messages += [{"role": "user", "content": question}]
        text = self.session.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                          enable_thinking=False)
        inputs = self.session.tokenizer(text, add_special_tokens=False, return_tensors="pt").to("cuda")
        end_id = self.session.tokenizer.convert_tokens_to_ids("<|im_end|>")
        with sdpa_kernel(SDPBackend.MATH):
            output = self.model.generate(**inputs, max_new_tokens=self.config["max_new_tokens"], do_sample=False,
                                         eos_token_id=end_id, pad_token_id=self.session.tokenizer.pad_token_id,
                                         use_cache=True)
        tokens = output[0, inputs.input_ids.shape[1]:].tolist()
        self.generation_details[question] = {"generated_token_ids": tokens,
                                             "ended_with_end_token": bool(tokens and tokens[-1] == end_id),
                                             "output_limit_reached": len(tokens) == self.config["max_new_tokens"] and tokens[-1] != end_id}
        return self.session.tokenizer.decode(tokens, skip_special_tokens=True)

    def evaluate(self, facts, with_context=False):
        result = super().evaluate(facts, with_context)
        for item in result["answers"]:
            item.update(self.generation_details[item["question"]])
            answer, expected = normalize(item["answer"]), normalize(item["expected"])
            item["correct_name_then_extra_text"] = answer.startswith(expected) and answer != expected
            item["exact_and_stopped"] = item["exact_match"] and item["ended_with_end_token"]
        for item in result["controls"]:
            item.update(self.generation_details[item["question"]])
        for key in ("ended_with_end_token", "output_limit_reached", "correct_name_then_extra_text", "exact_and_stopped"):
            result[key + "_rate"] = sum(a[key] for a in result["answers"]) / len(result["answers"])
        return result

    def validate_masking(self):
        evidence = []
        for arm in self.config["arms"]:
            row = self.fixtures["confirmation"][arm][0][0]
            losses = []
            for length in (len(row["input_ids"]), self.length):
                batch = pack([row], self.session.tokenizer, length)
                with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16):
                    result = self.model(**batch, use_cache=False)
                    positions = (batch["labels"][0] != -100).nonzero().flatten()
                    logits = result.logits[0, positions - 1].double()
                    targets = batch["labels"][0, positions]
                    reference = (logits.logsumexp(-1) - logits.gather(1, targets[:, None]).squeeze(1)).mean()
                assert abs(reference.item() - result.loss.item()) < 1e-4
                losses.append(result.loss.item())
            assert abs(losses[0] - losses[1]) < 0.02
            evidence.append({"arm": arm, "losses_unpadded_padded": losses,
                             "decoded_targets": self.session.tokenizer.decode([t for t in row["labels"] if t != -100])})
        self.report["masking_checks"] = evidence

    def run(self):
        self.initialize(42)
        self.validate_masking()
        facts = self.config["confirmation"]
        self.report["controls"]["base"] = self.evaluate(facts)
        self.report["controls"]["source_in_context"] = self.evaluate(facts, with_context=True)
        self.write()
        for seed in self.config["confirmation_seeds"]:
            initial = self.initialize(seed)
            for arm in self.config["arms"]:
                self.train_arm("confirmation", seed, arm, initial)
                run = self.report["runs"][-1]
                checkpoint = self.out / f"confirmation-{arm}-{seed}.safetensors"
                # Reload into a deliberately reset adapter, then repeat every final answer.
                self.initialize(seed + 10000)
                set_peft_model_state_dict(self.model, load_file(checkpoint))
                expected = run["checkpoints"][-1]["answers"]
                assert [self.generate(a["question"]) for a in expected] == [a["answer"] for a in expected]
                run["reload_all_final_answers_match"] = True
                run["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                self.write()
        self.report["base_sha256_after"] = parameter_hash(self.base_parameters)
        assert self.report["base_sha256_after"] == self.base_hash
        for path, sha in self.report["source_sha256"].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha
        self.report["status"] = "completed"
        self.write()
        print("RESULTS", self.out, flush=True)


if __name__ == "__main__":
    StoppingExperiment().run()
