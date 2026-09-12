"""Matched frozen/oracle/SQLite/QA-replay pilot using existing retention facts."""

import argparse
import json
import random
import time

from .paths import ROOT, configure_storage

configure_storage()

from .chat import ChatSession
from .check_update import parameter_hash
from .experiment_acquisition import encode_target, normalize, pack
from .learning import attach_adapter, update_once
from .memory import MemoryStore
from .artifacts import completed_report, file_hash, input_record, new_run, retention_checkpoint

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.nn.attention import SDPBackend, sdpa_kernel


class MemoryExperiment:
    def __init__(self, retention_report, output=None):
        self.source_path, source = completed_report(retention_report)
        self.config = source["evaluation_config"]
        self.learning = source["learning"]
        self.initial_path = retention_checkpoint(self.source_path, source)
        self.out = new_run("memory", output)
        self.records = []
        for group in ("old", "new"):
            for fact in source[group + "_facts"]:
                self.records.append({**fact, "id": len(self.records), "group": group,
                                     "content": self.config["statements"][0].format(**fact)})
        self.session = ChatSession()
        self.model = self.session.model
        self.base_parameters = list(self.model.named_parameters())
        self.base_hash = parameter_hash(self.base_parameters)
        self.report = {
            "protocol": "memory-comparison-v1", "status": "running", "seed": 42,
            "model": self.session.metadata, "evaluation_config": self.config,
            "learning": self.learning, "records": self.records, "arms": {},
            "directory": self.out.relative_to(ROOT).as_posix(),
            "inputs": {"retention": input_record(self.source_path, self.out), "initial_checkpoint": input_record(self.initial_path, self.out)},
            "initial_checkpoint": source["initial_checkpoint"],
            "initial_checkpoint_sha256": file_hash(self.initial_path),
            "base_sha256_before": self.base_hash,
            "selection": "Diagnostic reuse of existing facts and previously observed better replay16 setting; not fresh confirmation or an optimal parameter-memory claim.",
            "weight_protocol": "Reload acquisition seed42 QA64; repeat retention-v1 50% replay for 16 steps, answer-only labels, fresh AdamW per step. No new method selection.",
            "retrieval_protocol": "SQLite whole-name lookup, top1; no evidence on zero or multiple matching names. Only the question is passed to retrieve(). Not semantic search.",
            "generation_protocol": "Same original system prompt, greedy decoding, 24-token limit, im_end EOS, fresh full prefill per query. O/R add one raw source turn and Understood. F/W do not receive evidence.",
            "cost_scope": "Generation excludes model loading; query time includes lookup and prompt preparation. W update time excludes historical acquisition, evaluation and checkpoint IO. SQLite write time excludes model loading; full local costs are reported separately.",
        }
        paths = ["qwenbrain/experiment_memory.py", "qwenbrain/memory.py", "qwenbrain/chat.py",
                 "qwenbrain/experiment_acquisition.py", "qwenbrain/learning.py", "configs/chat.json",
                 "qwenbrain/artifacts.py", self.source_path.relative_to(ROOT).as_posix(), "uv.lock"]
        if "base_sha256_after" in source:
            assert self.base_hash == source["base_sha256_after"]
        self.report["source_sha256"] = {p: file_hash(ROOT / p) for p in paths}
        # Inspect only the live file's checksum; never load it into this experiment.
        self.live_path = ROOT / "adapters/live.safetensors"
        self.live_before = file_hash(self.live_path) if self.live_path.exists() else None
        self.write()

    def write(self):
        (self.out / "results.json").write_text(json.dumps(self.report, indent=2), encoding="utf-8")

    @torch.inference_mode()
    def generate(self, question, evidence):
        messages = list(self.session.messages)
        if evidence:
            messages += [{"role": "user", "content": evidence[0]["content"]},
                         {"role": "assistant", "content": "Understood."}]
        messages += [{"role": "user", "content": question}]
        prompt = self.session.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.session.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
        prompt_tokens = inputs.input_ids.shape[1]
        assert prompt_tokens + self.config["max_new_tokens"] <= self.session.config["context_tokens"]
        end_id = self.session.tokenizer.convert_tokens_to_ids("<|im_end|>")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with sdpa_kernel(SDPBackend.MATH):
            output = self.model.generate(
                **inputs, max_new_tokens=self.config["max_new_tokens"], do_sample=False,
                eos_token_id=end_id, pad_token_id=self.session.tokenizer.pad_token_id, use_cache=True,
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        tokens = output[0, prompt_tokens:].tolist()
        return {"answer": self.session.tokenizer.decode(tokens, skip_special_tokens=True),
                "generated_token_ids": tokens, "prompt_tokens": prompt_tokens,
                "generation_seconds": seconds, "ended_with_end_token": bool(tokens and tokens[-1] == end_id),
                "output_limit_reached": len(tokens) == self.config["max_new_tokens"] and tokens[-1] != end_id,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}

    def adapter_hash(self):
        return parameter_hash([(n, p) for n, p in self.model.named_parameters() if "lora_" in n])

    def evaluate(self, arm, records=None):
        self.model.eval()
        before = self.adapter_hash()
        questions = []
        for record in self.records if records is None else records:
            for template in self.config["test_questions"]:
                questions.append({"question": template.format(**record), "expected": record["value"],
                                  "gold_source_id": record["id"], "group": record["group"]})
        questions += [{"question": c["question"], "expected": c["answer"], "group": "control",
                       "gold_source_id": None} for c in self.config["controls"]]
        answers = []
        for index, item in enumerate(questions):
            started = time.perf_counter()
            if arm == "R":
                evidence = self.store.retrieve(item["question"])
            elif arm == "O" and item["gold_source_id"] is not None:
                record = self.records[item["gold_source_id"]]
                evidence = [{"id": record["id"], "content": record["content"]}]
            else:
                evidence = []
            lookup_seconds = time.perf_counter() - started
            result = self.generate(item["question"], evidence)
            result["query_seconds"] = time.perf_counter() - started
            exact = normalize(result["answer"]) == normalize(item["expected"])
            answers.append({**item, **result, "evidence": evidence, "lookup_seconds": lookup_seconds,
                            "exact_match": exact, "exact_and_stopped": exact and result["ended_with_end_token"],
                            "correct_prefix_with_extra_text": normalize(result["answer"]).startswith(normalize(item["expected"])) and not exact})
            if (index + 1) % 8 == 0:
                print(arm, "evaluated", index + 1, "/", len(questions), flush=True)
        assert before == self.adapter_hash()
        result = {"answers": answers, "adapter_unchanged_during_evaluation": True}
        for group in ("old", "new", "control"):
            rows = [r for r in answers if r["group"] == group]
            if rows:
                result[group] = {"correct": sum(r["exact_match"] for r in rows), "total": len(rows),
                                 "exact_and_stopped": sum(r["exact_and_stopped"] for r in rows)}
        facts = [r for r in answers if r["group"] != "control"]
        result["fact_correct"] = sum(r["exact_match"] for r in facts)
        result["fact_total"] = len(facts)
        result["fact_stopped"] = sum(r["ended_with_end_token"] for r in facts)
        result["fact_exact_and_stopped"] = sum(r["exact_and_stopped"] for r in facts)
        result["mean_query_seconds"] = sum(r["query_seconds"] for r in answers) / len(answers)
        result["mean_generation_seconds"] = sum(r["generation_seconds"] for r in answers) / len(answers)
        result["mean_prompt_tokens"] = sum(r["prompt_tokens"] for r in answers) / len(answers)
        if arm in ("O", "R"):
            result["recall_at_1"] = sum(bool(r["evidence"]) and r["evidence"][0]["id"] == r["gold_source_id"] for r in facts) / len(facts)
            result["control_evidence_injections"] = sum(bool(r["evidence"]) for r in answers if r["group"] == "control")
        print(arm, "facts", result["fact_correct"], "/", len(facts), "controls", result["control"], flush=True)
        return result

    def train_replay(self):
        self.model = attach_adapter(self.model, self.learning)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        set_peft_model_state_dict(self.model, load_file(self.initial_path))
        assert all(not p.requires_grad for _, p in self.base_parameters)
        self.report["weight_before"] = self.evaluate("W_before", [r for r in self.records if r["group"] == "old"])
        rows = {}
        for group in ("old", "new"):
            rows[group] = [[encode_target(
                self.session.tokenizer, self.session.messages + [{"role": "user", "content": question.format(**fact)}],
                "assistant", fact["value"],
            ) for fact in self.records if fact["group"] == group] for question in self.config["training_questions"]]
        length = max(len(row["input_ids"]) for variants in rows.values() for variant in variants for row in variant)
        self.report["weight_steps"] = []
        self.report["weight_training_length"] = length
        started = time.perf_counter()
        for step in range(1, 17):
            variant = ((step - 1) // 2) % 4
            offset = ((step - 1) % 2) * 4
            selected = rows["new"][variant][offset:offset + 4] + rows["old"][variant][offset:offset + 4]
            random.Random(42000 + step).shuffle(selected)
            batch = pack(selected, self.session.tokenizer, length)
            with sdpa_kernel(SDPBackend.MATH):
                metrics = update_once(self.model, batch, self.learning)
            assert all(torch.isfinite(p).all() for p in self.model.parameters() if p.requires_grad)
            self.report["weight_steps"].append({"step": step, "seconds": metrics["seconds"], "loss": metrics["loss_before"],
                                                 "target_tokens": (batch["labels"] != -100).sum().item()})
        self.report["weight_update_wall_seconds"] = time.perf_counter() - started
        self.report["weight_update_compute_seconds"] = sum(s["seconds"] for s in self.report["weight_steps"])
        checkpoint = self.out / "qa-replay16.safetensors"
        started = time.perf_counter()
        save_file({k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(self.model).items()}, checkpoint)
        self.report["weight_checkpoint_write_seconds"] = time.perf_counter() - started
        self.report["weight_checkpoint_bytes"] = checkpoint.stat().st_size
        self.report["weight_checkpoint_sha256"] = file_hash(checkpoint)
        # Evaluate the saved artifact, after deliberately discarding in-memory adapter values.
        expected_hash = self.adapter_hash()
        set_peft_model_state_dict(self.model, load_file(self.initial_path))
        set_peft_model_state_dict(self.model, load_file(checkpoint))
        assert self.adapter_hash() == expected_hash
        self.report["weight_checkpoint_reload_matches"] = True
        self.write()

    def run(self):
        started = time.perf_counter()
        path = self.out / "memory.sqlite"
        self.store = MemoryStore(path)
        self.store.append(self.records)
        self.store.close()
        self.report["sqlite_write_seconds"] = time.perf_counter() - started
        self.report["sqlite_bytes"] = path.stat().st_size
        self.report["sqlite_sha256_before"] = file_hash(path)
        self.store = MemoryStore(path)
        # Warm up identical generation code once; exclude startup effects from arm means.
        self.generate("Reply with only OK.", [])
        for arm in ("F", "O", "R"):
            self.report["arms"][arm] = self.evaluate(arm)
            self.write()
        self.train_replay()
        self.report["arms"]["W"] = self.evaluate("W")
        baseline = {a["question"]: a for a in self.report["arms"]["F"]["answers"] if a["group"] == "control"}
        for result in self.report["arms"].values():
            result["controls_changed_from_correct_to_wrong"] = sum(
                baseline[a["question"]]["exact_match"] and not a["exact_match"]
                for a in result["answers"] if a["group"] == "control")
        self.store.close()
        self.report["sqlite_sha256_after"] = file_hash(path)
        assert self.report["sqlite_sha256_before"] == self.report["sqlite_sha256_after"]
        oracle, retrieval = self.report["arms"]["O"], self.report["arms"]["R"]
        self.report["oracle_retrieval_outputs_identical"] = [a["generated_token_ids"] for a in oracle["answers"]] == [a["generated_token_ids"] for a in retrieval["answers"]]
        self.report["base_sha256_after"] = parameter_hash(self.base_parameters)
        assert self.report["base_sha256_after"] == self.base_hash
        assert file_hash(self.initial_path) == self.report["initial_checkpoint_sha256"]
        assert (file_hash(self.live_path) if self.live_path.exists() else None) == self.live_before
        self.report["live_checkpoint_unchanged"] = True
        for path, sha in self.report["source_sha256"].items():
            assert file_hash(ROOT / path) == sha
        self.report["status"] = "completed"
        self.write()
        print("RESULTS", self.out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention-report", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    MemoryExperiment(args.retention_report, args.output).run()
