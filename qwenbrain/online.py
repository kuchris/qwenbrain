"""Sequential chat and candidate updates sharing one frozen base model."""

import json
import math
import msvcrt
import os
import time
from copy import deepcopy

from .learning import attach_adapter, build_message_batch, select_scored_logits, update_once
from .paths import ROOT

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn.attention import SDPBackend, sdpa_kernel


class OnlineLearner:
    def __init__(self, session, checkpoint=None):
        self.session = session
        self.config = json.loads((ROOT / "configs/learning.json").read_text(encoding="utf-8"))
        self.checkpoint = checkpoint or ROOT / "adapters/live.safetensors"
        if not self.checkpoint.resolve().is_relative_to(ROOT):
            raise ValueError("Learning checkpoint must stay inside this project.")
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        # A second chat must not overwrite the same live adapter. OS releases on exit/crash.
        self.lock = self.checkpoint.with_suffix(".lock").open("a+b")
        if self.lock.seek(0, 2) == 0:
            self.lock.write(b"0")
            self.lock.flush()
        self.lock.seek(0)
        try:
            msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self.lock.close()
            raise RuntimeError("This adapter is already open in another learning chat. Close that chat first.") from None
        try:
            self._initialize()
        except BaseException:
            self.close()
            raise

    def close(self):
        self.lock.close()

    def _initialize(self):
        self.version = 0
        self.session.model = attach_adapter(self.session.model, self.config, "serving")
        self.model = self.session.model
        self.model.add_adapter("candidate", deepcopy(self.model.peft_config["serving"]))
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.set_adapter("serving", inference_mode=True)
        self.model.eval()
        self.guards = []
        for text in self.config["guard_texts"]:
            batch = self.session.tokenizer(text, add_special_tokens=False, return_tensors="pt").to("cuda")
            batch["labels"] = batch.input_ids.clone()
            self.guards.append(batch)
        self.identity = {
            "format": 1, "model_id": self.session.config["model_id"],
            "revision": self.session.config["model_revision"], "learning_config": self.config,
        }
        if self.checkpoint.exists():
            with safe_open(self.checkpoint, framework="pt", device="cpu") as saved:
                metadata = json.loads(saved.metadata()["qwenbrain"])
                if metadata["identity"] != self.identity:
                    raise ValueError("Saved adapter has a different model or learning configuration.")
                state = {name: saved.get_tensor(name) for name in saved.keys()}
            self._load_state("serving", state)
            self.version = metadata["version"]
            self.guard_baseline = metadata["guard_baseline"]
            if (not isinstance(self.version, int) or self.version < 1
                    or len(self.guard_baseline) != len(self.guards)
                    or not all(math.isfinite(v) for v in self.guard_baseline)):
                raise ValueError("Invalid adapter version or guard baseline.")
        else:
            self.guard_baseline = [self.loss(batch) for batch in self.guards]
        self.model.requires_grad_(False)
        self.session.metadata["learning_enabled"] = True
        self.session.metadata["adapter_version"] = self.version

    def state(self, name="serving"):
        return {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(
            self.model, adapter_name=name, save_embedding_layers=False,
        ).items()}

    def _load_state(self, name, state):
        expected = get_peft_model_state_dict(self.model, adapter_name=name, save_embedding_layers=False)
        if state.keys() != expected.keys() or any(
            state[k].shape != expected[k].shape or state[k].dtype != expected[k].dtype
            or not torch.isfinite(state[k]).all() for k in expected
        ):
            raise ValueError("Adapter tensors have missing keys, incompatible shapes/dtypes, or nonfinite values.")
        set_peft_model_state_dict(self.model, state, adapter_name=name)

    @torch.inference_mode()
    def loss(self, batch):
        with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(**batch, use_cache=False).loss.item()

    def _save(self, state, version):
        # One atomic file contains both weights and version/config/baseline metadata.
        temporary = self.checkpoint.with_suffix(".tmp")
        metadata = {"identity": self.identity, "version": version, "guard_baseline": self.guard_baseline}
        try:
            save_file(state, temporary, metadata={"qwenbrain": json.dumps(metadata)})
            with temporary.open("r+b") as output:
                os.fsync(output.fileno())
            os.replace(temporary, self.checkpoint)
        finally:
            temporary.unlink(missing_ok=True)

    def learn(self, history, message):
        started = time.perf_counter()
        previous = self.state()
        result = {"status": "rejected", "version_before": self.version, "version_after": self.version,
                  "optimizer_steps": 0, "weight_updates": 0}
        try:
            batch, trace = build_message_batch(
                self.session.tokenizer, history, message, self.session.config["context_tokens"],
            )
            result["targets"] = trace
            batch = select_scored_logits({k: v.cuda() for k, v in batch.items()})
            self._load_state("candidate", previous)
            self.model.set_adapter("candidate")
            self.model.eval()
            result["loss_before"] = self.loss(batch)
            metrics = update_once(self.model, batch, self.config)
            result.update(optimizer_steps=1, step_seconds=metrics["seconds"],
                          gradient_norm_before_clip=metrics["gradient_norm_before_clip"],
                          peak_allocated_mib=metrics["peak_allocated_mib"])
            candidate = self.state("candidate")
            if not all(torch.isfinite(v).all() for v in candidate.values()):
                raise ValueError("Candidate contains nonfinite weights.")
            result["max_parameter_change"] = max((candidate[k] - previous[k]).abs().max().item() for k in previous)
            if result["max_parameter_change"] == 0:
                raise ValueError("Candidate weights did not change.")
            result["loss_after"] = self.loss(batch)
            if not math.isfinite(result["loss_after"]) or not result["loss_after"] < result["loss_before"]:
                raise ValueError("Candidate did not reduce the new message loss.")
            result["guard_losses"] = [self.loss(guard) for guard in self.guards]
            result["guard_baseline"] = self.guard_baseline
            if any(not math.isfinite(value) or value > baseline + self.config["max_guard_loss_increase"]
                   for value, baseline in zip(result["guard_losses"], self.guard_baseline)):
                raise ValueError("Candidate exceeded the fixed guard loss threshold.")
            # Serving is untouched during training and validation. Snapshot permits save-error rollback.
            self._load_state("serving", candidate)
            self._save(candidate, self.version + 1)
            self.version += 1
            result.update(status="accepted", version_after=self.version, weight_updates=1)
        except (ValueError, RuntimeError, OSError) as error:
            self._load_state("serving", previous)
            result["reason"] = str(error)
        except BaseException:
            self._load_state("serving", previous)
            raise
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.set_adapter("serving", inference_mode=True)
            self.model.requires_grad_(False)
            self.model.eval()
        result["total_seconds"] = time.perf_counter() - started
        self.session.metadata["adapter_version"] = self.version
        return result

    def turn(self, message, stream=False, record=None):
        version = self.version
        answer, metrics = self.session.reply(message, stream=stream)
        metrics["adapter_version"] = version
        if record:
            record({"event": "turn", "user": message, "assistant": answer, "metrics": metrics})
            record({"event": "update_start", "version": version})
        # reply() stores the bounded context followed by exactly this user/assistant pair.
        result = self.learn(self.session.messages[:-2], message)
        if record:
            record({"event": "update", **result})
        return answer, metrics, result
