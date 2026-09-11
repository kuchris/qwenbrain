"""Persistent local Qwen chat. Milestone 1: inference only, no weight updates."""

import json
import time
import warnings

from .paths import ROOT, configure_storage

configure_storage()

import torch
import transformers
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, TextStreamer

from .sampling import PresencePenalty


class ChatSession:
    def __init__(self):
        self.config = json.loads((ROOT / "configs/chat.json").read_text(encoding="utf-8"))
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This milestone requires an NVIDIA CUDA GPU with BF16 support.")
        manifest_path = ROOT / "models/download-manifest.json"
        if not manifest_path.exists():
            raise RuntimeError("Model files are not ready. Run setup.bat first.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["model_id"] != self.config["model_id"]
                or manifest["revision"] != self.config["model_revision"]):
            raise RuntimeError("Downloaded revision differs from configs/chat.json. Run setup.bat.")
        snapshot = (ROOT / manifest["snapshot"]).resolve()
        if not snapshot.is_relative_to(ROOT / "models"):
            raise RuntimeError("Model snapshot must be inside this project's models directory.")
        for entry in manifest["files"]:
            path = snapshot / entry["name"]
            if not path.is_file() or not path.resolve().is_relative_to(ROOT):
                raise RuntimeError("Missing or externally linked model file. Run setup.bat.")
        torch.set_num_threads(4)
        torch.manual_seed(self.config["seed"])
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            self.model, info = Qwen3_5ForCausalLM.from_pretrained(
                snapshot, local_files_only=True, dtype=torch.bfloat16,
                attn_implementation="sdpa", output_loading_info=True,
            )
        if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
            raise RuntimeError(f"Text model did not load completely: {info}")
        self.model.to("cuda").requires_grad_(False)
        self.model.eval()
        torch.cuda.synchronize()
        self.reset()
        self.metadata = {
            "model_id": self.config["model_id"], "revision": self.config["model_revision"],
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(), "dtype": "BF16", "text_only": True,
            "load_seconds": time.perf_counter() - started,
            "allocated_after_load_mib": torch.cuda.memory_allocated() / 2**20,
            "peak_load_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "unexpected_checkpoint_keys": sorted(info.get("unexpected_keys", [])),
            "load_warnings": list(dict.fromkeys(str(w.message) for w in captured)),
            "attention_backend": "SDPA math for full-attention layers; DeltaNet uses installed Transformers implementation",
            "learning_enabled": False,
        }

    def reset(self):
        self.messages = [{"role": "system", "content": self.config["system_prompt"]}]

    def prepare(self, message):
        if not message.strip():
            raise ValueError("Enter a non-empty message.")
        messages = self.messages + [{"role": "user", "content": message}]
        omitted = 0
        while True:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            if inputs.input_ids.shape[1] + self.config["max_new_tokens"] <= self.config["context_tokens"]:
                return messages, inputs, omitted
            if len(messages) <= 2:
                raise ValueError("This message exceeds the chat's 2,048-token context budget. Please shorten it.")
            del messages[1:3]  # Remove one oldest complete user/assistant turn.
            omitted += 1

    @torch.inference_mode()
    def reply(self, message, stream=False):
        messages, inputs, omitted = self.prepare(message)
        inputs = inputs.to("cuda")
        prompt_tokens = inputs.input_ids.shape[1]
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None
        # Qwen's assistant turn ends at im_end, not the pretraining endoftext token.
        end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with sdpa_kernel(SDPBackend.MATH):
            output = self.model.generate(
                **inputs, max_new_tokens=self.config["max_new_tokens"], do_sample=True,
                temperature=self.config["temperature"], top_p=self.config["top_p"],
                top_k=self.config["top_k"], eos_token_id=end_id,
                logits_processor=[PresencePenalty(self.config["presence_penalty"], prompt_tokens)],
                pad_token_id=self.tokenizer.pad_token_id or end_id,
                use_cache=True, streamer=streamer,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        generated = output[0, prompt_tokens:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        self.messages = messages + [{"role": "assistant", "content": text}]
        metrics = {
            "prompt_tokens": prompt_tokens, "generated_tokens": len(generated),
            "seconds": elapsed, "generated_tokens_per_second": len(generated) / elapsed,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "omitted_old_turns": omitted, "hit_output_limit": generated[-1].item() != end_id,
            "weight_updates": 0,
        }
        return text, metrics


def main():
    print("Loading local Qwen3.5-0.8B...", flush=True)
    session = ChatSession()
    transcript = ROOT / "data/conversations" / f"chat-{time.time_ns()}.jsonl"
    transcript.parent.mkdir(parents=True, exist_ok=True)

    def record(item):
        with transcript.open("a", encoding="utf-8") as output:
            output.write(json.dumps(item, ensure_ascii=False) + "\n")

    record({"event": "start", "config": session.config, "metadata": session.metadata})
    print(f"Model loaded once on {session.metadata['gpu']}. Learning is OFF for this first milestone.")
    print("Enter a message. /reset clears context; blank input or /quit exits.")
    print(f"Conversation saved locally: {transcript.relative_to(ROOT)}", flush=True)
    while True:
        try:
            message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message or message == "/quit":
            break
        if message == "/reset":
            session.reset()
            record({"event": "reset"})
            print("Conversation context cleared. Model remains loaded.")
            continue
        try:
            print("Qwen: ", end="", flush=True)
            answer, metrics = session.reply(message, stream=True)
        except ValueError as error:
            print(error)
            continue
        record({"event": "turn", "user": message, "assistant": answer, "metrics": metrics})
        print(f"[{metrics['generated_tokens']} tokens, {metrics['seconds']:.2f}s, learning OFF]", flush=True)
        if metrics["omitted_old_turns"]:
            print(f"Older context trimmed: {metrics['omitted_old_turns']} turn(s).")
        if metrics["hit_output_limit"]:
            print("[Reply reached the 256-token output limit.]")
    record({"event": "end"})
    print("Chat closed.")


if __name__ == "__main__":
    main()
