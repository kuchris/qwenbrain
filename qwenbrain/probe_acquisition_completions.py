"""Supplementary final-checkpoint probes of the trained statement prefix itself."""

import json
import os
import re

os.environ["HF_HUB_OFFLINE"] = "1"

from .chat import ChatSession, ROOT
from .learning import attach_adapter
from .check_update import parameter_hash

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel


@torch.inference_mode()
def generate(session, text, max_tokens):
    inputs = session.tokenizer(text, add_special_tokens=False, return_tensors="pt").to("cuda")
    with sdpa_kernel(SDPBackend.MATH):
        output = session.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=True,
                                        eos_token_id=session.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                                        pad_token_id=session.tokenizer.pad_token_id)
    return session.tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)


def probe(session, fact):
    text = session.tokenizer.apply_chat_template(
        session.messages + [{"role": "user", "content": f"{fact['subject']}'s project is named"}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    # Intentionally continue the user statement, not an assistant QA prompt.
    text = text.rsplit("<|im_end|>", 1)[0]
    completion = generate(session, text, 8)
    first_clause = re.split(r"[.\n]", completion.strip(), maxsplit=1)[0]
    normalized = re.sub(r"[\s\W_]+", "", first_clause.casefold())
    return {**fact, "completion": completion, "first_clause_correct": normalized == fact["value"].lower()}


def main():
    completed = []
    for path in (ROOT / "runs").glob("acquisition-*/results.json"):
        report = json.loads(path.read_text())
        if report.get("status") == "completed":
            completed.append((path, report))
    path, report = max(completed, key=lambda pair: pair[0].stat().st_mtime_ns)
    session = ChatSession()
    session.model = attach_adapter(session.model, report["learning"])
    session.model.requires_grad_(False).eval()
    results = {"scope": "Post-hoc direct-completion diagnostic at final (64-exposure) checkpoints only; not used for format selection.", "runs": []}
    for phase in ("pilot", "confirmation"):
        with session.model.disable_adapter():
            answers = [probe(session, fact) for fact in report["protocol"][phase]]
        results[phase + "_base"] = answers
    for run in report["runs"]:
        state = load_file(path.parent / f"{run['phase']}-{run['arm']}-{run['seed']}.safetensors")
        set_peft_model_state_dict(session.model, state)
        before = parameter_hash(list(session.model.named_parameters()))
        expected = run["checkpoints"][-1]["answers"][0]
        prompt = session.tokenizer.apply_chat_template(
            session.messages + [{"role": "user", "content": expected["question"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        assert generate(session, prompt, report["protocol"]["max_new_tokens"]) == expected["answer"]
        answers = [probe(session, fact) for fact in report["protocol"][run["phase"]]]
        assert before == parameter_hash(list(session.model.named_parameters()))
        result = {"phase": run["phase"], "arm": run["arm"], "seed": run["seed"], "answers": answers,
                  "correct": sum(a["first_clause_correct"] for a in answers), "total": len(answers),
                  "reloaded_first_qa_matches": True}
        results["runs"].append(result)
        print(result["phase"], result["arm"], result["seed"], result["correct"], "/", result["total"], flush=True)
    (path.parent / "completions.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("RESULTS", path.parent, flush=True)


if __name__ == "__main__":
    main()
