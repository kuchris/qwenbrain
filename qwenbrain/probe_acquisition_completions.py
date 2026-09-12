"""Supplementary final-checkpoint probes of the trained statement prefix itself."""

import json
import argparse
import os
import re

os.environ["HF_HUB_OFFLINE"] = "1"

from .chat import ChatSession, ROOT
from .learning import attach_adapter
from .check_update import parameter_hash
from .artifacts import checked_input, completed_report, file_hash, input_record

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


def main(acquisition_report, output=None):
    path, report = completed_report(acquisition_report)
    target = (ROOT / output).resolve() if output else path.parent / "completions.json"
    if not target.is_relative_to(ROOT / "runs") or target.exists():
        raise ValueError("Completion output must be a new file under runs/.")
    target.parent.mkdir(parents=True, exist_ok=True)
    checkpoints = {name: checked_input(path.parent, {"path": name, "sha256": digest})
                   for name, digest in report["checkpoint_sha256"].items()}
    session = ChatSession()
    session.model = attach_adapter(session.model, report["learning"])
    session.model.requires_grad_(False).eval()
    results = {"status": "running", "inputs": {"acquisition": input_record(path, target.parent)},
               "scope": "Post-hoc direct-completion diagnostic at final (64-exposure) checkpoints only; not used for format selection.", "runs": []}
    for phase in ("pilot", "confirmation"):
        with session.model.disable_adapter():
            answers = [probe(session, fact) for fact in report["protocol"][phase]]
        results[phase + "_base"] = answers
    for run in report["runs"]:
        state = load_file(checkpoints[f"{run['phase']}-{run['arm']}-{run['seed']}.safetensors"])
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
    assert file_hash(path) == results["inputs"]["acquisition"]["sha256"]
    results["status"] = "completed"
    target.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("RESULTS", target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-report", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    main(args.acquisition_report, args.output)
