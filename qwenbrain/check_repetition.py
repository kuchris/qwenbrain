"""Replay the reported identity-answer repetition through the real GPU chat path."""

import argparse
from collections import Counter
import json
import os
import re

os.environ["HF_HUB_OFFLINE"] = "1"

from .chat import ChatSession, ROOT
from .sampling import PresencePenalty

import torch


def repetition_count(text):
    # Repeated six-word spans catch the user's loop without hard-coding its wording.
    words = re.findall(r"\w+", text.lower())
    return max(Counter(tuple(words[i:i + 6]) for i in range(len(words) - 5)).values(), default=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/repetition-checks.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--single", action="store_true", help="Omit the preceding greeting for minimization")
    args = parser.parse_args()
    # Numerical semantics: exclude prompt, apply once despite duplicate tokens,
    # keep batches independent, and preserve the caller's unmodified scores.
    scores = torch.tensor([[3., -1., 2., 4., 0.], [1., 2., 3., 4., -2.]])
    saved = scores.clone()
    ids = torch.tensor([[0, 1, 2, 2, 3], [3, 2, 4, 4, 1]])
    penalty = PresencePenalty(2.0, prompt_length=2)
    torch.testing.assert_close(penalty(ids[:, :2], scores), scores)
    torch.testing.assert_close(penalty(ids, scores), torch.tensor([[3., -1., 0., 2., 0.], [1., 0., 3., 4., -4.]]))
    torch.testing.assert_close(scores, saved)
    session = ChatSession()
    cases = []
    for seed in args.seeds:
        session.reset()
        torch.manual_seed(seed)
        turns = []
        for prompt in (["im chris, who are you"] if args.single else ["hi", "im chris, who are you"]):
            text, metrics = session.reply(prompt)
            turns.append({"user": prompt, "assistant": text, "metrics": metrics,
                          "max_repeated_six_word_span": repetition_count(text)})
        passed = all(t["max_repeated_six_word_span"] < 3 and not t["metrics"]["hit_output_limit"] for t in turns)
        cases.append({"seed": seed, "passed": passed, "turns": turns})
    report = {"status": "passed" if all(c["passed"] for c in cases) else "failed",
              "configuration": session.config, "cases": cases,
              "processor_checks": "Prompt excluded; penalty once per unique generated token; separate batches; original scores unchanged",
              "scope": "Bounded reproduction: repeated six-word span three times or reaching the token limit fails. Not a general language-quality score."}
    (ROOT / args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    assert report["status"] == "passed", "Repeated phrase or output limit in the reported chat scenario"


if __name__ == "__main__":
    main()
