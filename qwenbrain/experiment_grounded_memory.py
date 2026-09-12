"""Paired raw/checked answers on fixed fresh temporal-memory fixtures."""

import argparse
import gc
from collections import Counter

from .paths import ROOT, configure_storage

configure_storage()

from .experiment_memory import file_hash
from .experiment_temporal_memory import TemporalExperiment
from .grounding import canonical_value, check_answer
from .memory import ProjectMemoryStore

import torch


class GroundedExperiment(TemporalExperiment):
    store_class = ProjectMemoryStore
    arms = ("R",)

    def __init__(self, seed):
        super().__init__("explicit", config_path=f"configs/grounded-memory-{seed}.json", run_prefix=f"grounded-memory-{seed}")
        for path in ("qwenbrain/experiment_grounded_memory.py", "qwenbrain/grounding.py"):
            self.report["source_sha256"][path] = file_hash(ROOT / path)
        self.report["scope"] = (
            "Paired raw and checked outputs from the same single generation per query. "
            "Typed project values are supplied by the fixture encoder and persisted in SQLite. "
            "The checker gets only question, observation date, proposed answer and store; never expected answer or gold id. "
            "Mismatch/unsupported/missing-source becomes UNKNOWN; false abstentions are not filled. "
            "Simple arithmetic/capital grammar bypasses verification. No general entailment or automatic fact extraction claim."
        )
        self.report["score_normalization"] = "Casefold and whitespace normalization; preserve punctuation."
        self.write()

    def evaluate_phase(self, arm, phase):
        result = super().evaluate_phase(arm, phase)
        for item in result["answers"]:
            checked = check_answer(item["question"], phase["known_at"], item["answer"], self.store)
            item["checked"] = checked
            item["exact_match"] = canonical_value(item["answer"]) == canonical_value(item["expected"])
            item["exact_and_stopped"] = item["exact_match"] and item["ended_with_end_token"]
            item["checked_exact_match"] = canonical_value(checked["answer"]) == canonical_value(item["expected"])
            item["checker_changed_answer"] = checked["answer"] != item["answer"]
            if checked["verified"]:
                # This checks the actual applicable DB row, not a fixture label.
                source = self.store.retrieve(item["question"], phase["known_at"])[0]
                assert checked["source_id"] == source["id"]
                assert checked["answer"] == source["value"]
        result["paired_scores"] = {}
        for group, predicate in (
            ("current", lambda a: a["category"] == "current"),
            ("history", lambda a: a["category"] == "history"),
            ("unknown", lambda a: a["expected"] == "UNKNOWN"),
            ("control", lambda a: a["category"] == "control"),
        ):
            rows = [a for a in result["answers"] if predicate(a)]
            result["paired_scores"][group] = {
                "total": len(rows), "raw_correct": sum(a["exact_match"] for a in rows),
                "checked_correct": sum(a["checked_exact_match"] for a in rows),
                "correct_raw_blocked": sum(a["exact_match"] and not a["checked_exact_match"] for a in rows),
            }
        result["checker_decisions"] = dict(Counter(a["checked"]["decision"] for a in result["answers"]))
        result["raw_nonexact_nonabstaining_personal_answers"] = sum(
            a["category"] != "control" and not a["exact_match"] and canonical_value(a["answer"]) != "unknown"
            for a in result["answers"])
        result["checked_nonexact_nonabstaining_personal_answers"] = sum(
            a["category"] != "control" and not a["checked_exact_match"] and canonical_value(a["checked"]["answer"]) != "unknown"
            for a in result["answers"])
        result["raw_false_abstentions"] = sum(
            a["gold_source_id"] is not None and canonical_value(a["answer"]) == "unknown"
            for a in result["answers"])
        result["checked_false_abstentions"] = sum(
            a["gold_source_id"] is not None and canonical_value(a["checked"]["answer"]) == "unknown"
            for a in result["answers"])
        print("CHECKED", phase["name"], result["paired_scores"], flush=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(73, 101, 137))
    arguments = parser.parse_args()
    for seed in [arguments.seed] if arguments.seed is not None else (73, 101, 137):
        experiment = GroundedExperiment(seed)
        experiment.run()
        del experiment
        gc.collect()
        torch.cuda.empty_cache()
