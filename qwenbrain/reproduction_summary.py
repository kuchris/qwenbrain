"""Recount completed GPU pipeline outputs using only the Python standard library."""

from collections import defaultdict
import json
from pathlib import Path
import re

from .artifacts import checked_input, completed_report, file_hash, normalize


def score(rows, prediction="answer", expected="expected"):
    return {"correct": sum(normalize(r[prediction]) == normalize(r[expected]) for r in rows), "total": len(rows)}


def summarize(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "pipeline.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Pipeline is not completed; partial results are not a final summary.")
    reports, report_paths = {}, {}
    if [stage["name"] for stage in manifest["stages"]] != ["acquisition", "completions", "retention", "memory"]:
        raise ValueError("Expected exactly the four pipeline stages in order.")
    for stage in manifest["stages"]:
        path = checked_input(directory, stage["result"])
        _, report = completed_report(path)
        for entry in report.get("inputs", {}).values():
            checked_input(path.parent, entry)
        reports[stage["name"]] = report
        report_paths[stage["name"]] = path
    for stage, parent in (("completions", "acquisition"), ("retention", "acquisition"), ("memory", "retention")):
        actual = checked_input(report_paths[stage].parent, reports[stage]["inputs"][parent])
        if actual != report_paths[parent]:
            raise ValueError(f"Stage {stage} references a different {parent} report.")
    acquisition, retention, memory = (reports[name] for name in ("acquisition", "retention", "memory"))
    if len({r["base_sha256_after"] for r in (acquisition, retention, memory)}) != 1:
        raise ValueError("Stages did not use the same frozen base.")
    if any(r["base_sha256_before"] != r["base_sha256_after"] or not r["live_checkpoint_unchanged"]
           for r in (acquisition, retention, memory)):
        raise ValueError("Frozen-base or live-checkpoint protection failed.")
    grouped = defaultdict(list)
    for run in acquisition["runs"]:
        for checkpoint in run["checkpoints"]:
            grouped[(run["phase"], run["arm"], checkpoint["exposures_per_fact"])].extend(checkpoint["answers"])
    acquisition_scores = [{"phase": phase, "arm": arm, "exposures": exposures, **score(rows)}
                          for (phase, arm, exposures), rows in grouped.items()]
    completion_groups = defaultdict(list)
    for run in reports["completions"]["runs"]:
        completion_groups[(run["phase"], run["arm"])].extend(run["answers"])
    completions = [{"phase": phase, "arm": arm, "total": len(rows), "correct": sum(
        normalize(re.split(r"[.\n]", r["completion"].strip(), maxsplit=1)[0]) == normalize(r["value"]) for r in rows)}
        for (phase, arm), rows in completion_groups.items()]
    retention_scores = [{"arm": run["arm"], "steps": checkpoint["steps"],
                         **{group: score(checkpoint[group]["answers"]) for group in ("old", "new")}}
                        for run in retention["runs"] for checkpoint in run["checkpoints"]]
    arms = {}
    for arm, result in memory["arms"].items():
        rows = result["answers"]
        arms[arm] = {group: score([r for r in rows if r["group"] == group]) for group in ("old", "new", "control")}
        facts = [r for r in rows if r["group"] != "control"]
        arms[arm]["facts"] = score(facts)
        arms[arm]["exact_and_stopped"] = sum(normalize(r["answer"]) == normalize(r["expected"]) and r["ended_with_end_token"] for r in facts)
        if arm in ("O", "R"):
            arms[arm]["correct_source_selections"] = sum(bool(r["evidence"]) and r["evidence"][0]["id"] == r["gold_source_id"] for r in facts)
    summary = {"protocol": "memory-reproduction-summary-v1", "status": "completed",
               "model": memory["model"], "acquisition": acquisition_scores, "completions": completions,
               "retention": retention_scores, "memory": arms,
               "pipeline_sha256": file_hash(directory / "pipeline.json"),
               "scope": "Reproduction of the existing synthetic protocol, not fresh-data confirmation or a universal comparison. Counts include repeated question forms and training seeds. Accuracy is recounted from saved text, not cached score counters."}
    lines = ["# GPU memory pipeline: reproduced results", "", summary["scope"], "",
             "## Four-arm comparison", "", "| Arm | Old | New | All facts | Controls |",
             "| --- | ---: | ---: | ---: | ---: |"]
    fraction = lambda entry: f"{entry['correct']}/{entry['total']}"
    for arm, row in arms.items():
        lines.append("| " + " | ".join([arm, *[fraction(row[group]) for group in ("old", "new", "facts", "control")]]) + " |")
    lines += ["", "## Acquisition", "", "| Phase | Arm | Exposures | Correct |", "| --- | --- | ---: | ---: |"]
    lines += [f"| {r['phase']} | {r['arm']} | {r['exposures']} | {fraction(r)} |" for r in acquisition_scores]
    lines += ["", "## Completion probes", "", "| Phase | Arm | Correct |", "| --- | --- | ---: |"]
    lines += [f"| {r['phase']} | {r['arm']} | {fraction(r)} |" for r in completions]
    lines += ["", "## Retention", "", "| Arm | Steps | Old | New |", "| --- | ---: | ---: | ---: |"]
    lines += [f"| {r['arm']} | {r['steps']} | {fraction(r['old'])} | {fraction(r['new'])} |" for r in retention_scores]
    lines += ["", "F = frozen; O = oracle evidence; R = exact-name SQLite lookup; W = seed-42 QA64 plus 16 replay steps.",
              "", "The default protocol has eight confirmation identities and sixteen identities in the four-arm comparison.",
              "LoRA training still excludes the end token. Timing is recorded per stage and query in the raw reports; this is not an isolated performance benchmark.",
              "", "Frozen-base equality, input hashes and live-checkpoint protection passed. See pipeline.json for commands, logs, inputs and stage report hashes.", ""]
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (directory / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
