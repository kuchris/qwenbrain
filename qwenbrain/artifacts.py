"""CPU-only paths and checksum contracts for reproducible experiment artifacts."""

import hashlib
import json
import os
from pathlib import Path
import re
import time

from .paths import ROOT


def normalize(text):
    return re.sub(r"[\s\W_]+", "", text.casefold())


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def project_file(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Input must be inside the project: {value}")
    if not path.is_file():
        raise ValueError(f"Missing input: {value}. Run its preceding pipeline stage first.")
    return path


def new_run(prefix, output=None):
    path = (ROOT / output).resolve() if output else ROOT / "runs" / f"{prefix}-{time.time_ns()}"
    if not path.is_relative_to(ROOT / "runs") or path == ROOT / "runs":
        raise ValueError("Experiment output must be a new directory under runs/.")
    path.mkdir(parents=True, exist_ok=False)
    return path


def completed_report(value):
    path = project_file(value)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "completed":
        raise ValueError(f"Input report is not completed: {value}")
    return path, report


def input_record(path, directory):
    return {"path": Path(os.path.relpath(path, directory)).as_posix(), "sha256": file_hash(path)}


def checked_input(directory, entry):
    path = project_file(directory / entry["path"])
    if file_hash(path) != entry["sha256"]:
        raise ValueError(f"Input checksum mismatch: {path}")
    return path


def acquisition_checkpoint(path, report):
    runs = [r for r in report["runs"] if (r["phase"], r["arm"], r["seed"]) == ("confirmation", "qa", 42)]
    if report.get("selected_arm") != "qa" or len(runs) != 1 or runs[0]["checkpoints"][-1]["exposures_per_fact"] != 64:
        raise ValueError("Retention-v1 requires selected QA and the seed-42 final 64-exposure confirmation checkpoint.")
    name = "confirmation-qa-42.safetensors"
    digest = report.get("checkpoint_sha256", {}).get(name)
    if not digest:
        raise ValueError("Acquisition report lacks checkpoint hashes. Run the acquisition stage with the current runner.")
    return checked_input(path.parent, {"path": name, "sha256": digest})


def retention_checkpoint(path, report):
    entry = report.get("inputs", {}).get("initial_checkpoint")
    if entry:
        return checked_input(path.parent, entry)
    # Historical reports use project-relative paths; keep explicit replay possible.
    checkpoint = project_file(report["initial_checkpoint"])
    if file_hash(checkpoint) != report["initial_checkpoint_sha256"]:
        raise ValueError(f"Input checkpoint checksum mismatch: {checkpoint}")
    return checkpoint
