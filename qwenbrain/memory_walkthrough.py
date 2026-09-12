"""Replay the memory-demo CLI with a fresh database and a new process per command."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .paths import ROOT


def run(output=None):
    out = Path(output).resolve() if output else ROOT / "runs" / f"memory-demo-{time.time_ns()}"
    out.mkdir(parents=True, exist_ok=False)
    db = out / "memory.sqlite"
    report = {"status": "running", "mode": "source_value", "model_loaded": False,
              "process_per_command": True, "steps": []}
    current = "What is Aster Rook's project called?"
    historical = "What was Aster Rook's project called on 2026-01-15?"
    steps = [
        ("Encode first name", ["remember", "Aster Rook", "project_name", "Cobalt", "--valid-from", "2026-01-01", "--recorded-at", "2026-01-02"], {"source_id": 1, "status": "stored"}),
        ("Read after process restart", ["ask", current, "--known-at", "2026-01-02"], {"answer": "Cobalt", "source_id": 1}),
        ("Record a new name", ["remember", "Aster Rook", "project_name", "Vesper", "--valid-from", "2026-02-01", "--recorded-at", "2026-02-02"], {"source_id": 2, "status": "stored"}),
        ("Read new current state", ["ask", current, "--known-at", "2026-02-02"], {"answer": "Vesper", "source_id": 2}),
        ("Read the original historical state", ["ask", historical, "--known-at", "2026-02-02"], {"answer": "Cobalt", "source_id": 1}),
        ("Append a late historical correction", ["remember", "Aster Rook", "project_name", "Laurel", "--valid-from", "2026-01-01", "--recorded-at", "2026-02-10"], {"source_id": 3, "status": "stored"}),
        ("Current name survives historical correction", ["ask", current, "--known-at", "2026-02-10"], {"answer": "Vesper", "source_id": 2}),
        ("History uses the correction", ["ask", historical, "--known-at", "2026-02-10"], {"answer": "Laurel", "source_id": 3}),
        ("Earlier observation cutoff excludes the correction", ["ask", historical, "--known-at", "2026-02-02"], {"answer": "Cobalt", "source_id": 1}),
        ("Unknown person", ["ask", "What is Belen Frost's project called?", "--known-at", "2026-02-10"], {"answer": "UNKNOWN", "reason": "no_applicable_source"}),
        ("Unsupported attribute", ["ask", "What is Aster Rook's favorite color?", "--known-at", "2026-02-10"], {"answer": "UNKNOWN", "reason": "unsupported_query"}),
        ("Before the first effective record", ["ask", "What was Aster Rook's project called on 2025-12-01?", "--known-at", "2026-02-10"], {"answer": "UNKNOWN", "reason": "no_applicable_source"}),
        ("Future query", ["ask", "What was Aster Rook's project called on 2026-03-01?", "--known-at", "2026-02-10"], {"answer": "UNKNOWN", "reason": "future_query"}),
    ]
    print("QwenBrain | deterministic source-value walkthrough | no model or GPU", flush=True)
    for label, arguments, expected in steps:
        before = hashlib.sha256(db.read_bytes()).hexdigest() if db.exists() else None
        process = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "qwenbrain.memory_demo", "--db", str(db), *arguments],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        after = hashlib.sha256(db.read_bytes()).hexdigest() if db.exists() else None
        actual = json.loads(process.stdout) if process.returncode == 0 else {}
        passed = process.returncode == 0 and all(actual.get(k) == v for k, v in expected.items())
        if arguments[0] == "ask":
            passed = passed and before == after
        report["steps"].append({"label": label, "arguments": arguments, "expected": expected,
                                "actual": actual, "exit_code": process.returncode, "stderr": process.stderr,
                                "database_sha256_before": before, "database_sha256_after": after,
                                "passed": passed})
        report["status"] = "running" if passed else "failed"
        (out / "transcript.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{'PASS' if passed else 'FAIL'} | {label} | {actual.get('answer', actual.get('status'))} "
              f"| source_id={actual.get('source_id')}", flush=True)
        if not passed:
            print(f"Inspect {out / 'transcript.json'}", flush=True)
            return 1
    report.update(status="completed", passed_steps=len(steps))
    (out / "transcript.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{len(steps)}/{len(steps)} scenario steps passed. Artifacts: {out}", flush=True)
    return 0
