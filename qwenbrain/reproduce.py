"""Run the acquisition/completion/retention/four-arm GPU pipeline with explicit inputs."""

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tomllib

from .artifacts import file_hash, input_record, new_run, project_file
from .paths import ROOT
from .reproduction_summary import summarize


def preflight(config, learning):
    data = json.loads(config.read_text(encoding="utf-8"))
    if (data["milestones"] != [1, 4, 16, 64] or data["confirmation_seeds"] != [42, 43, 44]
            or len(data["confirmation"]) != 8 or len(data["pilot"]) != 4
            or any(len(data[key]) != 4 for key in ("statements", "training_questions"))):
        raise ValueError("This pipeline preserves the acquisition-v1 sizes, seeds, four training forms and 1/4/16/64 exposure schedule.")
    json.loads(learning.read_text(encoding="utf-8"))
    versions = {}
    for requirement in tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]:
        name, expected = requirement.split("==")
        actual = importlib.metadata.version(name)
        if actual.split("+")[0] != expected:
            raise ValueError(f"Expected {requirement}, found {actual}. Run uv sync --locked.")
        versions[name] = actual
    manifest_path = project_file("models/download-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pinned = json.loads((ROOT / "reports/download-manifest.json").read_text(encoding="utf-8"))
    chat = json.loads((ROOT / "configs/chat.json").read_text(encoding="utf-8"))
    if (manifest["model_id"], manifest["revision"]) != (chat["model_id"], chat["model_revision"]) or manifest != pinned:
        raise ValueError("Local model manifest differs from the pinned project manifest. Run setup.bat.")
    snapshot = (ROOT / manifest["snapshot"]).resolve()
    for entry in manifest["files"]:
        path = project_file(snapshot / entry["name"])
        if not path.is_relative_to(ROOT / "models") or file_hash(path) != entry["sha256"]:
            raise ValueError(f"Model checksum or location mismatch: {entry['name']}")
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "model_manifest_sha256": file_hash(manifest_path), "model_files_verified": len(manifest["files"])}


def run(config, learning, output):
    config, learning = project_file(config), project_file(learning)
    print("Verifying locked packages and every pinned model file...", flush=True)
    environment = preflight(config, learning)
    out = new_run("reproduction", output)
    (out / "inputs").mkdir()
    for source, name in ((config, "acquisition.json"), (learning, "learning.json")):
        shutil.copyfile(source, out / "inputs" / name)
    paths = [*sorted((ROOT / "qwenbrain").glob("*.py")), config, learning,
             ROOT / "configs/chat.json", ROOT / "uv.lock", ROOT / "pyproject.toml"]
    source_hashes = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}
    live = ROOT / "adapters/live.safetensors"
    live_before = file_hash(live) if live.exists() else None
    commands = [
        ("acquisition", "qwenbrain.experiment_acquisition", ["--config", str(out / "inputs/acquisition.json"), "--learning-config", str(out / "inputs/learning.json"), "--output", str(out / "acquisition")], out / "acquisition/results.json"),
        ("completions", "qwenbrain.probe_acquisition_completions", ["--acquisition-report", str(out / "acquisition/results.json"), "--output", str(out / "completions.json")], out / "completions.json"),
        ("retention", "qwenbrain.experiment_retention", ["--acquisition-report", str(out / "acquisition/results.json"), "--output", str(out / "retention")], out / "retention/results.json"),
        ("memory", "qwenbrain.experiment_memory", ["--retention-report", str(out / "retention/results.json"), "--output", str(out / "memory")], out / "memory/results.json"),
    ]
    manifest = {"protocol": "memory-reproduction-v1", "status": "running", "environment": environment,
                "source_sha256": source_hashes, "stages": [], "fresh_output_directory": True}
    def save():
        (out / "pipeline.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save()
    try:
        for name, module, arguments, result_path in commands:
            command = [sys.executable, "-X", "utf8", "-m", module, *arguments]
            entry = {"name": name, "status": "running", "command": ["python", "-X", "utf8", "-m", module,
                     *[Path(a).relative_to(ROOT).as_posix() if a.startswith(str(ROOT)) else a for a in arguments]], "log": name + ".log"}
            manifest["stages"].append(entry)
            save()
            print(f"STAGE {name}: {out / entry['log']}", flush=True)
            started = time.perf_counter()
            with (out / entry["log"]).open("x", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", env={**os.environ, "HF_HUB_OFFLINE": "1"},
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
                finally:
                    process.stdout.close()
            entry.update(exit_code=code, wall_seconds=time.perf_counter() - started)
            if code != 0:
                entry["status"] = "failed"
                raise RuntimeError(f"Stage {name} failed; inspect {out / entry['log']}.")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "completed":
                raise ValueError(f"Stage {name} did not produce a completed report.")
            entry.update(status="completed", result=input_record(result_path, out))
            save()
        if any(file_hash(ROOT / path) != digest for path, digest in source_hashes.items()):
            raise ValueError("Source/config changed during this run.")
        if (file_hash(live) if live.exists() else None) != live_before:
            raise ValueError("Live adapter changed during this run.")
        manifest.update(status="completed", live_checkpoint_unchanged=True, source_files_unchanged=True)
        save()
        summarize(out)
        print(f"COMPLETED {out / 'summary.md'}", flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        save()
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Run all four GPU stages into a new directory.")
    run_parser.add_argument("--config", default="configs/acquisition.json")
    run_parser.add_argument("--learning-config", default="configs/learning.json")
    run_parser.add_argument("--output")
    summary_parser = commands.add_parser("summarize", help="Recount one completed run without loading a model.")
    summary_parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    try:
        if args.command == "run":
            run(args.config, args.learning_config, args.output)
        else:
            directory = project_file(Path(args.run_dir) / "pipeline.json").parent
            summarize(directory)
            print(directory / "summary.md")
    except (OSError, ValueError, RuntimeError, KeyError, importlib.metadata.PackageNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
