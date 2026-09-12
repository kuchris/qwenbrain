"""CPU-only explicit-input memory demo; answers are source values, not generations."""

import argparse
from datetime import date
import json
from pathlib import Path
import re
import shlex
import sqlite3
import sys

from .memory import ProjectMemoryStore
from .paths import ROOT


def iso_date(value):
    try:
        if date.fromisoformat(value).isoformat() == value:
            return value
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Dates must use YYYY-MM-DD and be valid calendar dates.")


def subject_name(value):
    name = " ".join(value.split())
    if not re.fullmatch(r"\w+(?: \w+)*", name):
        raise argparse.ArgumentTypeError("Names support letters, numbers, underscores and spaces.")
    return name


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--db", type=Path, default=ROOT / "data/memory-demo.sqlite")
    commands = result.add_subparsers(dest="command")
    remember = commands.add_parser("remember", help="Append a project claim; never overwrite an old row.")
    remember.add_argument("subject", type=subject_name)
    remember.add_argument("relation", choices=("project_name",))
    remember.add_argument("value")
    remember.add_argument("--valid-from", required=True, type=iso_date)
    remember.add_argument("--recorded-at", type=iso_date, default=date.today().isoformat())
    ask = commands.add_parser("ask", help="Read a project value using the documented question templates.")
    ask.add_argument("question")
    ask.add_argument("--known-at", type=iso_date, default=date.today().isoformat())
    walkthrough = commands.add_parser("walkthrough", help="Replay a checked scenario in separate processes.")
    walkthrough.add_argument("--output", type=Path, help="New output directory; existing paths are refused.")
    return result


def answer(args, store):
    result = {"answer": "UNKNOWN", "mode": "abstention", "source_id": None,
              "reason": "unsupported_query", "known_at": args.known_at}
    question = " ".join(args.question.split())
    current = re.fullmatch(r"What is (.+)'s project called\?", question, re.I)
    historical = re.fullmatch(r"What was (.+)'s project called on (\d{4}-\d{2}-\d{2})\?", question, re.I)
    match = historical or current
    if not match:
        return result
    try:
        subject = subject_name(match[1])
        as_of = iso_date(historical[2]) if historical else args.known_at
    except argparse.ArgumentTypeError:
        return {**result, "reason": "invalid_query"}
    result.update(subject=subject, as_of=as_of)
    if as_of > args.known_at:
        return {**result, "reason": "future_query"}
    source = store.lookup(subject, as_of, args.known_at)
    if source is None or source["value"] is None:
        return {**result, "reason": "no_applicable_source"}
    return {**result, "answer": source["value"], "mode": "source_value", "reason": "found",
            "source_id": source["id"], "valid_from": source["valid_from"],
            "recorded_at": source["recorded_at"], "source_input": source["content"]}


def execute(args, source_input=None):
    if args.command == "remember" and (not args.value.strip() or args.value.strip().casefold() == "unknown"):
        raise ValueError("Project name must be nonempty and cannot be UNKNOWN.")
    args.db.parent.mkdir(parents=True, exist_ok=True)
    store = ProjectMemoryStore(args.db)
    try:
        if args.command == "ask":
            return answer(args, store)
        # Serialize ID allocation with the append, including concurrent CLI writers.
        store.connection.execute("BEGIN IMMEDIATE")
        source_id = store.connection.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM versions").fetchone()[0]
        if source_input is None:
            source_input = shlex.join(["remember", args.subject, args.relation, args.value,
                                       "--valid-from", args.valid_from, "--recorded-at", args.recorded_at])
        store.append([{"id": source_id, "subject": args.subject, "value": args.value,
                       "content": source_input, "valid_from": args.valid_from,
                       "recorded_at": args.recorded_at}])
        return {"status": "stored", "mode": "explicit_input", "source_id": source_id,
                "subject": args.subject, "relation": args.relation, "value": args.value,
                "valid_from": args.valid_from, "recorded_at": args.recorded_at}
    finally:
        store.close()


def interactive(db):
    print("QwenBrain memory demo | CPU only | explicit input -> SQLite -> source_value")
    print(f"Database: {db.resolve()}")
    print('Try: remember "Aster Rook" project_name Cobalt --valid-from 2026-01-01')
    print('Then: ask "What is Aster Rook\'s project called?"')
    print("help shows commands; quit closes the process. Every stored record survives restart.")
    command_parser = parser()
    while True:
        try:
            line = input("memory> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nMemory demo closed.")
            return 0
        if line in ("quit", "/quit", "exit"):
            print("Memory demo closed.")
            return 0
        if not line:
            continue
        if line == "help":
            command_parser.print_help()
            continue
        try:
            args = command_parser.parse_args(["--db", str(db), *shlex.split(line)])
            if args.command not in ("remember", "ask"):
                raise ValueError("Use remember or ask here; run walkthrough outside the interactive shell.")
            print(json.dumps(execute(args, line), ensure_ascii=False, indent=2))
        except SystemExit:
            pass  # argparse has printed help or an input error; keep the shell usable.
        except (ValueError, OSError, sqlite3.Error) as error:
            print(f"Error: {error}", file=sys.stderr)


def main(argv=None):
    tokens = sys.argv[1:] if argv is None else argv
    command_parser = parser()
    args = command_parser.parse_args(tokens)
    try:
        if args.command == "walkthrough":
            from .memory_walkthrough import run
            return run(args.output)
        if args.command is None:
            return interactive(args.db)
        print(json.dumps(execute(args), ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, sqlite3.Error) as error:
        command_parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
