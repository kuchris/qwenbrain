"""Conservative answer checking for the supported personal-project query grammar."""

import re


def canonical_value(text):
    # Preserve meaningful punctuation: C, C++, and C# must not become equal.
    return " ".join(text.split()).casefold()


def query_kind(question):
    text = " ".join(question.strip().split())
    # Only these explicit answer-format suffixes are part of the pilot grammar.
    text = re.sub(
        r" Reply with only (?:its name or UNKNOWN|the color or UNKNOWN|the number|the city name)\.$",
        "", text, flags=re.I,
    )
    general = (r"What is \d+\s*[+-]\s*\d+\?", r"What is the capital of [\w -]+\?")
    if any(re.fullmatch(pattern, text, re.I) for pattern in general):
        return "general"
    patterns = (
        r"Which project belongs to [\w -]+(?: now)?\?",
        r"Give the current name of [\w -]+'s project\.",
        r"What is [\w -]+'s project called\?",
        r"On \d{4}-\d{2}-\d{2}, what (?:was|will) [\w -]+'s project (?:be )?named\?",
        r"What project does [\w -]+ work on\?",
    )
    return "project_name" if any(re.fullmatch(pattern, text, re.I) for pattern in patterns) else "unsupported"


def check_answer(question, known_at, answer, store):
    """Return a source-checked answer without access to benchmark answer labels.

    Mismatches become UNKNOWN, never an automatically filled source value.
    General arithmetic/capital queries bypass this narrowly scoped checker.
    """
    kind = query_kind(question)
    if kind == "general":
        return {"answer": answer, "decision": "outside_scope", "source_id": None, "verified": False}
    if kind == "unsupported":
        return {"answer": "UNKNOWN", "decision": "unsupported_query", "source_id": None, "verified": False}
    evidence = store.retrieve(question, known_at)
    if len(evidence) != 1 or evidence[0].get("relation") != kind:
        return {"answer": "UNKNOWN", "decision": "no_applicable_typed_source", "source_id": None, "verified": False}
    source = evidence[0]
    if canonical_value(answer) == canonical_value(source["value"]):
        return {"answer": source["value"], "decision": "verified", "source_id": source["id"], "verified": True}
    decision = "model_abstained" if canonical_value(answer) == "unknown" else "answer_mismatch"
    return {"answer": "UNKNOWN", "decision": decision, "source_id": source["id"], "verified": False}
