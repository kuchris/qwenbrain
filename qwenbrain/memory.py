"""Small source store for the synthetic person/project retrieval experiment."""

import re
import sqlite3
from datetime import date


def words(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


class MemoryStore:
    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS memories "
            "(id INTEGER PRIMARY KEY, subject_key TEXT NOT NULL, content TEXT NOT NULL)"
        )

    def append(self, records):
        with self.connection:
            self.connection.executemany(
                "INSERT INTO memories VALUES (?, ?, ?)",
                [(r["id"], words(r["subject"]), r["content"]) for r in records],
            )

    def retrieve(self, question):
        """Match a complete stored name using only the query, never an answer/id.

        This deliberately tests explicit-name lookup, not semantic retrieval.
        Ambiguous queries return no evidence rather than silently choosing a name.
        """
        rows = self.connection.execute(
            "SELECT id, content FROM memories "
            "WHERE instr(?, ' ' || subject_key || ' ') > 0 ORDER BY id",
            (" " + words(question) + " ",),
        ).fetchall()
        return [dict(rows[0])] if len(rows) == 1 else []

    def close(self):
        self.connection.close()


class VersionedMemoryStore:
    """Append-only project claims with effective and observation dates.

    Explicit ISO dates in queries are supported; relative dates and aliases are
    outside this pilot. Later observations supersede claims for the same person
    and effective date. A late historical correction does not replace a newer
    effective state.
    """

    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS versions "
            "(id INTEGER PRIMARY KEY, subject_key TEXT NOT NULL, content TEXT NOT NULL, "
            "valid_from TEXT NOT NULL, recorded_at TEXT NOT NULL)"
        )

    def append(self, records):
        rows = []
        for record in records:
            for key in ("valid_from", "recorded_at"):
                if date.fromisoformat(record[key]).isoformat() != record[key]:
                    raise ValueError("Dates must use YYYY-MM-DD.")
            rows.append((record["id"], words(record["subject"]), record["content"],
                         record["valid_from"], record["recorded_at"]))
        with self.connection:
            self.connection.executemany("INSERT INTO versions VALUES (?, ?, ?, ?, ?)", rows)

    def retrieve(self, question, known_at):
        date.fromisoformat(known_at)
        dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", question)
        if len(dates) > 1:
            return []
        as_of = dates[0] if dates else known_at
        try:
            date.fromisoformat(as_of)
        except ValueError:
            return []
        if as_of > known_at:
            return []  # This store does not predict future states.
        # Resolve identity before the effective-date filter: two named people
        # remain ambiguous even if only one has a record for the requested date.
        subjects = self.connection.execute(
            "SELECT DISTINCT subject_key FROM versions WHERE recorded_at <= ? "
            "AND instr(?, ' ' || subject_key || ' ') > 0",
            (known_at, " " + words(question) + " "),
        ).fetchall()
        if len(subjects) != 1:
            return []
        row = self.connection.execute(
            "SELECT id, content, valid_from, recorded_at FROM versions "
            "WHERE subject_key = ? AND recorded_at <= ? AND valid_from <= ? "
            "ORDER BY valid_from DESC, recorded_at DESC, id DESC LIMIT 1",
            (subjects[0]["subject_key"], known_at, as_of),
        ).fetchone()
        return [dict(row)] if row else []

    def close(self):
        self.connection.close()


class ProjectMemoryStore(VersionedMemoryStore):
    """Versioned source text plus a trusted, typed project-name field.

    The caller supplies the field when encoding a source; this store does not
    infer its truth from text. Existing untyped records remain unverifiable.
    """

    def __init__(self, path):
        super().__init__(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS project_values "
            "(source_id INTEGER PRIMARY KEY, value TEXT NOT NULL)"
        )

    def append(self, records):
        versions, values = [], []
        for record in records:
            value = record["value"]
            if not isinstance(value, str) or not value.strip() or value.strip().casefold() == "unknown":
                raise ValueError("Project name must be nonempty and cannot be the reserved UNKNOWN response.")
            for key in ("valid_from", "recorded_at"):
                if date.fromisoformat(record[key]).isoformat() != record[key]:
                    raise ValueError("Dates must use YYYY-MM-DD.")
            versions.append((record["id"], words(record["subject"]), record["content"],
                             record["valid_from"], record["recorded_at"]))
            values.append((record["id"], value))
        with self.connection:
            self.connection.executemany("INSERT INTO versions VALUES (?, ?, ?, ?, ?)", versions)
            self.connection.executemany("INSERT INTO project_values VALUES (?, ?)", values)

    def retrieve(self, question, known_at):
        evidence = super().retrieve(question, known_at)
        if evidence:
            row = self.connection.execute(
                "SELECT value FROM project_values WHERE source_id = ?", (evidence[0]["id"],)
            ).fetchone()
            if row:
                evidence[0].update(relation="project_name", value=row["value"])
        return evidence

    def lookup(self, subject, as_of, known_at):
        """Look up an explicit identity and dates, without scanning a question."""
        for value in (as_of, known_at):
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError("Dates must use YYYY-MM-DD.")
        if as_of > known_at:
            return None
        row = self.connection.execute(
            "SELECT v.id, v.content, v.valid_from, v.recorded_at, p.value "
            "FROM versions v LEFT JOIN project_values p ON p.source_id = v.id "
            "WHERE v.subject_key = ? AND v.valid_from <= ? AND v.recorded_at <= ? "
            "ORDER BY v.valid_from DESC, v.recorded_at DESC, v.id DESC LIMIT 1",
            (words(subject), as_of, known_at),
        ).fetchone()
        return dict(row) if row else None
