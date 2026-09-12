"""CPU checks for source persistence and retrieval without gold-answer access."""

import tempfile
import unittest
from pathlib import Path

from qwenbrain.memory import MemoryStore


class MemoryTests(unittest.TestCase):
    def test_persistence_boundaries_and_ambiguous_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite"
            store = MemoryStore(path)
            store.append([
                {"id": 1, "subject": "Kian Vale", "content": "Kian Vale's project is named Quartz."},
                {"id": 2, "subject": "Mara Vale", "content": "Mara Vale's project is named Cedar."},
            ])
            store.close()
            store = MemoryStore(path)
            for question in ("Which project belongs to Kian Vale?", "KIAN VALE's project?", "Kian\nVale?"):
                self.assertEqual(store.retrieve(question), [
                    {"id": 1, "content": "Kian Vale's project is named Quartz."}
                ])
            for question in ("Vale?", "Kian Voss?", "Kian Valerian?", "What is the capital of France?",
                             "Kian Vale and Mara Vale?", "' OR 1=1 --"):
                self.assertEqual(store.retrieve(question), [])
            store.close()

    def test_new_facts_are_absent_until_written(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / "memory.sqlite")
            question = "What project does Rhea Pike work on?"
            self.assertEqual(store.retrieve(question), [])
            store.append([{"id": 9, "subject": "Rhea Pike", "content": "Rhea Pike's project is named Silver."}])
            self.assertEqual(store.retrieve(question)[0]["id"], 9)
            store.close()


if __name__ == "__main__":
    unittest.main()
