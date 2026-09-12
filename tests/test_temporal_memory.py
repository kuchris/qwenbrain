import tempfile
import unittest
from pathlib import Path

from qwenbrain.memory import VersionedMemoryStore


class TemporalMemoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "temporal.sqlite"
        self.store = VersionedMemoryStore(self.path)
        self.store.append([
            {"id": 1, "subject": "Aster Rook", "content": "Old", "valid_from": "2026-01-01", "recorded_at": "2026-01-02"},
            {"id": 2, "subject": "Aster Rook", "content": "New", "valid_from": "2026-02-01", "recorded_at": "2026-02-02"},
            {"id": 3, "subject": "Aster Rook", "content": "Corrected old", "valid_from": "2026-01-01", "recorded_at": "2026-02-10"},
            {"id": 4, "subject": "Belen Frost", "content": "Other", "valid_from": "2026-02-01", "recorded_at": "2026-02-02"},
        ])

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_current_and_late_historical_correction(self):
        self.assertEqual(self.store.retrieve("Aster Rook now?", "2026-02-10")[0]["id"], 2)
        self.assertEqual(self.store.retrieve("Aster Rook on 2026-01-15?", "2026-02-10")[0]["id"], 3)
        self.assertEqual(self.store.retrieve("Aster Rook on 2026-01-15?", "2026-02-02")[0]["id"], 1)

    def test_no_future_observation_leakage(self):
        self.assertEqual(self.store.retrieve("Aster Rook now?", "2026-01-02")[0]["id"], 1)
        self.assertEqual(self.store.retrieve("Aster Rook now?", "2026-01-01"), [])
        self.assertEqual(self.store.retrieve("Aster Rook on 2026-02-01?", "2026-01-02"), [])

    def test_same_effective_date_correction_and_restart(self):
        self.store.append([{"id": 5, "subject": "Aster Rook", "content": "Corrected new",
                            "valid_from": "2026-02-01", "recorded_at": "2026-02-12"}])
        self.store.close()
        self.store = VersionedMemoryStore(self.path)
        self.assertEqual(self.store.retrieve("Aster Rook?", "2026-02-11")[0]["id"], 2)
        self.assertEqual(self.store.retrieve("Aster Rook?", "2026-02-12")[0]["id"], 5)

    def test_missing_invalid_and_ambiguous_queries(self):
        for question in ("Aster Rook before 2025-12-01?", "Aster Rook on 2026-02-30?",
                         "Aster Rook on 2026-01-01 and 2026-02-01?", "Aster Rookie?",
                         "Aster Frost?", "Aster Rook and Belen Frost on 2026-01-15?"):
            self.assertEqual(self.store.retrieve(question, "2026-02-10"), [])

    def test_invalid_record_batch_is_not_partially_written(self):
        with self.assertRaises(ValueError):
            self.store.append([
                {"id": 5, "subject": "New Person", "content": "Valid", "valid_from": "2026-01-01", "recorded_at": "2026-01-02"},
                {"id": 6, "subject": "New Person", "content": "Invalid", "valid_from": "2026-02-30", "recorded_at": "2026-02-01"},
            ])
        self.assertEqual(self.store.retrieve("New Person?", "2026-02-10"), [])


if __name__ == "__main__":
    unittest.main()
