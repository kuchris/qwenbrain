"""Artifact and recount checks run without model packages or a GPU."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qwenbrain import artifacts
from qwenbrain.reproduction_summary import summarize


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name).resolve()
        self.patcher = patch.object(artifacts, "ROOT", self.root)
        self.patcher.start()
        self.out = artifacts.new_run("test", "runs/bundle")

    def tearDown(self):
        self.patcher.stop()
        self.directory.cleanup()

    def write(self, name, report):
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def fixture(self):
        proof = {"status": "completed", "base_sha256_before": "base", "base_sha256_after": "base", "live_checkpoint_unchanged": True}
        row = {"answer": "Cobalt", "expected": "Cobalt", "exact_match": False}
        acquisition = {**proof, "selected_arm": "qa", "runs": [{"phase": "confirmation", "arm": "qa", "seed": 42,
                       "checkpoints": [{"exposures_per_fact": 64, "answers": [row]}]}]}
        acq = self.write("acquisition/results.json", acquisition)
        checkpoint = acq.parent / "confirmation-qa-42.safetensors"
        checkpoint.write_bytes(b"synthetic checkpoint bytes for checksum tests, not model weights")
        acquisition["checkpoint_sha256"] = {checkpoint.name: artifacts.file_hash(checkpoint)}
        self.write("acquisition/results.json", acquisition)
        completion = {"status": "completed", "inputs": {"acquisition": artifacts.input_record(acq, self.out)},
                      "runs": [{"phase": "confirmation", "arm": "qa", "answers": [{"completion": " Cobalt.", "value": "Cobalt", "first_clause_correct": False}]}]}
        comp = self.write("completions.json", completion)
        retention = {**proof, "inputs": {"acquisition": artifacts.input_record(acq, self.out / "retention"),
                     "initial_checkpoint": artifacts.input_record(checkpoint, self.out / "retention")},
                     "runs": [{"arm": "replay_50_percent", "checkpoints": [{"steps": 16, "old": {"answers": [row]}, "new": {"answers": [row]}}]}]}
        ret = self.write("retention/results.json", retention)
        fact = {**row, "group": "old", "gold_source_id": 0, "evidence": [{"id": 0}], "ended_with_end_token": True}
        control = {"group": "control", "answer": "wrong", "expected": "Paris", "exact_match": True}
        memory = {**proof, "model": {"model_id": "fixture"}, "inputs": {"retention": artifacts.input_record(ret, self.out / "memory"),
                  "initial_checkpoint": artifacts.input_record(checkpoint, self.out / "memory")},
                  "arms": {arm: {"answers": [fact, control], "fact_correct": 999} for arm in ("F", "O", "R", "W")}}
        mem = self.write("memory/results.json", memory)
        manifest = {"status": "completed", "stages": [{"name": name, "result": artifacts.input_record(path, self.out)}
                    for name, path in (("acquisition", acq), ("completions", comp), ("retention", ret), ("memory", mem))]}
        self.write("pipeline.json", manifest)
        return acquisition, retention, manifest

    def test_missing_incomplete_and_output_reuse_fail_without_overwrite(self):
        with self.assertRaisesRegex(ValueError, "Missing input"):
            artifacts.completed_report("missing.json")
        path = self.write("partial.json", {"status": "running"})
        with self.assertRaisesRegex(ValueError, "not completed"):
            artifacts.completed_report(path)
        with self.assertRaises(FileExistsError):
            artifacts.new_run("test", "runs/bundle")
        for path in ("outside", "runs/../outside", "runs"):
            with self.assertRaises(ValueError):
                artifacts.new_run("test", path)

    def test_checkpoint_is_explicit_hashed_and_relocatable(self):
        acquisition, retention, _ = self.fixture()
        acq = self.out / "acquisition/results.json"
        checkpoint = artifacts.acquisition_checkpoint(acq, acquisition)
        self.assertEqual(checkpoint, artifacts.retention_checkpoint(self.out / "retention/results.json", retention))
        moved = self.root / "runs/relocated"
        self.assertTrue(self.out.is_relative_to(self.root) and moved.is_relative_to(self.root))
        self.out.rename(moved)
        self.out = moved
        acq = moved / "acquisition/results.json"
        checkpoint = artifacts.acquisition_checkpoint(acq, acquisition)
        self.assertEqual(checkpoint, artifacts.retention_checkpoint(moved / "retention/results.json", retention))
        checkpoint.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            artifacts.acquisition_checkpoint(acq, acquisition)
        del acquisition["checkpoint_sha256"]
        with self.assertRaisesRegex(ValueError, "lacks checkpoint hashes"):
            artifacts.acquisition_checkpoint(acq, acquisition)

    def test_retention_requires_the_declared_qa64_checkpoint(self):
        acquisition, _, _ = self.fixture()
        acquisition["runs"][0]["checkpoints"][0]["exposures_per_fact"] = 16
        with self.assertRaisesRegex(ValueError, "64-exposure"):
            artifacts.acquisition_checkpoint(self.out / "acquisition/results.json", acquisition)

    def test_summary_recounts_text_instead_of_trusting_cached_scores(self):
        self.fixture()
        before = artifacts.file_hash(self.out / "memory/results.json")
        result = summarize(self.out)
        self.assertEqual(result["acquisition"][0]["correct"], 1)
        self.assertEqual(result["completions"][0]["correct"], 1)
        self.assertEqual(result["memory"]["R"]["facts"], {"correct": 1, "total": 1})
        self.assertEqual(result["memory"]["R"]["control"], {"correct": 0, "total": 1})
        self.assertEqual(result["memory"]["R"]["correct_source_selections"], 1)
        self.assertEqual(artifacts.file_hash(self.out / "memory/results.json"), before)

    def test_partial_or_tampered_pipeline_cannot_be_summarized(self):
        _, _, manifest = self.fixture()
        manifest["status"] = "failed"
        self.write("pipeline.json", manifest)
        with self.assertRaisesRegex(ValueError, "not completed"):
            summarize(self.out)
        manifest["status"] = "completed"
        self.write("pipeline.json", manifest)
        self.write("memory/results.json", {"status": "completed"})
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            summarize(self.out)

    def test_mixed_reports_are_rejected_even_with_valid_individual_hashes(self):
        _, _, manifest = self.fixture()
        other = self.write("other/results.json", {"status": "completed"})
        path = self.out / "completions.json"
        report = json.loads(path.read_text())
        report["inputs"]["acquisition"] = artifacts.input_record(other, self.out)
        self.write("completions.json", report)
        manifest["stages"][1]["result"] = artifacts.input_record(path, self.out)
        self.write("pipeline.json", manifest)
        with self.assertRaisesRegex(ValueError, "different acquisition"):
            summarize(self.out)


if __name__ == "__main__":
    unittest.main()
