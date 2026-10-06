"""CPU-only checks of resource-validation errors and report delivery."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from project.scripts.smoke import extract_tag, find_split, validate_row
from project.scripts.summarize_smoke import placeholder_query, render


def example(identifier):
    return {"_id": identifier, "question": "Who founded Example School?", "answer": "Ada",
            "type": "compositional", "context": [["Example School", ["Ada founded Example School."]]],
            "supporting_facts": [["Example School", 0]],
            "evidences": [["Example School", "founder", "Ada"]]}


class ResourceSmokeTests(unittest.TestCase):
    def test_legacy_placeholder_pass_is_flagged_in_summary(self):
        self.assertTrue(placeholder_query(" query "))
        self.assertFalse(placeholder_query("Louis XIV place of death"))
        summary = render({"stage": "models", "status": "passed",
                          "tool_roundtrip": {"query": "query"}})
        self.assertIn("INVALID PLACEHOLDER", summary)

    def test_support_out_of_range_fails(self):
        row = example("a")
        row["supporting_facts"][0][1] = 3
        with self.assertRaisesRegex(ValueError, "sentence index"):
            validate_row(row)

    def test_conflicting_titles_fail(self):
        row = example("a")
        row["context"].append(["Example School", ["Different text"]])
        with self.assertRaisesRegex(ValueError, "Ambiguous title"):
            validate_row(row)

    def test_protocol_rejects_incomplete_or_multiple_actions(self):
        self.assertEqual(extract_tag("<search>school founder</search>", "search"), "school founder")
        for text in ("<search>unfinished", "<search>a</search><search>b</search>"):
            with self.assertRaises(ValueError):
                extract_tag(text, "search")

    def test_duplicate_downloads_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "old").mkdir()
            (root / "train.json").write_text("[]")
            (root / "old/train.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "Expected one"):
                find_split(root, "train")

    def test_reports_and_failed_rerun_replace_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            for split in ("train", "dev"):
                (raw / f"{split}.json").write_text(json.dumps([example(split)]))
            env = {**os.environ, "AGENTIC_RAW_DATA_DIR": str(raw), "AGENTIC_RUNS_DIR": str(root / "runs")}
            def run():
                return subprocess.run([sys.executable, "-m", "project.scripts.smoke", "data"],
                                      env=env, capture_output=True, text=True)
            result = run()
            self.assertEqual(result.returncode, 0, result.stderr)
            out = root / "runs/resource_smoke"
            self.assertEqual(json.loads((out / "data_report.json").read_text())["status"], "passed")
            corpus = json.loads((out / "sample_corpus.jsonl").read_text())
            self.assertEqual(set(corpus), {"id", "title", "text"})
            questions = json.loads((out / "sample_questions.json").read_text())
            self.assertEqual(set(questions[0]), {"id", "question"})
            invalid = example("train")
            invalid["supporting_facts"] = [["Unknown title", 0]]
            (raw / "train.json").write_text(json.dumps([invalid]))
            self.assertNotEqual(run().returncode, 0)
            report = json.loads((out / "data_report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("absent from context", report["errors"][0]["error"])


if __name__ == "__main__":
    unittest.main()
