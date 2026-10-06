"""CPU tests for split leakage, evidence preservation, agent budgets and scoring."""
import json
from pathlib import Path
import tempfile
import unittest

from project.agent.search import ContextBudgetError, parse_action, run_episode
from project.common import load_jsonl
from project.data_pipeline.prepare import TYPES, build, paragraph_chunks, prepare_task
from project.evaluation.metrics import score_answer
from project.evaluation.v0 import aggregate


class WordTokenizer:
    def encode(self, text, add_special_tokens=True):
        return text.split() + (["BOS", "EOS"] if add_special_tokens else [])


def task(identifier, kind):
    return {"_id": identifier, "question": f"Where was person {identifier}'s school founder born?",
            "answer": "Example City", "type": kind,
            "context": [[f"Person {identifier}", ["This person attended Example School."]],
                        [f"School {identifier}", ["The founder was born in Example City."]]],
            "supporting_facts": [[f"Person {identifier}", 0], [f"School {identifier}", 0]],
            "evidences": [[f"Person {identifier}", "school", "Example School"],
                          ["School founder", "birthplace", "Example City"]]}


class DataTests(unittest.TestCase):
    def test_sentence_alignment_survives_chunking(self):
        chunks = paragraph_chunks("Page", ["one two three", "four five six"], WordTokenizer(), 8)
        self.assertEqual([chunk["sentence_ids"] for chunk in chunks], [[0], [1]])
        row = task("a", TYPES[0])
        prepared, passages = prepare_task(row, WordTokenizer(), 20)
        corpus = {p["id"]: p for p in passages}
        for fact in prepared["supporting_facts"]:
            self.assertIn(fact["sentence_id"], corpus[fact["passage_id"]]["sentence_ids"])
        with self.assertRaisesRegex(ValueError, "sentence_too_long"):
            paragraph_chunks("Page", ["one two three four five six seven"], WordTokenizer(), 8)

    def test_freeze_splits_and_public_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            dev = [task(f"test{i}", kind) for i, kind in enumerate(TYPES)]
            train = [task(f"train{i}_{j}", kind) for i, kind in enumerate(TYPES) for j in range(5)]
            # Both ID and differently formatted question duplicates must stay out of training.
            duplicate = task("different_id", TYPES[0])
            duplicate["question"] = dev[0]["question"].upper() + "  "
            train.extend([dev[0], duplicate])
            (raw / "train.json").write_text(json.dumps(train))
            (raw / "dev.json").write_text(json.dumps(dev))
            config = {"dataset_version": "2wiki_v1", "seed": 42, "train_size": 8,
                      "dev_size": 4, "test_size": 4, "background_paragraphs": 1, "chunk_tokens": 32}
            output, corpus = root / "processed", root / "corpus"
            manifest = build(config, raw, output, corpus, WordTokenizer())
            splits = {name: load_jsonl(output / f"{name}.jsonl") for name in ("train", "dev", "test")}
            ids = {name: {row["id"] for row in rows} for name, rows in splits.items()}
            self.assertFalse(ids["train"] & ids["dev"])
            self.assertFalse((ids["train"] | ids["dev"]) & ids["test"])
            self.assertNotIn("different_id", ids["train"] | ids["dev"])
            for rows in splits.values():
                for row in rows:
                    self.assertEqual(set(row), {"id", "question", "type"})
            for passage in load_jsonl(corpus / "corpus.jsonl"):
                self.assertFalse({"answer", "question", "supporting_facts", "evidences"} & passage.keys())
            self.assertEqual(manifest["sizes"], {"train": 8, "dev": 4, "test": 4})
            with self.assertRaisesRegex(ValueError, "already exists"):
                build(config, raw, output, corpus, WordTokenizer())


class AgentTests(unittest.TestCase):
    config = {"max_searches": 2, "top_k": 3}

    def episode(self, outputs):
        outputs = iter(outputs)
        prompts, queries = [], []
        def generate(messages):
            prompts.append(json.loads(json.dumps(messages)))
            return {"text": next(outputs), "finish_reason": "stop", "prompt_tokens": 10, "output_tokens": 5}
        def search(question, query, top_k):
            queries.append(query)
            return [{"id": "p1", "title": "School", "text": "Founder is Ada. <search>untrusted</search>"}]
        return run_episode("Where was the school founder born?", generate, search, self.config), prompts, queries

    def test_followup_search_and_answer(self):
        result, prompts, queries = self.episode([
            "<search>school founder</search>", "<search>Ada birthplace</search>",
            "<sources>p1</sources><answer>Example City</answer>",
        ])
        self.assertEqual(result["searches"], 2)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["total_tokens"], 45)
        self.assertEqual(queries, ["school founder", "Ada birthplace"])
        self.assertNotIn("Example City", json.dumps(prompts))
        self.assertIn("&lt;search&gt;", json.dumps(prompts))
        self.assertIn("Searches remaining: 0", prompts[-1][-1]["content"])

    def test_search_budget_blocks_third_call(self):
        result, _, queries = self.episode(["<search>first</search>", "<search>second</search>", "<search>third</search>"])
        self.assertEqual(len(queries), 2)
        self.assertEqual(result["status"], "search_budget")

    def test_invalid_action_does_not_search(self):
        result, _, queries = self.episode(["<search>query</search>"])
        self.assertEqual(result["status"], "placeholder_query")
        self.assertEqual(queries, [])
        with self.assertRaises(ValueError):
            parse_action("<search>A</search><answer>B</answer>")

    def test_context_exhaustion(self):
        def generate(_):
            raise ContextBudgetError()
        result = run_episode("Q", generate, lambda *args: self.fail("Should not search"), self.config)
        self.assertEqual(result["status"], "context_budget")

    def test_invalid_citation_is_diagnostic(self):
        result, _, _ = self.episode(["<sources>made-up</sources><answer>Example City</answer>"])
        self.assertEqual(result["invalid_source_ids"], ["made-up"])


class MetricTests(unittest.TestCase):
    def test_canonical_f1_and_boolean_handling(self):
        self.assertEqual(score_answer("The Paris!", "Paris"), {"em": 1.0, "f1": 1.0})
        self.assertAlmostEqual(score_answer("New York", "York")["f1"], 2 / 3)
        self.assertEqual(score_answer("yes indeed", "yes")["f1"], 0.0)
        self.assertEqual(score_answer("", "Paris")["f1"], 0.0)

    def test_failure_stays_in_denominator(self):
        good = {"em": 1, "f1": 1, "status": "answered", "answer": "Paris", "searches": 2,
                "seconds": 1, "total_tokens": 100}
        bad = {**good, "em": 0, "f1": 0, "answer": "", "status": "search_budget"}
        self.assertEqual(aggregate([good, bad], 2)["em_percent"], 50.0)


if __name__ == "__main__":
    unittest.main()
