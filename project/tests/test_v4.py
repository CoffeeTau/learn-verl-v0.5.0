"""V4 pairing, selection, provenance and split routing; no GPU downloads."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from project.common import save_json, save_jsonl, sha256, model_inventory, signature
from project.training.v4 import prepare_validation, validate_retriever
from project.training.v4_monitor import summarize_validation, select_checkpoint, token_cost


def rows():
    return [dict(task_id=task, eval_group=group, score=score, em=score, total_tokens=100,
                 searches=2, status=status) for task, group, score, status in
            [('a', 'natural', 1., 'answered'), ('b', 'natural', 0., 'judge_action_mismatch'),
             ('b', 'hard', 0., 'answered')]]


class V4Tests(unittest.TestCase):
    def test_pairing_keeps_failures_and_rejects_incomplete_or_duplicate(self):
        expected = {'natural': ['a', 'b'], 'hard': ['b']}
        groups = summarize_validation(rows(), expected)
        self.assertEqual(groups['natural']['em'], .5)
        self.assertEqual(groups['natural']['protocol_failures'], 1)
        self.assertEqual(groups['natural_matched']['em'], 0.)
        for bad in (rows()[:-1], rows() + [rows()[0]]):
            with self.assertRaises(ValueError):
                summarize_validation(bad, expected)

    def test_selection_guards_and_step0_fallback(self):
        def record(step, em, f1, failures, hard, tokens=100):
            return dict(step=step, groups={'natural': dict(em=em, f1=f1, protocol_failures=failures, tokens=tokens),
                                           'hard': dict(em=hard)})
        history = [record(0, .6, .7, 4, .6), record(25, .8, .9, 5, .8), record(50, .8, .9, 1, .5)]
        self.assertEqual(select_checkpoint(history)['selected_step'], 0)
        history += [record(75, .65, .75, 3, .6), record(100, .65, .75, 2, .6, 90)]
        self.assertEqual(select_checkpoint(history)['selected_step'], 100)
        history += [record(125, .65, .75, 2, .6, 90)]
        self.assertEqual(select_checkpoint(history)['selected_step'], 100)

    def test_token_cost_counts_repeated_prefix_not_padding(self):
        self.assertEqual(token_cost(10, [1, 1, 0, 0, 0, 1]), (10 + 2) + (15 + 1))

    def test_validation_hook_writes_selection_and_rejects_duplicate_step(self):
        upstream = ModuleType('verl.trainer.ppo.ray_trainer')
        class Base:
            def _validate(self):
                return {'upstream/metric': 1.}
        upstream.RayPPOTrainer = Base
        with patch.dict(sys.modules, {upstream.__name__: upstream}):
            spec = importlib.util.spec_from_file_location('v4_trainer_test', 'project/training/v4_trainer.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'validation').mkdir()
            save_json(root / 'validation_tasks.json', {'natural': ['a', 'b'], 'hard': ['b']})
            save_jsonl(root / 'validation/0.jsonl', rows())
            trainer = module.V4Trainer()
            trainer.config = SimpleNamespace(trainer=SimpleNamespace(
                default_local_dir=str(root / 'checkpoints'), validation_data_dir=str(root / 'validation')))
            trainer.global_steps = 0
            metrics = trainer._validate()
            self.assertEqual(metrics['dev/natural/em'], .5)
            self.assertEqual(metrics['dev/natural_matched/em'], 0.)
            self.assertEqual(json.loads((root / 'selection.json').read_text())['selected_step'], 0)
            with self.assertRaises(ValueError):
                trainer._validate()

    def test_dev_views_use_same_ids_and_private_plan(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tasks = [dict(id=str(i), question=f'q{i}') for i in range(200)]
            save_jsonl(root / 'dev.jsonl', tasks)
            save_jsonl(root / 'dev.labels.jsonl', [dict(id=t['id'], answer='secret', supporting_facts=[
                {'passage_id': 'a'}, {'passage_id': 'b'}]) for t in tasks])
            save_jsonl(root / 'corpus.jsonl', [dict(id='a', paragraph_id='pa'), dict(id='b', paragraph_id='pb')])
            records = [dict(prompt=[dict(role='user', content=t['question'])], data_source='old', agent_name='eval',
                            reward_model={'ground_truth': 'secret'}, extra_info={'task_id': t['id'], 'split': 'dev'}) for t in tasks]
            with patch.dict(os.environ, AGENTIC_PROCESSED_DATA_DIR=folder, AGENTIC_CORPUS_DIR=folder):
                for smoke, natural, hard in ((False, 200, 50), (True, 8, 2)):
                    pq.write_table(pa.Table.from_pylist(records), root / 'dev.parquet')
                    expected = prepare_validation(root, root, smoke)
                    actual = pq.read_table(root / 'dev.parquet').to_pylist()
                    self.assertEqual((len(expected['natural']), len(expected['hard'])), (natural, hard))
                    self.assertEqual(len(actual), natural + hard)
                    self.assertTrue(set(expected['hard']) <= set(expected['natural']))
                    for row in actual:
                        self.assertNotIn('secret', json.dumps(row['prompt']))
                        is_hard = row['extra_info']['eval_group'] == 'hard'
                        self.assertEqual(row['agent_name'], 'agentic_correction_dev_hard' if is_hard else 'agentic_correction_eval')

    def test_actual_retriever_routes_dev_separately_from_train(self):
        ray, local = ModuleType('ray'), ModuleType('project.retrieval.local')
        ray.remote = lambda **kwargs: lambda cls: cls
        local.LocalRetriever = object
        with patch.dict(sys.modules, {'ray': ray, local.__name__: local}):
            spec = importlib.util.spec_from_file_location('v4_retriever_test', 'project/training/retriever.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        actor = object.__new__(module.RetrievalActor)
        hits = [dict(id='a'), dict(id='b')]
        actor.retriever = SimpleNamespace(search=lambda *args: hits)
        actor.plan = {'train': dict(mode='withhold_one', exclude_paragraphs=['pa'])}
        actor.dev_plan = {'dev': dict(mode='withhold_one', exclude_paragraphs=['pb'])}
        actor.paragraphs, actor.audit_path = {'a': 'pa', 'b': 'pb'}, None
        self.assertEqual(actor.search_v2('dev', 'q', 3, 0, 'dev'), hits[:1])
        self.assertEqual(actor.search_v2('dev', 'q', 3, 1, 'dev'), hits)
        self.assertEqual(actor.search_v2('dev', 'q', 3, 0, True), hits)
        self.assertEqual(actor.search_v2('train', 'q', 3, 0, 'dev'), hits)
        self.assertEqual(actor.search_v2('train', 'q', 3, 0, True), hits[1:])

    def test_retriever_provenance_rejects_stale_embedding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, corpus, run = root / 'source', root / 'corpus', root / 'run'
            for p in (source, corpus, run / 'model', run / 'index'):
                p.mkdir(parents=True)
            (corpus / 'corpus.jsonl').write_text('{}\n')
            digest = sha256(corpus / 'corpus.jsonl')
            save_json(source / 'manifest.json', {'corpus_sha256': digest})
            save_json(run / 'model/config.json', {})
            inventory = model_inventory(run / 'model')
            save_json(run / 'report.json', {'status': 'passed', 'model_inventory': inventory})
            save_json(run / 'manifest.json', {'pairs_manifest': {'source_manifest_sha256': sha256(source / 'manifest.json')}})
            (run / 'index/embeddings.npy').write_bytes(b'original')
            save_json(run / 'index/manifest.json', {'status': 'passed', 'model_inventory_signature': signature(inventory),
                      'corpus_sha256': digest, 'embeddings_sha256': sha256(run / 'index/embeddings.npy')})
            validate_retriever(run, source, corpus)
            (run / 'index/embeddings.npy').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                validate_retriever(run, source, corpus)


if __name__ == '__main__':
    unittest.main()
