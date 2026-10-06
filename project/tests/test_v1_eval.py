"""Matched-evaluation boundaries and export reuse; no GPU dependencies."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from project.agent.search import SYSTEM_PROMPT
from project.evaluation.v0 import validate_baseline
from project.evaluation.v1 import export_model


class V1EvalTests(unittest.TestCase):
    def test_changed_labels_are_rejected(self):
        dataset = {'files': {'dev.jsonl': 'questions', 'dev.labels.jsonl': 'labels'}, 'corpus_sha256': 'corpus'}
        manifest = {'system_prompt': SYSTEM_PROMPT, 'enable_thinking': False,
                    'scoring': 'canonical_answer_em_f1_no_aliases', 'dataset_manifest': copy.deepcopy(dataset)}
        report = {'metrics': {'completed': 200, 'planned': 200}}
        validate_baseline(manifest, report, dataset, 200)
        dataset['files']['dev.labels.jsonl'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Development input'):
            validate_baseline(manifest, report, dataset, 200)

    def test_export_reuse_checks_source_and_export(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            checkpoint = run / 'checkpoints/global_step_125/actor'
            (checkpoint / 'huggingface').mkdir(parents=True)
            (checkpoint / 'fsdp_config.json').write_text('{"world_size": 1}')
            (checkpoint / 'huggingface/config.json').write_text('{}')
            shard = checkpoint / 'model_world_size_1_rank_0.pt'
            shard.write_bytes(b'fixture')
            def merge(command, **kwargs):
                target = Path(command[command.index('--target_dir') + 1])
                target.mkdir()
                (target / 'config.json').write_text('{}')
                (target / 'model.safetensors').write_bytes(b'fixture')
                return SimpleNamespace(returncode=0)
            with patch('project.evaluation.v1.subprocess.run', side_effect=merge) as process:
                target = export_model(run, 125)
                self.assertEqual(target, export_model(run, 125))
                self.assertEqual(process.call_count, 1)
                shard.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'differs'):
                    export_model(run, 125)


if __name__ == '__main__':
    unittest.main()
