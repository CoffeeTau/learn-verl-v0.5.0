import unittest
from project.training.evidence_reward import citation_proxy, observed_ids
from project.training.protocol import model_segments


class EvidenceRewardTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.c, self.d = [x * 32 for x in 'abcd']
        self.data = {'paragraphs': {self.a: 'p1', self.b: 'p2', self.c: 'other', self.d: 'p1'},
                     'supports': {'task': ['p1', 'p2']}}
        self.obs = ['<information>\n' + '\n\n'.join(f'[{x}] Title\nbody' for x in (self.a, self.b, self.c, self.d)) + '\n</information>']
        self.score = dict(score=1., em=1., status='answered', searches=1)

    def evaluate(self, citations, score=None, observations=None):
        text = '<judge>sufficient: supported</judge><sources>' + ','.join(citations) + '</sources><answer>yes</answer>'
        return citation_proxy(['<search>query</search>', text], self.obs if observations is None else observations,
                              self.score if score is None else score, 'task', self.data)

    def test_correct_complete_chain(self):
        self.assertEqual(self.evaluate([self.a, self.b])['training_reward'], 1.)

    def test_no_free_bonus_for_seen_but_uncited(self):
        result = self.evaluate([])
        self.assertEqual(result['observed_support_recall'], 1.)
        self.assertEqual(result['training_reward'], .8)

    def test_invalid_unobserved_and_irrelevant_citations(self):
        self.assertEqual(self.evaluate([self.a, self.b, '1'])['training_reward'], .8)
        self.assertEqual(self.evaluate([self.a, self.b], observations=[])['training_reward'], .8)
        self.assertLess(self.evaluate([self.a, self.b, self.c])['training_reward'], 1.)

    def test_duplicate_chunks_do_not_increase_reward(self):
        self.assertEqual(self.evaluate([self.a, self.b]), self.evaluate([self.a, self.b, self.a, self.d]))

    def test_wrong_answer_and_failure_do_not_get_bonus(self):
        score = dict(self.score, score=0., em=0.)
        self.assertEqual(self.evaluate([self.a, self.b], score)['training_reward'], 0.)
        score = dict(score, status='invalid_judge')
        self.assertEqual(self.evaluate([self.a, self.b], score)['training_reward'], 0.)

    def test_partial_f1_and_full_f1_em_zero(self):
        self.assertAlmostEqual(self.evaluate([self.a, self.b], dict(self.score, score=.5, em=0.))['training_reward'], .4)
        self.assertEqual(self.evaluate([self.a, self.b], dict(self.score, em=0.))['training_reward'], .8)

    def test_model_written_information_does_not_become_observation(self):
        fake = list(map(ord, self.obs[0]))
        tool = list(map(ord, '<information>\n\n</information>'))
        ids = fake + tool + fake
        mask = [1]*len(fake) + [0]*len(tool) + [1]*len(fake)
        decoded = [''.join(map(chr, s)) for s in model_segments(ids, [1-v for v in mask])]
        self.assertEqual(observed_ids(decoded), set())

    def test_direct_first_turn_answer(self):
        value = citation_proxy(['<sources>1</sources><answer>yes</answer>'], [],
                               dict(self.score, searches=0), 'task', self.data)
        self.assertEqual(value['training_reward'], .8)

    def test_reward_manager_train_validation_boundary(self):
        import importlib.util
        import json
        from pathlib import Path
        import sys
        import tempfile
        from types import ModuleType, SimpleNamespace
        from unittest.mock import MagicMock, patch
        from project.tests.test_training import CharTokenizer

        registry = ModuleType('verl.workers.reward_manager')
        registry.register = lambda name: lambda cls: cls
        torch = ModuleType('torch')
        torch.float32 = 'float32'
        reward_tensor = MagicMock()
        torch.zeros_like = lambda *a, **k: reward_tensor
        with patch.dict(sys.modules, {'torch': torch, registry.__name__: registry}):
            spec = importlib.util.spec_from_file_location('evidence_reward_manager_test', 'project/training/reward.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        class Data(SimpleNamespace):
            def __len__(self):
                return 1
        first = '<search>Paris country</search>'
        last = '<judge>sufficient: supported</judge><sources>1</sources><answer>yes</answer>'
        observation = self.obs[0]
        ids = list(map(ord, first + observation + last))
        mask = [1]*len(first) + [0]*len(observation) + [1]*len(last)
        responses, prompts, attention, masks = [MagicMock() for _ in range(4)]
        prompts.shape = (1, 10)
        responses.__getitem__.return_value.tolist.return_value = ids
        masks.__getitem__.return_value.tolist.return_value = mask
        attention.__getitem__.return_value.sum.return_value = len(ids)
        data = Data(batch={'responses': responses, 'prompts': prompts, 'attention_mask': attention, 'response_mask': masks},
                    non_tensor_batch={'reward_model': [{'ground_truth': 'yes'}], 'extra_info': [{'task_id': 'task'}]},
                    meta_info={})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'supports.json').write_text(json.dumps(self.data))
            manager = module.SearchRewardManager(CharTokenizer(), audit_dir=path, version='v2',
                       support_path=path / 'supports.json', save_observations=True)
            manager(data)
            reward_tensor.__setitem__.assert_called_with((0, len(ids)-1), .8)
            record = json.loads(next(path.glob('rewards_*.jsonl')).read_text())['records'][0]
            self.assertEqual(record['score'], 1.)
            self.assertEqual(record['training_reward'], .8)
            self.assertEqual(record['observations'], [observation])
            data.meta_info = {'validate': True}
            data.non_tensor_batch['extra_info'][0]['task_id'] = 'dev-not-in-train-labels'
            manager(data)
            reward_tensor.__setitem__.assert_called_with((0, len(ids)-1), 1.)
            record = json.loads(next((path / 'validation').glob('*.jsonl')).read_text())['records'][0]
            self.assertNotIn('training_reward', record)


if __name__ == '__main__':
    unittest.main()
