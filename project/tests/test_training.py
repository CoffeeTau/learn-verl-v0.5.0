"""Focused CPU checks for training's token/mask/reward boundary and reporting."""
import asyncio
import json
from collections import deque
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

from project.training.protocol import continuation_ids, model_segments, trajectory_score
from project.training.summarize import summarize
from project.agent.search import SYSTEM_PROMPT


class CharTokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))

    def convert_tokens_to_ids(self, text):
        return 999999

    def apply_chat_template(self, messages, **kwargs):
        return self.encode('initial prompt')


class TrainingTests(unittest.TestCase):
    def test_reward_audit_json_roundtrip(self):
        # Exercise the actual serialization boundary without loading GPU libraries.
        registry = ModuleType('verl.workers.reward_manager')
        registry.register = lambda name: lambda cls: cls
        with patch.dict(sys.modules, {'torch': ModuleType('torch'), registry.__name__: registry}):
            spec = importlib.util.spec_from_file_location('reward_under_test', Path('project/training/reward.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        records = [{'score': 1.0, 'status': 'answered', 'model_tokens': 12,
                    'tool_template_tokens': 100, 'model_outputs': ['<answer>yes</answer>']}]
        for groups, expected in [({}, 0), ({'a': [0., 0.]}, 0),
                                 ({'a': [0., 1.], 'b': [0.5, 0.5], 'c': [1.]}, 1)]:
            decoded = json.loads(module.reward_audit_json(records, groups))
            self.assertIs(type(decoded['varying_groups']), int)
            self.assertEqual(decoded['varying_groups'], expected)
            self.assertEqual(decoded['records'], records)

    def test_tools_cannot_supply_answer(self):
        tokenizer = CharTokenizer()
        search = tokenizer.encode('<search>Paris country</search>')
        tool = tokenizer.encode('<answer>France</answer>')
        output = tokenizer.encode('<answer>Germany</answer>')
        segments = model_segments(search + tool + output, [1]*len(search) + [0]*len(tool) + [1]*len(output))
        score = trajectory_score([tokenizer.decode(x) for x in segments], 'France')
        self.assertEqual(score['score'], 0)
        self.assertEqual(len(segments), 2)

    def test_strict_reward_budget_and_intermediate_format(self):
        self.assertEqual(trajectory_score(['explanation <answer>yes</answer>'], 'yes')['score'], 0)
        self.assertEqual(trajectory_score(['<search>Paris country</search>']*5 + ['<answer>yes</answer>'], 'yes')['score'], 0)
        self.assertEqual(trajectory_score(['<answer>yes</answer>', '<answer>yes</answer>'], 'yes')['score'], 0)
        self.assertEqual(trajectory_score(['<search>Paris country</search>', '<answer>France</answer>'], 'France')['score'], 1)
        self.assertEqual(trajectory_score(['<search>Paris country</search>'], 'France')['score'], 0)

    def test_continuation_does_not_repeat_eos(self):
        tokenizer = CharTokenizer()
        regular = tokenizer.decode(continuation_ids(tokenizer, 'evidence', [1]))
        eos = tokenizer.decode(continuation_ids(tokenizer, 'evidence', [999999]))
        self.assertTrue(regular.startswith('<|im_end|>\n'))
        self.assertTrue(eos.startswith('\n<|im_start|>user'))
        self.assertTrue(eos.endswith('<think>\n\n</think>\n\n'))

    def test_actual_adapter_preserves_sampled_prefix_and_masks(self):
        # Stub only external framework/Ray types; run the actual adapter logic.
        framework = ModuleType('verl.experimental.agent_loop.agent_loop')
        framework.AgentLoopBase = object
        framework.AgentLoopMetrics = lambda: SimpleNamespace(generate_sequences=0., tool_calls=0.)
        framework.AgentLoopOutput = lambda **kwargs: SimpleNamespace(**kwargs)
        ray = ModuleType('ray')
        with patch.dict(sys.modules, {'ray': ray, framework.__name__: framework}):
            spec = importlib.util.spec_from_file_location('adapter_under_test', Path('project/training/agent_loop.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        tokenizer = CharTokenizer()
        outputs = deque([tokenizer.encode('<search>Paris country</search>'), tokenizer.encode('<answer>France</answer>')])
        generated = list(outputs)
        calls = []
        async def generate(request_id, prompt_ids, sampling_params):
            calls.append(list(prompt_ids))
            return outputs.popleft()
        async def search(question, query, k):
            return [{'id': 'p1', 'title': 'Paris', 'text': 'Paris is in France.'}]
        agent = module.SearchAgentLoop()
        agent.settings = {'max_searches': 4, 'max_new_tokens': 512, 'max_context_tokens': 8192, 'top_k': 3}
        agent.tokenizer = tokenizer
        agent.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(
            prompt_length=512, response_length=8192, top_k=20)))
        agent.server_manager = SimpleNamespace(generate=generate)
        agent.retriever = SimpleNamespace(search=SimpleNamespace(remote=search))
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT.format(max_searches=4)}, {'role': 'user', 'content': 'Where?'}]
        result = asyncio.run(agent.run(messages, {'temperature': .7}))
        self.assertEqual(model_segments(result.response_ids, result.response_mask), generated)
        self.assertEqual(calls[1], result.prompt_ids + result.response_ids[:-len(generated[-1])])
        self.assertGreater(result.response_mask.count(0), 0)
        self.assertEqual(result.response_mask[-1], 1)

    def test_summary_accepts_completed_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'smoke_example'
            (root / 'reward_audit').mkdir(parents=True)
            checkpoint = root / 'checkpoints/global_step_2/actor'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'model_world_size_8_rank_0.pt').touch()
            (root / 'train.log').write_text(''.join(
                f'step:{step} - actor/grad_norm:np.float64(0.1) - actor/probe_parameter_delta_max:1e-7\n'
                for step in (1, 2)))
            batch = {'varying_groups': 1, 'groups': 1, 'records': [
                {'status': 'answered', 'tool_template_tokens': 100}]}
            (root / 'reward_audit/rewards_1.jsonl').write_text(json.dumps(batch) + '\n')
            self.assertTrue(summarize(root, 2, 0))

    def test_durable_metrics_survive_missing_console_step(self):
        spec = importlib.util.spec_from_file_location('tracking_under_test', Path('verl/utils/tracking.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'main_example'
            root.mkdir()
            logger = module.Tracking('test', 'test', default_backend=['jsonl'],
                                     config={'trainer': {'metrics_jsonl': str(root / 'metrics.jsonl')}})
            for step in (1, 2):
                logger.log({'actor/grad_norm': .1, 'actor/probe_parameter_delta_max': 1e-6}, step)
            (root / 'train.log').write_text('step:1 - actor/grad_norm:0.1\n')
            summarize(root, 2, 0)
            report = json.loads((root / 'report.json').read_text())
            self.assertEqual(set(report['steps']), {'1', '2'})
            self.assertEqual(report['metric_source'], 'metrics.jsonl')
            # Missing checkpoint/reward evidence must still prevent PASSED.
            self.assertEqual(report['status'], 'NEEDS_REVIEW')

    def test_summary_never_promotes_zero_update(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'smoke_example'
            root.mkdir()
            (root / 'train.log').write_text('step:1 - actor/grad_norm:0.0 - actor/probe_parameter_delta_max:0.0\n')
            self.assertFalse(summarize(root, 2, 0))
            self.assertIn('NEEDS_REVIEW', (root / 'summary.txt').read_text())


if __name__ == '__main__':
    unittest.main()
