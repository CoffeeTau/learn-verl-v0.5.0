"""V2 protocol, reward and controlled retrieval boundaries."""
import asyncio
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from project.agent.correction import parse_action, SYSTEM_PROMPT
from project.training.protocol import trajectory_score, model_segments
from project.training.hard_episodes import build_plan, transform_hits
from project.tests.test_training import CharTokenizer


class CorrectionTests(unittest.TestCase):
    def test_judgement_is_required_and_not_a_reward(self):
        search = '<search>Example School founder</search>'
        with self.assertRaisesRegex(ValueError, 'missing_judge'):
            parse_action(search, after_search=True)
        with self.assertRaisesRegex(ValueError, 'judge_action_mismatch'):
            parse_action('<judge>sufficient: founder known</judge>' + search, after_search=True)
        result = trajectory_score([search, '<judge>sufficient: birth city found</judge><answer>Paris</answer>'],
                                  'London', version='v2')
        self.assertEqual(result['score'], 0)
        result = trajectory_score([search, '<judge>insufficient: birthplace missing</judge><answer>insufficient evidence</answer>'],
                                  'Paris', version='v2')
        self.assertEqual(result['score'], 0)
        with self.assertRaisesRegex(ValueError, 'unexpected_judge'):
            parse_action('<judge>insufficient: school unknown</judge>' + search)

    def test_bridge_then_followup_scores_terminal_answer(self):
        texts = ['<search>Person school</search>',
                 '<judge>insufficient: school identified but founder birthplace missing</judge><search>School founder birthplace</search>',
                 '<judge>sufficient: founder and birthplace supported</judge><answer>Paris</answer>']
        score = trajectory_score(texts, 'Paris', version='v2')
        self.assertEqual(score['score'], 1)
        self.assertEqual(score['searches'], 2)
        self.assertEqual(trajectory_score(texts[:2], 'Paris', version='v2')['score'], 0)

    def test_plan_is_deterministic_and_keeps_most_tasks_normal(self):
        tasks = [{'id': str(i), 'question': f'question {i}'} for i in range(16)]
        labels = {row['id']: {'supporting_facts': [{'passage_id': 'a'}, {'passage_id': 'b'}]} for row in tasks}
        passages = [{'id': 'a', 'paragraph_id': 'pa'}, {'id': 'b', 'paragraph_id': 'pb'}]
        plan = build_plan(tasks, labels, passages)
        self.assertEqual(len(plan), 4)
        self.assertEqual(plan, build_plan(list(reversed(tasks)), labels, passages))
        self.assertEqual(sum(row['mode'] == 'withhold_one' for row in plan.values()), 2)
        self.assertTrue(all(set(row) == {'task_id', 'mode', 'exclude_paragraphs'} for row in plan.values()))

    def test_no_document_fabrication_or_permanent_removal(self):
        hits = [{'id': x, 'text': 'original'} for x in 'abcde']
        paragraphs = dict(zip('abcde', ['pa', 'pa', 'pb', 'pc', 'pd']))
        selected, applied = transform_hits(hits[:3], hits, {'mode': 'withhold_one', 'exclude_paragraphs': ['pa']}, paragraphs, 3)
        self.assertEqual(selected, [hits[2]])
        self.assertTrue(applied)
        selected, _ = transform_hits(hits[:3], hits, {'mode': 'related_distractors', 'exclude_paragraphs': ['pa', 'pb']}, paragraphs, 3)
        self.assertEqual(selected, hits[3:])
        self.assertEqual(len(hits), 5)

    def test_actual_retriever_perturbs_only_first_training_return(self):
        ray = ModuleType('ray')
        ray.remote = lambda **kwargs: lambda cls: cls
        local = ModuleType('project.retrieval.local')
        local.LocalRetriever = object
        with patch.dict(sys.modules, {'ray': ray, local.__name__: local}):
            spec = importlib.util.spec_from_file_location('retriever_test', Path('project/training/retriever.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        actor = object.__new__(module.RetrievalActor)
        hits = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]
        actor.retriever = SimpleNamespace(search=lambda q, query, k: hits[:k])
        actor.plan = {'train question': {'mode': 'withhold_one', 'exclude_paragraphs': ['pa']}}
        actor.paragraphs = {'a': 'pa', 'b': 'pb', 'c': 'pc'}
        actor.audit_path = None
        self.assertEqual(actor.search_v2('train question', 'q', 3, 0, True), hits[1:])
        self.assertEqual(actor.search_v2('train question', 'q', 3, 1, True), hits)
        self.assertEqual(actor.search_v2('train question', 'q', 3, 0, False), hits)
        self.assertEqual(actor.search_v2('dev question', 'q', 3, 0, True), hits)

    def test_actual_loop_keeps_judges_trainable_and_eval_clean(self):
        framework = ModuleType('verl.experimental.agent_loop.agent_loop')
        framework.AgentLoopBase = object
        framework.AgentLoopMetrics = lambda: SimpleNamespace(generate_sequences=0., tool_calls=0.)
        framework.AgentLoopOutput = lambda **kwargs: SimpleNamespace(**kwargs)
        with patch.dict(sys.modules, {'ray': ModuleType('ray'), framework.__name__: framework}):
            spec = importlib.util.spec_from_file_location('correction_adapter_test', Path('project/training/agent_loop.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        tokenizer = CharTokenizer()
        for perturb in (True, False):
            outputs = ['<search>Paris country</search>', '<judge>sufficient: location supported</judge><answer>France</answer>']
            tokenized = [tokenizer.encode(s) for s in outputs]
            pending = list(tokenized)
            flags = []
            async def generate(*args, **kwargs):
                return pending.pop(0)
            async def search(question, query, k, turn, enabled):
                flags.append((turn, enabled))
                return [{'id': 'x', 'title': 'Paris', 'text': 'Paris is in France.'}]
            agent = object.__new__(module.CorrectionAgentLoop)
            agent.version, agent.perturb = 'v2', perturb
            agent.settings = dict(max_searches=4, max_new_tokens=512, max_context_tokens=8192, top_k=3)
            agent.tokenizer = tokenizer
            agent.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(prompt_length=512, response_length=8192, top_k=20)))
            agent.server_manager = SimpleNamespace(generate=generate)
            agent.retriever = SimpleNamespace(search_v2=SimpleNamespace(remote=search))
            result = asyncio.run(agent.run([{'role': 'system', 'content': SYSTEM_PROMPT.format(max_searches=4)},
                                           {'role': 'user', 'content': 'Where?'}], {}))
            self.assertEqual(flags, [(0, perturb)])
            self.assertEqual(model_segments(result.response_ids, result.response_mask), tokenized)
