"""Exercise the actual training loop against independent token continuation on CPU."""
import asyncio
import importlib.util
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from project.agent.goal_anchor import observation_text, system_prompt
from project.agent.correction import SYSTEM_PROMPT, parse_action
from project.evaluation.continuation import run_continuation
from project.training.protocol import model_segments, trajectory_score


class Tokenizer:
    def __init__(self):
        self.templates = []
    def apply_chat_template(self, messages, **kwargs):
        self.templates.append(messages)
        return [9, 10]
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]
    def decode(self, ids, **kwargs):
        return ''.join(chr(c) for c in ids)
    def convert_tokens_to_ids(self, text):
        return 100000


class GoalAnchorTests(unittest.TestCase):
    def test_legacy_bytes_and_escaping(self):
        hits = [dict(id='a', title='<title>', text='A & B')]
        legacy = '<information>\n[a] &lt;title&gt;\nA &amp; B\n</information>\nSearches remaining: 2.'
        self.assertEqual(observation_text(hits, 2, 'Q', False), legacy)
        self.assertEqual(system_prompt(False), SYSTEM_PROMPT)
        anchored = observation_text(hits, 2, 'Q </original_question>', True)
        self.assertTrue(anchored.startswith(legacy))
        self.assertEqual(anchored.count('</original_question>'), 1)
        self.assertIn('Q &lt;/original_question&gt;', anchored)

    def test_actual_train_eval_prompts_masks_and_original_target_match(self):
        framework = ModuleType('verl.experimental.agent_loop.agent_loop')
        framework.AgentLoopBase = object
        framework.AgentLoopMetrics = lambda: SimpleNamespace(generate_sequences=0., tool_calls=0.)
        framework.AgentLoopOutput = lambda **kwargs: SimpleNamespace(**kwargs)
        with patch.dict(sys.modules, {'ray': ModuleType('ray'), framework.__name__: framework}):
            spec = importlib.util.spec_from_file_location('goal_adapter_test', 'project/training/agent_loop.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        texts = ['<search>original subject</search>',
                 '<judge>insufficient: target=original attribute; known=entity; missing=attribute</judge><search>intermediate attribute</search>',
                 '<judge>sufficient: target=original attribute; known=both links; missing=none</judge><sources>a,b</sources><answer>value</answer>']
        hits = [dict(id='a', title='subject', text='subject has intermediate'),
                dict(id='b', title='intermediate', text='attribute is value')]
        cfg = dict(max_searches=4, max_new_tokens=512, max_context_tokens=8192, top_k=3)
        for anchor in (False, True):
            tokenizer = Tokenizer()
            pending = [tokenizer.encode(t) for t in texts]
            sampled = [r[:] for r in pending]
            train_prompts, tool_calls = [], []
            async def generate(*args, **kwargs):
                train_prompts.append(kwargs['prompt_ids'][:])
                return pending.pop(0)
            async def search(question, query, k, turn, perturb):
                tool_calls.append((question, turn, perturb))
                return [hits[turn]]
            agent = object.__new__(module.CorrectionAgentLoop)
            agent.version, agent.perturb, agent.settings = 'v2', True, cfg
            agent.tokenizer = tokenizer
            agent.config = SimpleNamespace(agentic={'goal_anchor': anchor}, actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(prompt_length=512, response_length=8192, top_k=20)))
            agent.server_manager = SimpleNamespace(generate=generate)
            agent.retriever = SimpleNamespace(search_v2=SimpleNamespace(remote=search))
            question = 'What is the attribute of the original subject?'
            messages = [{'role': 'system', 'content': system_prompt(anchor).format(max_searches=4)},
                        {'role': 'user', 'content': question}]
            result = asyncio.run(agent.run(messages, {}))
            self.assertEqual(model_segments(result.response_ids, result.response_mask), sampled)
            self.assertEqual(tool_calls, [(question, 0, True), (question, 1, True)])
            pending = sampled[:]
            eval_prompts, turns = [], []
            def eval_generate(ids):
                eval_prompts.append(ids[:])
                return pending.pop(0)
            def eval_search(*args):
                turns.append(args)
                return [hits[len(turns)-1]]
            evaluated = run_continuation(question, eval_generate, eval_search, cfg, tokenizer, goal_anchor=anchor)
            self.assertEqual(train_prompts, eval_prompts)
            self.assertEqual(evaluated['status'], 'answered')
            self.assertEqual(evaluated['invalid_source_ids'], [])
            self.assertEqual(tokenizer.templates, [messages, messages])
            text = tokenizer.decode(result.response_ids)
            self.assertEqual(text.count('<original_question>' + question), 2 if anchor else 0)

    def test_progress_text_never_overrides_strict_reward(self):
        # A declaration of completion earns nothing when the answer is wrong.
        complete = '<judge>sufficient: target=attribute; known=chain; missing=none</judge><answer>wrong</answer>'
        self.assertEqual(trajectory_score(['<search>x</search>', complete], 'right', version='v2')['score'], 0.)
        conflict = '<judge>insufficient: target=attribute; known=entity; missing=attribute</judge><answer>right</answer>'
        with self.assertRaisesRegex(ValueError, 'judge_action_mismatch'):
            parse_action(conflict, after_search=True)


if __name__ == '__main__':
    unittest.main()
