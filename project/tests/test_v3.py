"""V3 supervision boundaries and source-preserving state integration."""
from copy import deepcopy
import unittest

from project.agent.correction import parse_action
from project.agent.search import run_episode
from project.agent.state import SYSTEM_PROMPT, state_messages
from project.retrieval.v3 import allowed_negative, relation_candidates


class V3Tests(unittest.TestCase):
    def test_positive_requires_labeled_sentence_and_exact_subject_title(self):
        corpus = [dict(id='p', paragraph_id='a', title='Colin, First Earl',
                       sentences=['His father was Archibald, Master.', 'Other text.'], sentence_ids=[0, 1])]
        tasks = [dict(id='train', question='Who is the grandfather?')]
        label = dict(supporting_facts=[dict(passage_id='p', sentence_id=0)],
                     evidences=[['Colin, First Earl', 'father', 'Archibald, Master']])
        pairs, _ = relation_candidates(tasks, {'train': label}, corpus)
        self.assertEqual(len(pairs), 1)
        self.assertNotIn('Archibald', pairs[0]['query'])
        label['supporting_facts'][0]['sentence_id'] = 1
        self.assertEqual(relation_candidates(tasks, {'train': label}, corpus)[0], [])
        label['supporting_facts'][0]['sentence_id'] = 0
        corpus[0]['title'] = 'Colin, Fourth Earl'
        self.assertEqual(relation_candidates(tasks, {'train': label}, corpus)[0], [])

    def test_negative_excludes_support_and_alternative_subject_object_mentions(self):
        candidate = dict(subject='Colin, First Earl', object='Archibald Master', excluded_paragraphs=['a'])
        row = dict(title='Other person', text='Unrelated relation', paragraph_id='b')
        self.assertTrue(allowed_negative(row, candidate))
        for changed in (dict(paragraph_id='a'), dict(text='Colin First Earl has a father'),
                        dict(title='Archibald Master')):
            self.assertFalse(allowed_negative({**row, **changed}, candidate))

    def test_state_retains_distinct_entities_sources_and_unverified_claims(self):
        hits = [dict(id='p', title='Archibald, Master', text='Original <sentence>.'),
                dict(id='q', title='Archibald, Fourth Earl', text='Different entity.')]
        steps = [dict(query='father', hits=hits),
                 dict(query='grandfather', hits=[hits[0]], judge='insufficient', reason='Need father of father')]
        before = deepcopy(steps)
        payload = state_messages('question', steps, 2, 'system')[-1]['content']
        self.assertEqual(payload.count('[p]'), 1)
        self.assertIn('[q]', payload)
        self.assertIn('Original &lt;sentence&gt;.', payload)
        self.assertIn('unverified', payload)
        self.assertIn('false', payload)
        self.assertEqual(steps, before)
        changed = deepcopy(steps)
        changed[-1]['hits'] = [{**changed[-1]['hits'][0], 'text': 'silently replaced'}]
        with self.assertRaisesRegex(ValueError, 'changed content'):
            state_messages('question', changed, 2, 'system')

    def test_actual_loop_keeps_strict_mismatch_and_same_call_budget(self):
        texts = iter(['<search>Colin father</search>',
                      '<judge>insufficient: missing relation</judge><answer>Archibald</answer>'])
        prompts = []
        def generate(messages):
            prompts.append(deepcopy(messages))
            return dict(text=next(texts), finish_reason='stop', prompt_tokens=10, output_tokens=5)
        def search(*args):
            return [dict(id='p', title='Colin', text='Original text')]
        result = run_episode('question', generate, search, dict(max_searches=4, top_k=3),
                             system_prompt=SYSTEM_PROMPT, action_parser=parse_action, history_builder=state_messages)
        self.assertEqual(result['status'], 'judge_action_mismatch')
        self.assertEqual(result['answer'], '')
        self.assertEqual(result['total_tokens'], 30)
        self.assertEqual(len(prompts), 2)
        self.assertIn('<task_state>', prompts[1][-1]['content'])
        self.assertNotIn('gold', prompts[1][-1]['content'])


if __name__ == '__main__':
    unittest.main()
