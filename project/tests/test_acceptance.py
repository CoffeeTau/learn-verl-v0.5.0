"""Acceptance boundaries: fixed hard subset, first-call-only perturbation, strict scoring."""
import unittest
from types import SimpleNamespace
from project.evaluation.acceptance import episode_search, aggregate, parse_gpus, compatible_code, LEGACY_SERIAL_SHA
from project.training.hard_episodes import build_plan


class AcceptanceTests(unittest.TestCase):
    def test_parallel_devices_and_narrow_migration(self):
        self.assertEqual(parse_gpus('0,1,2,3'), ['0', '1', '2', '3'])
        for value in ('', '0,0', '0,', '-1', '0;1'):
            with self.assertRaises(ValueError):
                parse_gpus(value)
        key = 'project/evaluation/acceptance.py'
        old = {key: LEGACY_SERIAL_SHA, 'scoring': 'unchanged'}
        new = {key: 'parallel', 'scoring': 'unchanged'}
        self.assertTrue(compatible_code(old, new))
        self.assertFalse(compatible_code(old, {**new, 'scoring': 'changed'}))
        self.assertFalse(compatible_code({**old, key: 'unknown'}, new))

    def test_test_plan_is_fixed_and_balanced(self):
        tasks = [dict(id=str(i), question=f'question {i}') for i in range(300)]
        labels = {r['id']: {'supporting_facts': [{'passage_id': 'a'}, {'passage_id': 'b'}]} for r in tasks}
        corpus = [dict(id='a', paragraph_id='pa'), dict(id='b', paragraph_id='pb')]
        plan = build_plan(tasks, labels, corpus, 42)
        self.assertEqual(plan, build_plan(list(reversed(tasks)), labels, corpus, 42))
        self.assertEqual(len(plan), 75)
        self.assertEqual(sum(p['mode'] == 'withhold_one' for p in plan.values()), 38)
        self.assertTrue(all('answer' not in p for p in plan.values()))

    def test_perturbation_is_private_and_only_first_search(self):
        hits = [dict(id='a', title='A', text='Original A'), dict(id='b', title='B', text='Original B')]
        seen = []
        def search(question, query, k):
            seen.append((question, query, k))
            return hits
        retriever = SimpleNamespace(search=search)
        spec = dict(mode='withhold_one', exclude_paragraphs=['pa'])
        audit = {'applied': False}
        wrapped = episode_search(retriever, spec, {'a': 'pa', 'b': 'pb'}, audit)
        self.assertEqual(wrapped('q', 'first', 3), hits[1:])
        self.assertEqual(wrapped('q', 'second', 3), hits)
        self.assertTrue(audit['applied'])
        self.assertEqual(seen, [('q', 'first', 20), ('q', 'second', 3)])
        natural_audit = {'applied': False}
        self.assertEqual(episode_search(retriever, None, {}, natural_audit)('q', 'first', 3), hits)
        self.assertFalse(natural_audit['applied'])

    def test_policy_failures_stay_in_score_denominator(self):
        base = dict(answer='x', searches=1, total_tokens=10, seconds=1)
        result = aggregate([dict(base, em=1, f1=1, status='answered'),
                            dict(base, em=0, f1=0, status='missing_judge')], 2)
        self.assertEqual(result['em_percent'], 50)
        self.assertEqual(result['completed'], 2)


if __name__ == '__main__':
    unittest.main()
