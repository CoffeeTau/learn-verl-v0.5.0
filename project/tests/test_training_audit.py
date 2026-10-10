import json
from pathlib import Path
import tempfile
import unittest

from project.scripts.audit_v4_training import audit, candidates, classify


def batch(scores, em=None):
    return dict(episodes=len(scores), groups=1,
                varying_groups=int(min(scores) != max(scores)),
                records=[dict(task_id='same-task', score=s, em=e, status='answered',
                              searches=3, model_outputs=['<search>query</search>'])
                         for s, e in zip(scores, em or [int(s == 1) for s in scores])])


class TrainingAuditTest(unittest.TestCase):
    def test_reward_classes(self):
        self.assertEqual([classify(v) for v in ([0]*4, [1]*4, [.5]*4, [0, .5, 1, 1])],
                         ['all_zero', 'all_full_f1', 'same_partial_f1', 'varied'])

    def test_batch_boundaries_validation_and_f1_em(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            root = run / 'reward_audit'
            (root / 'validation').mkdir(parents=True)
            rows = [batch([0]*4), batch([1]*4, [0]*4), batch([0, 1, 0, 1])]
            (root / 'rewards_1.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            (root / 'validation' / 'rewards_1.jsonl').write_text('invalid json')
            report, groups = audit(run)
            self.assertEqual(report['groups'], 3)
            self.assertEqual(report['episodes'], 12)
            self.assertEqual(report['full_f1_but_em_zero'], 4)
            self.assertEqual(report['em_groups'], {'all_wrong': 2, 'mixed': 1})
            self.assertFalse(report['warnings'])
            self.assertEqual(len(candidates(groups, 1)['mixed_em']), 1)

    def test_incomplete_and_header_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'reward_audit').mkdir()
            row = batch([0, 0])
            row['episodes'] = 4
            (run / 'reward_audit' / 'rewards_1.jsonl').write_text(json.dumps(row))
            report, groups = audit(run)
            self.assertEqual(report['complete_groups'], 0)
            self.assertEqual(len(report['warnings']), 2)
            self.assertEqual(candidates(groups, 2), {})

    def test_corruption_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'reward_audit').mkdir()
            path = run / 'reward_audit' / 'rewards_1.jsonl'
            path.write_text('{')
            with self.assertRaisesRegex(ValueError, 'rewards_1.jsonl:1'):
                audit(run)
            row = batch([0]*4)
            row['records'][0]['eval_group'] = 'natural'
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'Non-training'):
                audit(run)


if __name__ == '__main__':
    unittest.main()
