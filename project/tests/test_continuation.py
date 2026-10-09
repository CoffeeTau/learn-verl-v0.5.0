import unittest
from project.evaluation.continuation import run_continuation
from project.training.protocol import trajectory_score


class Tokenizer:
    def __init__(self, texts):
        self.texts = texts
        self.templates = 0
        self.inserted = []
    def apply_chat_template(self, messages, **kwargs):
        self.templates += 1
        return [10, 11]
    def decode(self, ids, **kwargs):
        return self.texts[ids[0]]
    def convert_tokens_to_ids(self, text):
        return 99
    def encode(self, text, **kwargs):
        self.inserted.append(text)
        return [20, 21]


class ContinuationTests(unittest.TestCase):
    def run_case(self, outputs, texts, **overrides):
        cfg = dict(max_searches=1, max_new_tokens=4, max_context_tokens=40, top_k=3)
        cfg.update(overrides)
        tokenizer = Tokenizer(texts)
        prompts, calls = [], []
        it = iter(outputs)
        def generate(ids):
            prompts.append(ids[:])
            return next(it)
        def search(*args):
            calls.append(args)
            return [dict(id='p', title='<title>', text='A & B')]
        result = run_continuation('Q', generate, search, cfg, tokenizer)
        return result, tokenizer, prompts, calls

    def test_preserves_ids_and_eos_with_one_template(self):
        r, t, prompts, calls = self.run_case([[1, 99], [2]],
            {1:'<search>entity</search>', 2:'<judge>sufficient: found</judge><answer>A</answer>'})
        self.assertEqual(prompts[1], [10, 11, 1, 99, 20, 21])
        self.assertEqual(t.templates, 1)
        self.assertTrue(t.inserted[0].startswith('\n<|im_start|>user'))
        self.assertIn('A &amp; B', t.inserted[0])
        self.assertEqual(r['status'], 'answered')
        self.assertEqual(r['total_tokens'], 11)

    def test_cap_rejects_even_complete_answer(self):
        r, _, _, _ = self.run_case([[1, 2, 3, 4]], {1:'<answer>A</answer>'})
        self.assertEqual(r['status'], 'generation_truncated')
        self.assertEqual(r['answer'], '')

    def test_no_room_for_observation_stops_before_next_generation(self):
        r, _, prompts, calls = self.run_case([[1]], {1:'<search>entity</search>'}, max_context_tokens=8)
        self.assertEqual(len(prompts), 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(r['status'], 'unfinished_or_budget')

    def test_excess_search_and_conflict_match_trainer_status(self):
        for last in ('<judge>insufficient: missing</judge><search>other</search>',
                     '<judge>insufficient: missing</judge><answer>A</answer>'):
            texts = {1:'<search>entity</search>', 2:last}
            r, _, _, calls = self.run_case([[1], [2]], texts)
            expected = trajectory_score(list(texts.values()), 'A', max_searches=1, version='v2')
            self.assertEqual(r['status'], expected['status'])
            self.assertEqual(r['searches'], expected['searches'])
            self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
