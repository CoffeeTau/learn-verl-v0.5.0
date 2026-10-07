"""CPU-only validation accounting and a predeclared V4 checkpoint rule."""
import math


def token_cost(prompt_tokens, mask):
    """Logical sampled input+output tokens, counting repeated prompt reads (no padding)."""
    total, active = 0, False
    for index, sampled in enumerate(mask):
        if sampled and not active:
            total += prompt_tokens + index
        if sampled:
            total += 1
        active = bool(sampled)
    return total


def summarize_validation(rows, expected):
    buckets = {'natural': [], 'hard': []}
    for row in rows:
        group = row['eval_group']
        if group not in buckets:
            raise ValueError('Unexpected validation group')
        buckets[group].append(row)
    for group, values in buckets.items():
        ids = [r['task_id'] for r in values]
        if len(ids) != len(set(ids)) or set(ids) != set(expected[group]):
            raise ValueError(f'Incomplete/duplicate validation tasks: {group}')
    hard_ids = set(expected['hard'])
    if not hard_ids or not hard_ids <= set(expected['natural']):
        raise ValueError('Hard validation must be a nonempty natural subset')
    buckets['natural_matched'] = [r for r in buckets['natural'] if r['task_id'] in hard_ids]
    result = {}
    for group, values in buckets.items():
        for row in values:
            if not all(math.isfinite(float(row[k])) for k in ('em', 'score', 'total_tokens', 'searches')):
                raise ValueError('Nonfinite validation metric')
        n = len(values)
        result[group] = {'n': n, 'em': sum(r['em'] for r in values) / n,
                         'f1': sum(r['score'] for r in values) / n,
                         'protocol_failures': sum(r['status'] != 'answered' for r in values),
                         'tokens': sum(r['total_tokens'] for r in values) / n,
                         'searches': sum(r['searches'] for r in values) / n}
    return result


def select_checkpoint(evaluations):
    if not evaluations or int(evaluations[0]['step']) != 0:
        raise ValueError('Selection requires a measured step0')
    base = evaluations[0]['groups']
    eligible = [row for row in evaluations
                if row['groups']['natural']['protocol_failures'] <= base['natural']['protocol_failures']
                and row['groups']['hard']['em'] >= base['hard']['em']]
    def rank(row):
        natural = row['groups']['natural']
        return natural['em'], natural['f1'], -natural['tokens'], -int(row['step'])
    best = max(eligible, key=rank)
    return {'selected_step': int(best['step']), 'improved_over_step0': int(best['step']) != 0,
            'eligible_steps': [int(r['step']) for r in eligible], 'groups': best['groups'],
            'rule': 'natural EM, F1, fewer tokens; failures<=step0 and hard EM>=step0; exact ties earlier step'}
