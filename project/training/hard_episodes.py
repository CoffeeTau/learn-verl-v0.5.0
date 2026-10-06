"""Train-only retrieval perturbations. Labels remain private to the environment."""
import hashlib
import json
from collections import Counter
from project.common import load_jsonl, save_json, sha256


def build_plan(tasks, labels, passages, seed=42):
    by_id = {p['id']: p['paragraph_id'] for p in passages}
    ranked = sorted(tasks, key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest())
    plan = {}
    for index, row in enumerate(ranked[:len(ranked) // 4]):
        paragraphs = sorted({by_id[fact['passage_id']] for fact in labels[row['id']]['supporting_facts']})
        if len(paragraphs) < 2:
            raise ValueError('Hard episode needs at least two mapped support paragraphs')
        mode = 'withhold_one' if index % 2 == 0 else 'related_distractors'
        plan[row['question']] = {'task_id': row['id'], 'mode': mode,
                                 'exclude_paragraphs': paragraphs[:1] if mode == 'withhold_one' else paragraphs}
    return plan


def transform_hits(hits, candidates, specification, passage_paragraphs, top_k):
    """No document fabrication and no permanent corpus deletion.

    Support exclusion creates a controlled challenge, not a semantic proof that
    all remaining passages are useless: alternative evidence may still exist.
    """
    banned = set(specification['exclude_paragraphs'])
    pool = hits if specification['mode'] == 'withhold_one' else candidates
    selected = [hit for hit in pool if passage_paragraphs[hit['id']] not in banned][:top_k]
    return selected, [hit['id'] for hit in selected] != [hit['id'] for hit in hits]


def prepare_plan(source, corpus, output):
    manifest = json.loads((source / 'manifest.json').read_text())
    for name in ('train.jsonl', 'train.labels.jsonl'):
        if sha256(source / name) != manifest['files'][name]:
            raise ValueError('Frozen train data changed')
    if sha256(corpus / 'corpus.jsonl') != manifest['corpus_sha256']:
        raise ValueError('Frozen corpus changed')
    tasks = load_jsonl(source / 'train.jsonl')
    labels = {row['id']: row for row in load_jsonl(source / 'train.labels.jsonl')}
    plan = build_plan(tasks, labels, load_jsonl(corpus / 'corpus.jsonl'))
    save_json(output, plan)
    counts = dict(Counter(row['mode'] for row in plan.values()))
    return {'normal_tasks': len(tasks) - len(plan), 'planned_hard_tasks': len(plan), 'modes': counts,
            'plan_sha256': sha256(output), 'corpus_sha256': manifest['corpus_sha256'],
            'note': 'Only first retrieval; actual applied count may be smaller; no judgement labels exposed.'}
