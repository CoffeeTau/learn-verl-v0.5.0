"""Train-only annotated-support citation proxy, not a semantic entailment judge."""
import re
from project.agent.correction import parse_action
from project.common import load_jsonl, save_json, sha256


def prepare_supports(source, corpus, output):
    import json
    manifest = json.loads((source / 'manifest.json').read_text())
    if sha256(source / 'train.labels.jsonl') != manifest['files']['train.labels.jsonl']:
        raise ValueError('Frozen train labels changed')
    if sha256(corpus / 'corpus.jsonl') != manifest['corpus_sha256']:
        raise ValueError('Frozen corpus changed')
    paragraphs = {r['id']: r['paragraph_id'] for r in load_jsonl(corpus / 'corpus.jsonl')}
    supports = {r['id']: sorted({paragraphs[f['passage_id']] for f in r['supporting_facts']})
                for r in load_jsonl(source / 'train.labels.jsonl')}
    if any(not s for s in supports.values()):
        raise ValueError('Empty train support annotation')
    save_json(output, {'paragraphs': paragraphs, 'supports': supports})


def observed_ids(observations):
    # Only decode mask=0 spans, never model-written citations/known. Corpus text
    # escapes '<', so it cannot manufacture a second information wrapper.
    ids = set()
    for text in observations:
        match = re.search(r'<information>\n(.*?)\n</information>', text, re.S)
        if not match:
            raise ValueError('Observation information wrapper missing')
        ids.update(re.findall(r'(?:\A|\n\n)\[([0-9a-f]{24,64})\] ', match[1]))
    return ids


def citation_proxy(texts, observations, score, task_id, support_data, weight=0.2):
    if not 0 <= weight <= 1:
        raise ValueError('Invalid evidence reward weight')
    target = set(support_data['supports'][task_id])
    seen = observed_ids(observations)
    citations = set()
    if score['status'] == 'answered':
        citations = set(parse_action(texts[-1], after_search=score['searches'] > 0).get('sources', []))
    invalid = citations - seen
    paragraph_map = support_data['paragraphs']
    cited = {paragraph_map[c] for c in citations & seen if c in paragraph_map}
    overlap = len(cited & target)
    # Invalid IDs zero the bonus. Extra irrelevant paragraphs reduce precision;
    # repeating IDs or citing multiple chunks of one paragraph does not help.
    quality = 2 * overlap / (len(cited) + len(target)) if cited and not invalid else 0.
    reward = (1 - weight) * score['score'] + weight * float(score['em'] == 1) * quality
    return {'training_reward': reward, 'citation_support_f1': quality,
            'observed_support_recall': len({paragraph_map[c] for c in seen if c in paragraph_map} & target) / len(target),
            'invalid_citation_count': len(invalid)}
