"""Compact saved-trajectory review; never changes predictions or official scores."""
import argparse
from collections import Counter
import json
import re

from project.agent.correction import parse_action
from project.agent.search import parse_action as parse_base
from project.common import load_jsonl, resource_path
from project.scripts.review_v1 import paired_counts


def mismatch_detail(row):
    text = row['steps'][-1]['output'].strip()
    match = re.match(r'<judge>(sufficient|insufficient):\s*([^<>]+)</judge>\s*', text, re.DOTALL)
    try:
        parse_action(text, after_search=row['searches'] > 0)
    except ValueError as exc:
        if str(exc) != 'judge_action_mismatch':
            return 'parser_drift', '', ''
    else:
        return 'parser_drift', '', ''
    action = parse_base(text[match.end():])
    kind = action['kind']
    if kind == 'answer' and action['answer'].casefold() == 'insufficient evidence':
        kind = 'refusal'
    return f'{match[1]}->{kind}', match[2].strip(), action.get('query', action.get('answer', ''))


def costs(rows):
    if not rows:
        return 'n=0'
    return (f'n={len(rows)} searches={sum(r["searches"] for r in rows)/len(rows):.2f} '
            f'tokens={sum(r["total_tokens"] for r in rows)/len(rows):.0f}')


def build_summary(before, after):
    counts = paired_counts(before, after)
    old = {r['id']: r for r in before}
    failed = [r for r in after if r['status'] != 'answered']
    mismatches = [r for r in failed if r['status'] == 'judge_action_mismatch']
    details = [(r, mismatch_detail(r)) for r in mismatches]
    repeated = 0
    coverage = Counter()
    for row in after:
        queries = [' '.join(s['query'].casefold().split()) for s in row['steps'] if 'query' in s]
        repeated += len(queries) != len(set(queries))
        if not row['em']:
            found, total = row.get('support_chunks_found', 0), row.get('support_chunks_total', 0)
            coverage['unknown' if not total else 'all' if found == total else 'some' if found else 'none'] += 1
    paired = [r for r in after if r['status'] == old[r['id']]['status'] == 'answered']
    lines = ['=== V2 OFFLINE REVIEW | no inference / no rescoring ===',
             f'Tasks={len(after)} | fixed ID/question/gold alignment checked',
             f'Wrong->right={counts[0,1]} | right->wrong={counts[1,0]} | both right={counts[1,1]} | both wrong={counts[0,0]}',
             f'Statuses={dict(Counter(r["status"] for r in after))}',
             f'Mismatch types={dict(Counter(d[0] for _, d in details))}',
             f'Mismatch after searches={dict(Counter(r["searches"] for r in mismatches))}',
             f'Nonexact gold coverage={dict(coverage)} | repeated queries={repeated}/{len(after)}',
             'Failed costs: ' + costs(failed),
             'Both-answered V1: ' + costs([old[r['id']] for r in paired]),
             'Both-answered V2: ' + costs(paired),
             'Subset costs are descriptive; coverage does not establish failure cause.']
    # One example per conflict type, bounded for screenshot exchange.
    seen = set()
    for row, (kind, reason, action) in details:
        if kind in seen or len(seen) >= 3:
            continue
        seen.add(kind)
        compact = lambda s: ' '.join(s.split())[:150]
        lines.extend([f'Case {kind} | id={row["id"]} | searches={row["searches"]}',
                      'Judge: ' + compact(reason), 'Action: ' + compact(action)])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v1', default='dev_20261006T151952Z_111fcb')
    parser.add_argument('--v2', default='dev_20261007T024032Z_0bed4f')
    args = parser.parse_args()
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', value) for value in vars(args).values()):
        parser.error('Invalid run ID')
    runs = resource_path('AGENTIC_RUNS_DIR')
    folders = [runs / 'v1_eval' / args.v1, runs / 'v2_eval' / args.v2]
    rows = []
    for folder in folders:
        report = json.loads((folder / 'report.json').read_text())
        data = load_jsonl(folder / 'trajectories.jsonl')
        if report['status'] != 'passed' or len(data) != report['metrics']['planned'] or len(data) != report['metrics']['completed']:
            raise ValueError('Incomplete evaluation')
        rows.append(data)
    text = build_summary(*rows)
    (folders[1] / 'review_summary.txt').write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
