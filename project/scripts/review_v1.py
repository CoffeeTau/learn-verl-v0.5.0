"""One compact offline review before V2; no inference, rescore or sample selection."""
import argparse
from collections import Counter, defaultdict
import json
import re
from project.common import load_jsonl, resource_path


def paired_counts(before, after):
    old = {row['id']: row for row in before}
    new = {row['id']: row for row in after}
    if len(old) != len(before) or len(new) != len(after) or old.keys() != new.keys():
        raise ValueError('Duplicate/mismatched development IDs')
    counts = Counter()
    for key, row in new.items():
        if old[key]['question'] != row['question'] or old[key]['gold'] != row['gold']:
            raise ValueError('Questions/gold changed between evaluations')
        counts[(int(old[key]['em']), int(row['em']))] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v0', default='dev_20261006T043535Z_7da4f2')
    parser.add_argument('--v1', default='dev_20261006T151952Z_111fcb')
    parser.add_argument('--train', default='main_20261006T090840Z_c41e35')
    args = parser.parse_args()
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', x) for x in vars(args).values()):
        parser.error('Invalid run ID')
    runs = resource_path('AGENTIC_RUNS_DIR')
    folders = [runs / 'v0' / args.v0, runs / 'v1_eval' / args.v1]
    for folder in folders:
        report = json.loads((folder / 'report.json').read_text())
        if report['status'] != 'passed' or report['metrics']['completed'] != report['metrics']['planned']:
            raise ValueError('Evaluation is incomplete')
    before, after = [load_jsonl(folder / 'trajectories.jsonl') for folder in folders]
    counts = paired_counts(before, after)
    wrong = [row for row in after if row['em'] == 0]
    coverage = Counter()
    for row in wrong:
        found, total = row.get('support_chunks_found', 0), row.get('support_chunks_total', 0)
        coverage['unknown' if not total else 'all' if found == total else 'some' if found else 'none'] += 1
    repeated = 0
    for row in after:
        queries = [' '.join(step['query'].casefold().split()) for step in row['steps'] if 'query' in step]
        repeated += len(queries) != len(set(queries))
    groups = defaultdict(list)
    for path in sorted((runs / 'v1' / args.train / 'reward_audit').glob('*.jsonl')):
        for batch in load_jsonl(path):
            for row in batch['records']:
                groups[row['task_id']].append(float(row['score']))
    categories = Counter()
    for scores in groups.values():
        category = ('unexpected_count' if len(scores) != 4 else
                    'all_zero' if max(scores) == 0 else
                    'all_one' if min(scores) == 1 else
                    'varying' if max(scores) > min(scores) else 'same_partial')
        categories[category] += 1
    lines = ['=== V1 OFFLINE REVIEW | no inference ===',
             f'Dev tasks={len(after)} | fixed ID/question/gold alignment checked',
             f'Wrong->right={counts[0,1]} | right->wrong={counts[1,0]}',
             f'Both right={counts[1,1]} | both wrong={counts[0,0]}',
             f'V1 nonexact={len(wrong)} | gold chunk coverage={dict(coverage)}',
             f'V1 repeated-query episodes={repeated}/{len(after)}',
             f'Train reward groups={len(groups)} | {dict(categories)}',
             'Coverage is diagnostic, not a failure-cause label.',
             'Train categories describe historical samples, not final-policy difficulty.',
             'Dev is for diagnosis only; V2 training examples must come from train.']
    text = '\n'.join(lines) + '\n'
    (folders[1] / 'v2_review_summary.txt').write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
