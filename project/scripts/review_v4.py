"""Short V4 paired review of saved outputs; no model calls or rescoring."""
import argparse
from collections import Counter
import json
import re
from project.common import load_jsonl, resource_path


def keyed(rows, key):
    result = {str(r[key]): r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate task IDs')
    return result


def paired(before, after):
    if set(before) != set(after):
        raise ValueError('Paired task IDs differ')
    changes = Counter()
    regressions = Counter()
    for task_id in before:
        a, b = before[task_id], after[task_id]
        was, now = a['em'] == 1, b['em'] == 1
        changes[{(False, True): 'wrong->right', (True, False): 'right->wrong',
                 (True, True): 'both_right', (False, False): 'both_wrong'}[was, now]] += 1
        if was and not now:
            regressions[b['status']] += 1
    return dict(changes), dict(regressions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-run', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.eval_run):
        parser.error('Invalid run ID')
    runs = resource_path('AGENTIC_RUNS_DIR')
    out = runs / 'v4_eval' / args.eval_run
    report = json.loads((out / 'report.json').read_text())
    if report['status'] != 'complete':
        raise ValueError('Use a completed V4 development evaluation')
    train_id = report['train_run']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', train_id):
        raise ValueError('Invalid training run ID')
    train = runs / 'v4' / train_id
    step = int(report['selected_step'])
    natural = keyed(load_jsonl(out / 'natural.jsonl'), 'id')
    hard = keyed(load_jsonl(out / 'hard.jsonl'), 'id')
    for task_id, row in hard.items():
        if any(row[k] != natural[task_id][k] for k in ('question', 'gold')):
            raise ValueError('Paired question/gold changed')
    changes, regressions = paired({k: natural[k] for k in hard}, hard)
    lines = ['=== V4 SAVED-TRACE REVIEW | no inference / no rescoring ===',
             f'Run: {args.eval_run} | selected step={step}',
             f'Natural -> hard, same {len(hard)} IDs: {changes}',
             f'Hard regressions by status: {regressions}']
    selected_rows = load_jsonl(train / 'validation' / f'{step}.jsonl')
    for group, independent in (('natural', natural), ('hard', hard)):
        inside = keyed([r for r in selected_rows if r['eval_group'] == group], 'task_id')
        changes, regressions = paired(inside, independent)
        lines.append(f'Trainer -> independent {group}: {changes}')
        lines.append(f'  regressions by status: {regressions}')
    lines.append('Trainer checkpoints: step | natural EM/F1 | failures | matched EM | hard EM')
    for row in load_jsonl(train / 'validation_metrics.jsonl'):
        g = row['groups']
        lines.append(f"{row['step']}: {g['natural']['em']:.2%}/{g['natural']['f1']:.2%} | "
                     f"{g['natural']['protocol_failures']} | {g['natural_matched']['em']:.2%} | {g['hard']['em']:.2%}")
    lines.append('Costs/protocol differences are diagnostic; they do not identify a sole cause.')
    text = '\n'.join(lines)
    (out / 'review_summary.txt').write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
