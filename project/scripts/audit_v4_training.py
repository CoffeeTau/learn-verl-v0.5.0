"""Offline training reward-group audit; no inference, rescoring or evidence reconstruction."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import textwrap

from project.common import resource_path, save_json, save_jsonl


def classify(values):
    if min(values) != max(values):
        return 'varied'
    return 'all_zero' if values[0] == 0 else ('all_full_f1' if values[0] == 1 else 'same_partial_f1')


def audit(run, group_size=4):
    groups, warnings, inputs = [], [], []
    files = sorted((run / 'reward_audit').glob('rewards_*.jsonl'))
    if not files:
        raise ValueError('No training reward audit files found')
    for path in files:  # Deliberately excludes reward_audit/validation.
        inputs.append(str(path.relative_to(run)))
        with path.open(encoding='utf-8') as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                provenance = f'{path.name}:{line_no}'
                try:
                    batch = json.loads(line)
                    grouped = defaultdict(list)
                    for row in batch['records']:
                        if row.get('eval_group', 'train') != 'train':
                            raise ValueError('Non-training record in training audit')
                        for key in ('score', 'em'):
                            value = row[key]
                            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                                raise ValueError(f'Invalid {key}')
                        if row['em'] not in (0, 1):
                            raise ValueError('EM is not binary')
                        if not isinstance(row['model_outputs'], list) or not all(isinstance(x, str) for x in row['model_outputs']):
                            raise ValueError('Invalid model_outputs')
                        grouped[str(row['task_id'])].append(row)
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f'{provenance}: {exc}') from exc
                varied = sum(classify([r.get('training_reward', r['score']) for r in rows]) == 'varied' for rows in grouped.values())
                for key, actual in [('episodes', len(batch['records'])), ('groups', len(grouped)), ('varying_groups', varied)]:
                    if batch.get(key) != actual:
                        warnings.append(f'{provenance}: {key} header={batch.get(key)} reconstructed={actual}')
                for task_id, rows in grouped.items():
                    scores = [r.get('training_reward', r['score']) for r in rows]
                    correct = sum(r['em'] == 1 for r in rows)
                    complete = len(rows) == group_size
                    if not complete:
                        warnings.append(f'{provenance}/{task_id}: group size {len(rows)}, expected {group_size}')
                    groups.append(dict(batch=provenance, task_id=task_id, complete=complete,
                                       reward_type=classify(scores),
                                       em_type='all_wrong' if correct == 0 else ('all_right' if correct == len(rows) else 'mixed'),
                                       records=rows))
    records = [r for g in groups for r in g['records']]
    complete = [g for g in groups if g['complete']]
    report = dict(episodes=len(records), groups=len(groups), complete_groups=len(complete),
                  reward_groups=dict(Counter(g['reward_type'] for g in complete)),
                  em_groups=dict(Counter(g['em_type'] for g in complete)),
                  reward_em_cross=dict(Counter(g['reward_type'] + '/' + g['em_type'] for g in complete)),
                  statuses=dict(Counter(r['status'] for r in records)),
                  zero_score_statuses=dict(Counter(r['status'] for r in records if r['score'] == 0)),
                  reward_basis='training_reward if present; otherwise canonical F1',
                  observation_records=sum('observations' in r for r in records),
                  canonical_f1_groups=dict(Counter(classify([r['score'] for r in g['records']]) for g in complete)),
                  full_f1_but_em_zero=sum(r['score'] == 1 and r['em'] == 0 for r in records),
                  inputs=inputs, warnings=warnings)
    previous = run / 'report.json'
    if previous.exists():
        saved = json.loads(previous.read_text())
        for key, actual in [('groups', len(groups)), ('varying_groups', sum(g['reward_type'] == 'varied' for g in groups)), ('statuses', report['statuses'])]:
            if key in saved and saved[key] != actual:
                warnings.append(f'report.json {key} differs: saved={saved[key]}, reconstructed={actual}')
    return report, groups


def candidates(groups, limit):
    """Sampling buckets are review leads, never proof of successful recovery."""
    buckets = defaultdict(list)
    for group in groups:
        if not group['complete']:
            continue
        rows = group['records']
        tags = [group['reward_type']]
        if group['em_type'] == 'mixed':
            tags.append('mixed_em')
        if any(r['em'] == 1 and r.get('searches', 0) >= 3 for r in rows):
            tags.append('correct_with_3plus_searches')
        if all(r['score'] == 0 and r['status'] == 'answered' for r in rows):
            tags.append('all_zero_but_answered')
        for tag in tags:
            if len(buckets[tag]) < limit:
                buckets[tag].append(group)
    return dict(buckets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-run', required=True)
    parser.add_argument('--stage', choices=['v4', 'v4_base', 'v4_base_evidence'], default='v4_base')
    parser.add_argument('--group-size', type=int, default=4)
    parser.add_argument('--examples', type=int, default=2)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.train_run) or args.group_size < 2 or args.examples < 1:
        parser.error('Invalid run, group size or example count')
    run = resource_path('AGENTIC_RUNS_DIR') / args.stage / args.train_run
    report, groups = audit(run, args.group_size)
    out = run / 'offline_audit'
    out.mkdir(exist_ok=True)
    limitations = ('仅统计已记录的终局 F1/EM，不重新评分。组按日志文件+行号+题号区分；不推断训练 step。'
                   f"含逐轮 observation 的记录数={report['observation_records']}；没有 observation 的旧记录不能可靠重建证据。"
                   '因此无法据此确认成功纠错、证据充分性或证据未被使用。模型自述 known 不等于事实。'
                   '候选采用每类日志顺序前几个组，不代表发生率；三次以上搜索且答对不等于成功恢复。')
    report['limitations'] = limitations
    save_json(out / 'report.json', report)
    save_jsonl(out / 'groups.jsonl', groups)
    lines = ['=== OFFLINE TRAINING AUDIT | no inference / no rescoring ===', f'Run: {args.train_run}',
             f"Episodes={report['episodes']} groups={report['groups']} complete={report['complete_groups']}",
             f"Reward groups (actual training reward): {report['reward_groups']}", f"Canonical F1 groups: {report['canonical_f1_groups']}", f"EM groups: {report['em_groups']}",
             f"Reward x EM: {report['reward_em_cross']}",
             f"Zero-score statuses: {report['zero_score_statuses']}",
             f"Full F1 but EM=0: {report['full_f1_but_em_zero']}",
             f"Integrity warnings: {len(report['warnings'])}"]
    summary = '\n'.join(lines + report['warnings'] + ['', limitations])
    (out / 'summary.txt').write_text(summary + '\n', encoding='utf-8')
    traces = [summary, '\n候选输出及已保存的 observations（如有）']
    seen = set()
    for bucket, items in candidates(groups, args.examples).items():
        traces.append(f'\n=== {bucket} ===')
        for group in items:
            identity = (group['batch'], group['task_id'])
            traces.append(f"\n{identity} reward={group['reward_type']} EM={group['em_type']}")
            if identity in seen:
                traces.append('同一组已在前面展开。')
                continue
            seen.add(identity)
            for i, row in enumerate(group['records'], 1):
                traces.append(f"Sample {i}: F1={row['score']} EM={row['em']} status={row['status']} searches={row.get('searches')}")
                for turn, output in enumerate(row['model_outputs'], 1):
                    traces.append(f'Turn {turn}:\n{output}')
                    if turn <= len(row.get('observations', [])):
                        traces.append('Observation:\n' + row['observations'][turn - 1])
    (out / 'candidate_traces.txt').write_text('\n'.join(
        textwrap.fill(line, width=100, replace_whitespace=False) if line else ''
        for block in traces for line in block.split('\n')) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'TXT: {out / "candidate_traces.txt"}')
    print(f"Records with observations: {report['observation_records']}; recovery needs evidence review.")


if __name__ == '__main__':
    main()
