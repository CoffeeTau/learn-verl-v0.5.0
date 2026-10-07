"""Read saved dev cases across V0-V4; never infer or rescore."""
import argparse
import json
import re
from project.common import load_jsonl, resource_path
from project.scripts.review_v4 import keyed

HISTORY = {
    'V0': 'v0/dev_20261006T043535Z_7da4f2',
    'V1': 'v1_eval/dev_20261006T151952Z_111fcb',
    'V2': 'v2_eval/dev_20261007T024032Z_0bed4f',
    'V3': 'v3_eval/dev_20261007T045730Z_081068',
}
ANCHORS = [('grandfather', '2f761f100bb011ebab90acde48001122'),
           ('bands', '1f53f012087711ebbd67ac1f6bf848b6'),
           ('nationality', 'ec9fe92f08e311ebbda4ac1f6bf848b6')]


def compact(value, limit=130):
    text = ' '.join(str(value).split())
    return text if len(text) <= limit else text[:limit] + ' [clipped]'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-run', required=True)
    parser.add_argument('--case-id', help='Print one full case instead of the short overview')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.eval_run):
        parser.error('Invalid run ID')
    runs = resource_path('AGENTIC_RUNS_DIR')
    out = runs / 'v4_eval' / args.eval_run
    if json.loads((out / 'report.json').read_text())['status'] != 'complete':
        raise ValueError('Use a completed development run')
    versions = {}
    for version, path in HISTORY.items():
        file = runs / path / 'trajectories.jsonl'
        versions[version] = keyed(load_jsonl(file), 'id') if file.exists() else {}
    versions['V4'] = keyed(load_jsonl(out / 'natural.jsonl'), 'id')
    versions['V4-hard'] = keyed(load_jsonl(out / 'hard.jsonl'), 'id')
    natural, hard = versions['V4'], versions['V4-hard']
    for version, rows in versions.items():
        for task_id, row in rows.items():
            if task_id in natural and any(row[k] != natural[task_id][k] for k in ('question', 'gold')):
                raise ValueError(f'Question/gold mismatch: {version} {task_id}')
    chosen = [(name, key) for name, key in ANCHORS if key in natural]
    # Fixed ID order, no cherry-picking by answer text; these are inspection samples.
    for name, candidates in (
        ('hard-regression', [k for k, r in hard.items() if natural[k]['em'] == 1
                            and r['em'] != 1 and r['status'] == 'answered'
                            and r.get('perturbation', {}).get('applied')]),
        ('judge-conflict', [k for k, r in natural.items() if r['status'] == 'judge_action_mismatch']),
    ):
        candidates = sorted(set(candidates) - {k for _, k in chosen})
        if candidates:
            chosen.append((name, candidates[0]))
    if args.case_id:
        if args.case_id not in natural:
            parser.error('Case ID is absent from this development run')
        chosen = [('requested', args.case_id)]
    short = ['=== V4 CASE REVIEW | saved dev only; no rescoring ===']
    document = ['# V0–V4 开发集案例原始证据', '',
                '固定历史案例及按 ID 选出的诊断样本，不代表总体比例。缺失历史文件标记 missing。',
                '下列输出与 hits 为完整保存文本；模型判断不是事实标签。', '']
    for name, key in chosen:
        row = natural[key]
        short.append(f'{name} | {key}')
        short.append('Q: ' + compact(row['question']))
        document.extend([f'## {name}: {key}', '', 'Question: ' + row['question'],
                         'Gold: ' + str(row['gold']), ''])
        for version, rows in versions.items():
            r = rows.get(key)
            if r is None:
                short.append(f'  {version}: missing')
                document.append(f'### {version}: missing\n')
                continue
            line = (f"{version}: EM={r['em']:g} searches={r['searches']} "
                    f"{r['status']} | pred={compact(r['answer'], 90)}")
            short.append('  ' + line)
            document.extend([f'### {version}', '', line, '',
                             'Perturbation: ' + json.dumps(r.get('perturbation', {}), ensure_ascii=False), ''])
            for i, step in enumerate(r['steps'], 1):
                document.extend([f'#### Turn {i}', '', 'Output:', '```text', step['output'], '```', ''])
                for hit in step.get('hits', []):
                    document.extend([f"Evidence [{hit['id']}] {hit['title']}",
                                     '```text', hit['text'], '```', ''])
    target = out / (f'case_{args.case_id}.md' if args.case_id else 'badcase_evidence.md')
    if args.case_id and not re.fullmatch(r'[A-Za-z0-9_-]+', args.case_id):
        parser.error('Invalid case ID')
    target.write_text('\n'.join(document) + '\n', encoding='utf-8')
    if args.case_id:
        print('\n'.join(document))
    else:
        print('\n'.join(short))
    print('Full evidence:', target)


if __name__ == '__main__':
    main()
