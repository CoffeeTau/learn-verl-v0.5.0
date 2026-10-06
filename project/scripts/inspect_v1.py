"""Read-only V1 log inspection: no model loading or training."""
import argparse
from collections import Counter
import json
import re
from project.common import load_jsonl, resource_path

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        parser.error('Invalid run ID')
    out = resource_path('AGENTIC_RUNS_DIR') / 'v1' / args.run_id
    report = json.loads((out / 'report.json').read_text())
    raw = (out / 'train.log').read_text(errors='replace')
    log = ANSI.sub('', raw)
    lines = ['=== V1 SAVED-LOG REVIEW (no training) ===', f'Run: {args.run_id}',
             f"Exit={report['exit_code']} | expected_steps={report['expected_steps']} | summary_steps={list(report['steps'])}"]
    markers = Counter(re.findall(r'(?<![\w/])step\s*:\s*(\d+)\b', log))
    lines.append(f'Console step markers: {dict(markers)}')
    lines.append(f"Grad mentions: {log.count('actor/grad_norm')} | delta mentions: {log.count('actor/probe_parameter_delta_max')}")
    # Every discovered console step is bounded by the next marker, even if Ray
    # joined lines or inserted carriage returns. Show raw values, not a pass verdict.
    matches = list(re.finditer(r'(?<![\w/])step\s*:\s*(\d+)\b', log))
    for i, match in list(enumerate(matches))[-4:]:
        stop = matches[i + 1].start() if i + 1 < len(matches) else len(log)
        block = log[match.end():stop]
        fields = []
        for key in ('actor/grad_norm', 'actor/probe_parameter_delta_max', 'training/global_step'):
            found = re.search(re.escape(key) + r'\s*:\s*([^\r\n]+)', block)
            value = found[1].split(' - ')[0][:90] if found else 'MISSING'
            fields.append(f"{key.split('/')[-1]}={value}")
        lines.append(f"Raw step {match[1]}: " + ' | '.join(fields))
    for checkpoint in sorted((out / 'checkpoints').glob('global_step_*')):
        model_files = list((checkpoint / 'actor').glob('model_world_size_*_rank_*.pt'))
        lines.append(f'{checkpoint.name}: model_shards={len(model_files)} | all_nonempty={bool(model_files) and all(p.stat().st_size > 0 for p in model_files)}')
    latest = out / 'checkpoints/latest_checkpointed_iteration.txt'
    lines.append(f'Checkpoint marker: {latest.read_text().strip() if latest.exists() else "MISSING"}')
    batches = []
    for path in sorted((out / 'reward_audit').glob('*.jsonl')):
        batches.extend(load_jsonl(path))
    lines.append('Reward calls (file order): ' + ', '.join(
        f"{b['episodes']} episodes/{b['varying_groups']} varying" for b in batches[:8]))
    lines.append(f"Final validation log present: {'Final validation metrics:' in log}")
    # End of file helps distinguish missing Ray-forwarded output from a parser issue.
    tail = [line.strip() for line in log.splitlines() if line.strip()][-3:]
    lines.append('Log tail (clipped):')
    lines.extend(line[:160] for line in tail)
    text = '\n'.join(lines) + '\n'
    (out / 'review_summary.txt').write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
