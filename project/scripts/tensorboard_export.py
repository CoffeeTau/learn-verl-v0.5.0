"""Convert one completed training run and its matched dev evaluation to event files."""
import argparse
import json
import math
from pathlib import Path
import re
from uuid import uuid4
from project.common import load_jsonl, resource_path, save_json, sha256


def points(training, before, after, final_step):
    """Keep numeric training scalars; dev has only the two actually measured points."""
    for row in training:
        for tag, value in row['metrics'].items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                yield tag, float(value), int(row['step'])
        value = row['metrics'].get('critic/score/mean')
        if isinstance(value, (int, float)) and math.isfinite(value):
            yield 'train/answer_f1', float(value), int(row['step'])
    for step, report in ((0, before), (final_step, after)):
        for metric in ('em_percent', 'f1_percent', 'mean_searches', 'mean_tokens', 'mean_seconds'):
            yield 'eval/' + metric, float(report['metrics'][metric]), step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-run', required=True)
    parser.add_argument('--eval-run', required=True)
    args = parser.parse_args()
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', value) for value in (args.train_run, args.eval_run)):
        parser.error('Invalid run ID')
    runs = resource_path('AGENTIC_RUNS_DIR')
    train = runs / 'v1' / args.train_run
    evaluation = runs / 'v1_eval' / args.eval_run
    train_report = json.loads((train / 'report.json').read_text())
    after = json.loads((evaluation / 'report.json').read_text())
    manifest = json.loads((evaluation / 'manifest.json').read_text())
    step = int(train_report['expected_steps'])
    if train_report['status'] != 'PASSED' or after['status'] != 'passed':
        raise ValueError('Only completed successful runs may be imported')
    if Path(manifest['policy_model']['path']).resolve() != (train / f'hf_step_{step}').resolve():
        raise ValueError('Evaluation is not from the specified final checkpoint')
    baseline = Path(after['baseline_run'])
    before = json.loads((baseline / 'report.json').read_text())
    if before['status'] != 'passed':
        raise ValueError('Baseline evaluation is incomplete')
    training = load_jsonl(train / 'metrics.jsonl')
    source_files = [train / 'metrics.jsonl', train / 'report.json', evaluation / 'report.json',
                    evaluation / 'manifest.json', baseline / 'report.json']
    provenance = {str(path): sha256(path) for path in source_files}
    root = resource_path('AGENTIC_ROOT') / 'tensorboard'
    root.mkdir(parents=True, exist_ok=True)
    target = root / ('v1_' + args.train_run)
    if target.exists():
        saved = json.loads((target / 'import_manifest.json').read_text())
        if saved['source_sha256'] != provenance or not list(target.glob('events.out.tfevents.*')):
            raise ValueError('Existing import differs from inputs; do not mix records')
        print(f'TENSORBOARD IMPORT | REUSED\nRun: {target.name}\nEvents already available.')
        return
    from torch.utils.tensorboard import SummaryWriter
    # Stage outside the watched logdir; expose only a completed import.
    temporary = root.parent / ('.tb_import_' + uuid4().hex[:8])
    writer = SummaryWriter(log_dir=str(temporary))
    try:
        for tag, value, global_step in points(training, before, after, step):
            writer.add_scalar(tag, value, global_step)
        writer.flush()
    finally:
        writer.close()
    save_json(temporary / 'import_manifest.json', {'source_sha256': provenance,
              'note': 'Historical import: wall time is import time; use Step axis. Dev measured at 0 and final only.'})
    temporary.rename(target)
    print(f'TENSORBOARD IMPORT | PASSED\nRun: {target.name}\n'
          f'Training records: {len(training)} | dev measured at steps 0 and {step}\n'
          'Use Step axis; eval lines between endpoints are not intermediate measurements.\n'
          'Next: serve runtime/tensorboard and forward port 6006 in VS Code.')


if __name__ == '__main__':
    main()
