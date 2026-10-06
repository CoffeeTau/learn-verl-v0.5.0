"""Export the final V1 checkpoint with veRL and evaluate using frozen V0 settings."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from uuid import uuid4
from project.common import resource_path, save_json, sha256, model_inventory
from project.evaluation.v0 import validate_baseline


def export_model(run, step):
    checkpoint = run / 'checkpoints' / f'global_step_{step}' / 'actor'
    fsdp = json.loads((checkpoint / 'fsdp_config.json').read_text())
    world = int(fsdp['world_size'])
    shards = [checkpoint / f'model_world_size_{world}_rank_{rank}.pt' for rank in range(world)]
    if any(not p.is_file() or p.stat().st_size == 0 for p in shards):
        raise ValueError('Missing/empty model shard')
    source = {'checkpoint': str(checkpoint), 'step': step, 'world_size': world,
              'shards': {p.name: {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in shards},
              'config_sha256': sha256(checkpoint / 'huggingface/config.json')}
    target = run / f'hf_step_{step}'
    if target.exists():
        saved = json.loads((target / 'export_manifest.json').read_text())
        if saved['source'] != source or saved['model_inventory'] != export_inventory(target):
            raise ValueError('Existing export differs from checkpoint/inventory')
        print(f'EXPORT | REUSED | step={step}', flush=True)
        return target
    temporary = run / f'.hf_step_{step}_{uuid4().hex[:6]}'
    log_path = run / f'export_step_{step}.log'
    print(f'Exporting {world} FSDP shards on CPU; log: {log_path.name}', flush=True)
    with log_path.open('w') as log:
        result = subprocess.run([sys.executable, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
                                 '--local_dir', str(checkpoint), '--target_dir', str(temporary)],
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Export failed; see {log_path}')
    if not (temporary / 'config.json').exists() or not list(temporary.glob('*.safetensors')):
        raise ValueError('Export did not produce HF configuration and safetensors')
    temporary.rename(target)
    # Record inventory before adding this manifest (model_inventory ignores it below).
    save_json(target / 'export_manifest.json', {'source': source, 'model_inventory': export_inventory(target)})
    print(f'EXPORT | PASSED | step={step} | {target.name}', flush=True)
    return target


def export_inventory(path):
    inventory = model_inventory(path)
    inventory['files'].pop('export_manifest.json', None)
    return inventory


def main(version="v1"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-run', required=version == 'v1')
    parser.add_argument('--baseline-run', default='dev_20261006T043535Z_7da4f2')
    if version == "v2":
        parser.add_argument("--v1-eval", default="dev_20261006T151952Z_111fcb")
    args = parser.parse_args()
    if args.train_run is None:
        args.train_run = json.loads((resource_path("AGENTIC_RUNS_DIR") / version / "latest_run.json").read_text())["run_id"]
    for name in (args.train_run, args.baseline_run, getattr(args, "v1_eval", "valid")):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            parser.error('Invalid run ID')
    runs = resource_path('AGENTIC_RUNS_DIR')
    run = runs / version / args.train_run
    baseline = runs / 'v0' / args.baseline_run
    report = json.loads((run / 'report.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    if report['status'] != 'PASSED' or manifest['mode'] != 'main':
        raise ValueError('Use a completed main training run')
    step = int(manifest['steps'])
    if int((run / 'checkpoints/latest_checkpointed_iteration.txt').read_text()) != step:
        raise ValueError('Final checkpoint marker differs from training budget')
    old = json.loads((baseline / 'manifest.json').read_text())
    old_report = json.loads((baseline / 'report.json').read_text())
    data = resource_path('AGENTIC_PROCESSED_DATA_DIR')
    dataset = json.loads((data / 'manifest.json').read_text())
    if old_report['status'] != 'passed' or old.get('limit') is not None:
        raise ValueError('V0 baseline must cover the complete dev split')
    validate_baseline(old, old_report, dataset, dataset['sizes']['dev'])
    if manifest['data']['source_manifest_sha256'] != sha256(data / 'manifest.json'):
        raise ValueError('Training dataset manifest changed')
    model = export_model(run, step)
    print(f'Evaluating {version.upper()} on full dev with clean retrieval and fixed resource budgets.', flush=True)
    extra_args = []
    if version == "v2":
        extra_args = ["--protocol", "v2", "--compare-run", str(runs / "v1_eval" / args.v1_eval)]
    return subprocess.call([sys.executable, '-u', '-m', 'project.evaluation.v0',
                            '--model-path', str(model), '--baseline-run', str(baseline)] + extra_args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'V1 EVAL | FAILED | {type(exc).__name__}: {exc}', flush=True)
        raise SystemExit(1)
