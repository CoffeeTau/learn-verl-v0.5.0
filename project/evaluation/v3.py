"""Evaluate one final V3 system with frozen V2 weights and the adapted E5 index."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from project.common import resource_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v2-eval', default='dev_20261007T024032Z_0bed4f')
    parser.add_argument('--baseline-run', default='dev_20261006T043535Z_7da4f2')
    parser.add_argument('--retriever-run')
    args = parser.parse_args()
    runs = resource_path('AGENTIC_RUNS_DIR')
    if args.retriever_run is None:
        latest = json.loads((runs / 'v3_retriever/latest_run.json').read_text())
        if latest['status'] != 'passed':
            raise ValueError('Latest V3 retriever run has not passed')
        args.retriever_run = latest['run_id']
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', value) for value in vars(args).values()):
        parser.error('Invalid run ID')
    comparison = runs / 'v2_eval' / args.v2_eval
    manifest = json.loads((comparison / 'manifest.json').read_text())
    model = Path(manifest['policy_model']['path'])
    return subprocess.call([sys.executable, '-u', '-m', 'project.evaluation.v0', '--protocol', 'v3',
                            '--model-path', str(model), '--baseline-run', str(runs / 'v0' / args.baseline_run),
                            '--compare-run', str(comparison), '--retriever-run', str(runs / 'v3_retriever' / args.retriever_run)])


if __name__ == '__main__':
    raise SystemExit(main())
