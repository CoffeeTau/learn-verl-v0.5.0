"""V4: adapt the V2 policy to frozen V3 E5, with the original V2 conversation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid
from project.common import (code_info, load_jsonl, model_inventory, resource_path,
                            save_json, sha256, signature)
from project.training.hard_episodes import build_plan, prepare_plan
from project.training.train import build_config, print_progress
from project.training.summarize import summarize


def read(path):
    return json.loads(Path(path).read_text())


def validate_retriever(run, source, corpus):
    dataset = read(source / 'manifest.json')
    report, manifest = read(run / 'report.json'), read(run / 'manifest.json')
    model, index = run / 'model', run / 'index'
    inventory, indexed = model_inventory(model), read(index / 'manifest.json')
    if report['status'] != 'passed' or report['model_inventory'] != inventory:
        raise ValueError('V3 retriever training/model inventory differs')
    if manifest['pairs_manifest']['source_manifest_sha256'] != sha256(source / 'manifest.json'):
        raise ValueError('V3 retriever used a different frozen dataset')
    if (indexed['status'] != 'passed' or indexed['model_inventory_signature'] != signature(inventory)
            or indexed['corpus_sha256'] != dataset['corpus_sha256']
            or sha256(corpus / 'corpus.jsonl') != dataset['corpus_sha256']
            or sha256(index / 'embeddings.npy') != indexed['embeddings_sha256']):
        raise ValueError('V3 retriever/index/corpus mismatch')
    return {'run': str(run), 'model': inventory, 'index': indexed,
            'manifest_sha256': sha256(run / 'manifest.json')}


def prepare_validation(out, data_dir, smoke):
    import pyarrow as pa
    import pyarrow.parquet as pq
    source, corpus = resource_path('AGENTIC_PROCESSED_DATA_DIR'), resource_path('AGENTIC_CORPUS_DIR')
    tasks = load_jsonl(source / 'dev.jsonl')
    if smoke:
        tasks = tasks[:8]
    labels = {r['id']: r for r in load_jsonl(source / 'dev.labels.jsonl')}
    plan = build_plan(tasks, labels, load_jsonl(corpus / 'corpus.jsonl'))
    save_json(out / 'dev_hard_plan.json', plan)
    hard_ids = {r['task_id'] for r in plan.values()}
    records = pq.read_table(data_dir / 'dev.parquet').to_pylist()
    ids = {r['id'] for r in tasks}
    records = [r for r in records if r['extra_info']['task_id'] in ids]
    result = []
    for row in records:
        for group in ('natural', 'hard'):
            if group == 'hard' and row['extra_info']['task_id'] not in hard_ids:
                continue
            result.append({**row, 'data_source': 'v4_' + group,
                           'agent_name': 'agentic_correction_dev_hard' if group == 'hard' else 'agentic_correction_eval',
                           'extra_info': {**row['extra_info'], 'eval_group': group}})
    pq.write_table(pa.Table.from_pylist(result), data_dir / 'dev.parquet')
    expected = {'natural': sorted(ids), 'hard': sorted(hard_ids)}
    save_json(out / 'validation_tasks.json', expected)
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'main'), default='smoke')
    parser.add_argument('--gpus', type=int, choices=(2, 4, 8), default=8)
    parser.add_argument('--init-run', default='main_20261006T164842Z_548d39')
    parser.add_argument('--retriever-run', default='main_20261007T042356Z_357492')
    args = parser.parse_args()
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', v) for v in (args.init_run, args.retriever_run)):
        parser.error('Invalid run ID')
    smoke = args.mode == 'smoke'
    steps = 2 if smoke else 125
    runs = resource_path('AGENTIC_RUNS_DIR')
    run_id = args.mode + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:6]
    out = runs / 'v4' / run_id
    out.mkdir(parents=True)
    code = 1
    try:
        from project.training.prepare import prepare
        from project.evaluation.v1 import export_model
        from torch.utils.tensorboard import SummaryWriter  # fail early if absent
        import torch
        from omegaconf import OmegaConf
        if torch.cuda.device_count() < args.gpus:
            raise ValueError(f'Need {args.gpus} visible GPUs')
        source, corpus = resource_path('AGENTIC_PROCESSED_DATA_DIR'), resource_path('AGENTIC_CORPUS_DIR')
        retriever_run = runs / 'v3_retriever' / args.retriever_run
        retrieval = validate_retriever(retriever_run, source, corpus)
        parent = runs / 'v2' / args.init_run
        parent_report, parent_manifest = read(parent / 'report.json'), read(parent / 'manifest.json')
        if parent_report['status'] != 'PASSED' or parent_manifest['mode'] != 'main' or parent_manifest['version'] != 'v2':
            raise ValueError('V4 requires a passed V2 main training run')
        if parent_manifest['data']['source_manifest_sha256'] != sha256(source / 'manifest.json'):
            raise ValueError('V2 training dataset changed')
        policy = export_model(parent, int(parent_manifest['steps']))
        data_dir, prepared = prepare(version='v4', output_dir=out / 'data')
        if prepared['task'] != parent_manifest['data']['task']:
            raise ValueError('Task budgets/config changed since V2')
        hard = prepare_plan(source, corpus, out / 'hard_plan.json')
        expected = prepare_validation(out, data_dir, smoke)
        prepared['files'] = {name: sha256(data_dir / name) for name in ('train.parquet', 'dev.parquet')}
        prepared.update(version='v4', hard_episodes=hard, validation_sizes={k: len(v) for k, v in expected.items()})
        save_json(data_dir / 'manifest.json', prepared)
        config = build_config(out, data_dir, prepared, args.gpus, steps, smoke, 'v2', policy, overrides={
            'trainer.custom_trainer': {'path': 'pkg://project.training.v4_trainer', 'name': 'V4Trainer'},
            'trainer.tensorboard_dir': str(resource_path('AGENTIC_ROOT') / 'tensorboard' / ('v4_' + run_id)),
            'trainer.val_before_train': True, 'trainer.test_freq': 2 if smoke else 25,
            'trainer.validation_data_dir': str(out / 'validation'),
            'trainer.max_actor_ckpt_to_keep': 2 if smoke else 5,
            'reward_model.reward_kwargs': {'version': 'v2', 'monitoring': True, 'audit_dir': str(out / 'reward_audit'),
                                          'max_searches': prepared['task']['max_searches'],
                                          'max_new_tokens': prepared['task']['max_new_tokens']},
            'agentic': {'task': prepared['task'], 'retriever_name': 'retriever_' + run_id,
                        'corpus': str(corpus), 'corpus_sha256': prepared['corpus_sha256'],
                        'index': str(retriever_run / 'index'), 'retriever': str(retriever_run / 'model'),
                        'hard_plan': str(out / 'hard_plan.json'), 'dev_hard_plan': str(out / 'dev_hard_plan.json'),
                        'retrieval_audit': str(out / 'retrieval_audit.jsonl'), 'v4_monitor': True}})
        loop_path = out / 'agent_loop.yaml'
        loops = OmegaConf.load(loop_path)
        loops.append({'name': 'agentic_correction_dev_hard',
                      '_target_': 'project.training.agent_loop.CorrectionAgentLoop', 'perturb': 'dev'})
        OmegaConf.save(loops, loop_path)
        repo = Path(__file__).resolve().parents[2]
        core = ['verl/trainer/main_ppo.py', 'verl/trainer/ppo/ray_trainer.py', 'verl/utils/tracking.py',
                'verl/workers/actor/dp_actor.py', 'verl/trainer/ppo/reward.py']
        save_json(out / 'manifest.json', {'version': 'v4', 'mode': args.mode, 'steps': steps,
                  'code': code_info(), 'core_sha256': {p: sha256(repo / p) for p in core},
                  'data': prepared, 'policy': model_inventory(policy), 'parent_run': str(parent),
                  'tokenizer': model_inventory(resource_path('AGENTIC_MODEL_DIR')),
                  'retrieval': retrieval, 'protocol': 'v2', 'state_ledger': False,
                  'dev_hard_plan_sha256': sha256(out / 'dev_hard_plan.json'),
                  'validation_tasks_sha256': sha256(out / 'validation_tasks.json'),
                  'reward': 'strict canonical terminal F1; KL loss .001; reference=initial V2'})
        print(f'=== V4 PREPARE | PASSED ===\nPolicy: V2 | retriever/index: frozen V3 | input: V2 (no ledger)\n'
              f'Train: 2000 | hard candidates: {sum(hard["modes"].values())} | '
              f'dev natural={len(expected["natural"])} hard={len(expected["hard"])}\n'
              f'{steps} updates | GPUs={args.gpus} | 16 questions x 4 samples\n'
              f'Run: {run_id}\nLog: runtime/runs/v4/{run_id}/train.log', flush=True)
        env = dict(os.environ, VLLM_USE_V1='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
                   TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1')
        with (out / 'train.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-u', '-m', 'project.training.worker', '--config', str(out / 'config.yaml')],
                                       stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                while process.poll() is None:
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        print_progress(out, steps, 'v4')
                code = process.returncode
            finally:
                if process.poll() is None:
                    from project.evaluation.acceptance import stop_process_group
                    stop_process_group(process)
        if code == 0:
            selection = read(out / 'selection.json')
            measured = [int(r['step']) for r in load_jsonl(out / 'validation_metrics.jsonl')]
            if measured != ([0, 2] if smoke else [0, 25, 50, 75, 100, 125]):
                raise ValueError(f'Incomplete validation schedule: {measured}')
            if int((out / 'checkpoints/latest_checkpointed_iteration.txt').read_text()) != steps:
                raise ValueError('Final checkpoint marker differs')
            for step in {steps, selection['selected_step']} - {0}:
                checkpoint = out / 'checkpoints' / f'global_step_{step}' / 'actor'
                world = int(read(checkpoint / 'fsdp_config.json')['world_size'])
                if world != args.gpus or any(not (checkpoint / f'model_world_size_{world}_rank_{rank}.pt').is_file()
                    or (checkpoint / f'model_world_size_{world}_rank_{rank}.pt').stat().st_size == 0 for rank in range(world)):
                    raise ValueError('Selected/final checkpoint shards missing or empty')
    except Exception:
        code = 1
        import traceback
        with (out / 'train.log').open('a') as log:
            traceback.print_exc(file=log)
    passed = summarize(out, steps, code, version='v4')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
