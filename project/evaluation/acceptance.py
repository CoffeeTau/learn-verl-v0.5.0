"""Frozen test acceptance: one manifest, four policies, natural and fixed hard tasks."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from uuid import uuid4

from project.common import load_jsonl, model_inventory, resource_path, save_json, sha256, signature
from project.evaluation.v0 import aggregate
from project.evaluation.metrics import score_answer
from project.agent.search import ContextBudgetError, run_episode, SYSTEM_PROMPT as V1_PROMPT
from project.agent.correction import SYSTEM_PROMPT as V2_PROMPT, parse_action
from project.agent.state import SYSTEM_PROMPT as V3_PROMPT, state_messages
from project.training.hard_episodes import build_plan, transform_hits

SOURCES = {
    'V0': 'v0/dev_20261006T043535Z_7da4f2',
    'V1': 'v1_eval/dev_20261006T151952Z_111fcb',
    'V2': 'v2_eval/dev_20261007T024032Z_0bed4f',
    'V3': 'v3_eval/dev_20261007T045730Z_081068',
}
PROMPTS = {'V0': V1_PROMPT, 'V1': V1_PROMPT, 'V2': V2_PROMPT, 'V3': V3_PROMPT}


def read(path):
    return json.loads(Path(path).read_text())


def runtime_code():
    root = Path(__file__).resolve().parents[2]
    names = ['project/evaluation/acceptance.py', 'project/evaluation/v0.py',
             'project/evaluation/metrics.py', 'project/agent/search.py', 'project/agent/correction.py',
             'project/agent/state.py', 'project/retrieval/local.py', 'project/retrieval/e5.py',
             'project/training/hard_episodes.py', 'project/common.py',
             'project/scripts/summarize_smoke.py']
    return {name: sha256(root / name) for name in names}


def test_data():
    data = resource_path('AGENTIC_PROCESSED_DATA_DIR')
    corpus_path = resource_path('AGENTIC_CORPUS_DIR') / 'corpus.jsonl'
    dataset = read(data / 'manifest.json')
    for name in ('test.jsonl', 'test.labels.jsonl'):
        if sha256(data / name) != dataset['files'][name]:
            raise ValueError('Frozen test file changed: ' + name)
    if sha256(corpus_path) != dataset['corpus_sha256']:
        raise ValueError('Frozen corpus changed')
    tasks = load_jsonl(data / 'test.jsonl')
    label_rows = load_jsonl(data / 'test.labels.jsonl')
    labels = {r['id']: r for r in label_rows}
    if (len(tasks) != 300 or len({r['id'] for r in tasks}) != 300
            or len({r['question'] for r in tasks}) != 300 or len(labels) != len(label_rows)
            or set(labels) != {r['id'] for r in tasks}):
        raise ValueError('Expected 300 unique frozen tasks and matching labels')
    return dataset, tasks, labels, load_jsonl(corpus_path)


def freeze(out):
    dataset, tasks, labels, corpus = test_data()
    versions = {}
    runs = resource_path('AGENTIC_RUNS_DIR')
    for version, source in SOURCES.items():
        folder = runs / source
        manifest, report = read(folder / 'manifest.json'), read(folder / 'report.json')
        if (report['status'] != 'passed' or manifest.get('limit') is not None
                or manifest['question_split'] != 'dev'
                or report['metrics']['completed'] != 200 or report['metrics']['planned'] != 200):
            raise ValueError('Source must be a completed full dev run: ' + version)
        if manifest['dataset_manifest'] != dataset or manifest['system_prompt'] != PROMPTS[version]:
            raise ValueError('Source data or prompt differs: ' + version)
        if manifest['enable_thinking'] is not False or manifest['scoring'] != 'canonical_answer_em_f1_no_aliases':
            raise ValueError('Source scoring/thinking differs')
        if version == 'V3' and manifest['history_mode'] != 'v2_history_plus_state_v2':
            raise ValueError('Use repaired V3 history only')
        model = Path(manifest['policy_model']['path'])
        if model_inventory(model) != manifest['policy_model']:
            raise ValueError('Source policy changed: ' + version)
        if version == 'V3':
            retrieval_run = Path(manifest['retriever_training_run'])
            index, retriever = retrieval_run / 'index', retrieval_run / 'model'
        else:
            index, retriever = resource_path('AGENTIC_INDEX_DIR'), resource_path('AGENTIC_RETRIEVER_DIR')
        index_manifest = read(index / 'manifest.json')
        if index_manifest != manifest['index_manifest']:
            raise ValueError('Source index differs: ' + version)
        if (sha256(index / 'embeddings.npy') != index_manifest['embeddings_sha256']
                or signature(model_inventory(retriever)) != index_manifest['model_inventory_signature']):
            raise ValueError('Source retrieval files changed: ' + version)
        versions[version] = {'source_run': source, 'source_manifest_sha256': sha256(folder / 'manifest.json'),
                             'model': str(model), 'model_inventory': manifest['policy_model'],
                             'index': str(index), 'retriever': str(retriever), 'index_manifest': index_manifest,
                             'prompt': PROMPTS[version], 'config': manifest['config']}
    if any(v['config'] != versions['V0']['config'] for v in versions.values()):
        raise ValueError('Resource budgets differ')
    if versions['V3']['model_inventory'] != versions['V2']['model_inventory']:
        raise ValueError('V3 must freeze V2 policy')
    tokenizer = resource_path('AGENTIC_MODEL_DIR')
    if model_inventory(tokenizer) != versions['V0']['model_inventory']:
        raise ValueError('Original tokenizer/model changed')
    # Plan fixed before any test predictions. Gold supports are private environment metadata.
    plan = build_plan(tasks, labels, corpus, seed=42)
    hard_ids = sorted(item['task_id'] for item in plan.values())
    spec = {'schema': 1, 'candidate_selected_on_dev': 'V2', 'dataset': dataset,
            'natural_tasks': 300, 'hard_tasks': len(hard_ids), 'hard_ids': hard_ids,
            'hard_modes': dict(Counter(p['mode'] for p in plan.values())),
            'hard_plan': plan, 'hard_seed': 42, 'versions': versions,
            'tokenizer': str(tokenizer), 'code': runtime_code(),
            'scope': 'natural test plus fixed first-retrieval perturbation; no test-driven tuning'}
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'frozen.json', spec)
    return spec


def verify(spec):
    dataset, tasks, labels, corpus = test_data()
    if (set(spec['versions']) != set(SOURCES) or spec['natural_tasks'] != 300 or spec['hard_tasks'] != 75
            or spec['candidate_selected_on_dev'] != 'V2'
            or spec['hard_ids'] != sorted(p['task_id'] for p in spec['hard_plan'].values())):
        raise ValueError('Frozen acceptance structure changed')
    if dataset != spec['dataset'] or runtime_code() != spec['code']:
        raise ValueError('Frozen data/runtime code changed; do not silently resume different code')
    if build_plan(tasks, labels, corpus, seed=42) != spec['hard_plan']:
        raise ValueError('Frozen hard plan differs')
    if model_inventory(spec['tokenizer']) != spec['versions']['V0']['model_inventory']:
        raise ValueError('Tokenizer changed')
    for version, item in spec['versions'].items():
        if model_inventory(item['model']) != item['model_inventory'] or item['prompt'] != PROMPTS[version]:
            raise ValueError('Policy or prompt changed: ' + version)
        if read(Path(item['index']) / 'manifest.json') != item['index_manifest']:
            raise ValueError('Index manifest changed: ' + version)
        if (sha256(Path(item['index']) / 'embeddings.npy') != item['index_manifest']['embeddings_sha256']
                or signature(model_inventory(item['retriever'])) != item['index_manifest']['model_inventory_signature']):
            raise ValueError('Retrieval files changed: ' + version)
    return tasks, labels, corpus


def episode_search(retriever, specification, paragraphs, audit):
    calls = 0
    def search(question, query, top_k):
        nonlocal calls
        calls += 1
        selected_spec = specification if calls == 1 else None
        candidates = retriever.search(question, query, max(top_k, 20) if selected_spec else top_k)
        hits = candidates[:top_k]
        if selected_spec:
            selected, applied = transform_hits(hits, candidates, selected_spec, paragraphs, top_k)
            audit.update(mode=selected_spec['mode'], applied=applied,
                         original_ids=[h['id'] for h in hits], returned_ids=[h['id'] for h in selected])
            return selected
        return hits
    return search


def worker(out, version):
    spec = read(out / 'frozen.json')
    tasks, labels, corpus = verify(spec)
    item = spec['versions'][version]
    config = item['config']
    folder = out / version
    folder.mkdir(exist_ok=True)
    report = {'status': 'running', 'version': version, 'frozen_sha256': sha256(out / 'frozen.json'), 'groups': {}}
    try:
        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
        from project.retrieval.local import LocalRetriever
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        retriever = LocalRetriever(resource_path('AGENTIC_CORPUS_DIR'), Path(item['index']), Path(item['retriever']))
        tokenizer = AutoTokenizer.from_pretrained(spec['tokenizer'], local_files_only=True)
        llm = LLM(model=item['model'], tokenizer=spec['tokenizer'], dtype='bfloat16', tensor_parallel_size=1,
                  max_model_len=config['max_context_tokens'], gpu_memory_utilization=config['gpu_memory_utilization'],
                  max_num_seqs=1, enforce_eager=True, seed=config['seed'])
        params = SamplingParams(temperature=config['temperature'], max_tokens=config['max_new_tokens'],
                                stop=['</search>', '</answer>'], include_stop_str_in_output=True)
        def generate(messages):
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if len(ids) + config['max_new_tokens'] > config['max_context_tokens']:
                raise ContextBudgetError()
            result = llm.generate([{'prompt_token_ids': ids}], params, use_tqdm=False)[0].outputs[0]
            return dict(text=result.text, finish_reason=result.finish_reason, prompt_tokens=len(ids), output_tokens=len(result.token_ids))
        paragraphs = {row['id']: row['paragraph_id'] for row in corpus}
        for group in ('natural', 'hard'):
            selected = tasks if group == 'natural' else [r for r in tasks if r['id'] in set(spec['hard_ids'])]
            results = []
            with (folder / f'{group}.jsonl').open('w', encoding='utf-8') as handle:
                for number, task in enumerate(selected, 1):
                    audit = {'applied': False}
                    perturb = spec['hard_plan'][task['question']] if group == 'hard' else None
                    result = run_episode(task['question'], generate, episode_search(retriever, perturb, paragraphs, audit), config,
                                         system_prompt=item['prompt'], action_parser=parse_action if version in ('V2', 'V3') else None,
                                         history_builder=state_messages if version == 'V3' else None)
                    gold = labels[task['id']]
                    score = score_answer(result['answer'], gold['answer']) if result['status'] == 'answered' else {'em': 0., 'f1': 0.}
                    result.update(id=task['id'], question=task['question'], gold=gold['answer'], **score, perturbation=audit)
                    handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                    handle.flush()
                    results.append(result)
                    if number % 25 == 0 or number == len(selected):
                        print(f'{version} {group}: {number}/{len(selected)}', flush=True)
            metrics = aggregate(results, len(selected))
            metrics['perturbations_applied'] = sum(r['perturbation']['applied'] for r in results)
            report['groups'][group] = metrics
            save_json(folder / 'report.json', report)
        report['status'] = 'passed'  # Execution complete, including genuine policy failures in the denominator.
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    save_json(folder / 'report.json', report)
    return 0 if report['status'] == 'passed' else 1


def completed_report(out, version, spec):
    folder = out / version
    if not (folder / 'report.json').exists():
        return None
    report = read(folder / 'report.json')
    if report['status'] != 'passed':
        return None
    if report['frozen_sha256'] != sha256(out / 'frozen.json'):
        raise ValueError('Completed result belongs to a different frozen manifest')
    tasks = load_jsonl(resource_path('AGENTIC_PROCESSED_DATA_DIR') / 'test.jsonl')
    for group, expected in (('natural', {r['id'] for r in tasks}), ('hard', set(spec['hard_ids']))):
        rows = load_jsonl(folder / f'{group}.jsonl')
        if len(rows) != len(expected) or {r['id'] for r in rows} != expected:
            raise ValueError('Completed result IDs/count changed')
        measured = aggregate(rows, len(expected))
        if any(report['groups'][group][k] != v for k, v in measured.items()):
            raise ValueError('Completed report differs from trajectories')
    return report


def summarize(out, spec):
    reports = {v: completed_report(out, v, spec) for v in SOURCES}
    complete = all(reports.values())
    lines = [f'=== FROZEN TEST | {"COMPLETE" if complete else "INCOMPLETE"} ===',
             f'Run: {out.name} | dev-selected candidate: V2',
             'Version / group | N | EM% | F1% | searches | tokens | seconds | protocol failures | applied']
    for version, report in reports.items():
        if report is None:
            lines.append(version + ': pending/failed')
            continue
        for group, m in report['groups'].items():
            failed = m['completed'] - m['statuses'].get('answered', 0)
            lines.append(f'{version} {group}: {m["completed"]} | {m["em_percent"]:.2f} | {m["f1_percent"]:.2f} | '
                         f'{m["mean_searches"]:.2f} | {m["mean_tokens"]:.0f} | {m["mean_seconds"]:.2f} | {failed} | {m["perturbations_applied"]}')
    lines.extend(['Hard: same 75 task IDs/rules; actual changed returns depend on queries and retriever.',
                  'COMPLETE means all runs finished, not every task answered correctly. No test-driven tuning.'])
    text = '\n'.join(lines) + '\n'
    (out / 'summary.txt').write_text(text, encoding='utf-8')
    (out.parent / 'latest_summary.txt').write_text(text, encoding='utf-8')
    print(text, end='', flush=True)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, help='Existing acceptance run directory; no new selection')
    parser.add_argument('--worker', choices=tuple(SOURCES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.resume is None:
            parser.error('Worker requires frozen run directory')
        return worker(args.resume, args.worker)
    root = resource_path('AGENTIC_RUNS_DIR') / 'acceptance'
    if args.resume:
        out = args.resume.resolve()
        if out.parent != root.resolve():
            raise ValueError('Resume must name a run under runs/acceptance')
        spec = read(out / 'frozen.json')
    else:
        if (root / 'latest_run.json').exists():
            raise ValueError('Acceptance already initialized; resume its directory instead of creating another test run')
        out = root / ('test_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid4().hex[:6])
        spec = freeze(out)
        save_json(root / 'latest_run.json', {'run_id': out.name})
    verify(spec)
    print(f'Frozen acceptance: runs/acceptance/{out.name}', flush=True)
    print(f'Natural=300 | hard={spec["hard_tasks"]} | modes={spec["hard_modes"]} | GPUs=1', flush=True)
    for version in SOURCES:
        if completed_report(out, version, spec):
            print(version + ': reuse completed result', flush=True)
            continue
        folder = out / version
        if folder.exists():
            folder.rename(out / (version + '_interrupted_' + uuid4().hex[:6]))
        folder.mkdir()
        log_path = folder / 'run.log'
        print(f'{version}: running; log=runs/acceptance/{out.name}/{version}/run.log', flush=True)
        with log_path.open('w') as log:
            process = subprocess.Popen([sys.executable, '-u', '-m', 'project.evaluation.acceptance',
                                        '--resume', str(out), '--worker', version], stdout=log, stderr=subprocess.STDOUT)
            while True:
                try:
                    code = process.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    print(f'{version}: running; see run.log', flush=True)
                except KeyboardInterrupt:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
        summarize(out, spec)
        if code:
            report_path = folder / 'report.json'
            print(read(report_path).get('error', 'Worker failed') if report_path.exists() else 'Worker exited before reporting', flush=True)
            return 1
    return 0 if summarize(out, spec) else 1


if __name__ == '__main__':
    raise SystemExit(main())
