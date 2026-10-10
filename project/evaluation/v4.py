"""Independent V4 dev evaluation of the selected checkpoint; no old-test reuse."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import traceback
import uuid
from project.agent.correction import SYSTEM_PROMPT, parse_action
from project.agent.search import ContextBudgetError, run_episode
from project.common import code_info, load_jsonl, model_inventory, resource_path, save_json, sha256, show_summary
from project.evaluation.acceptance import episode_search
from project.evaluation.metrics import score_answer
from project.evaluation.v0 import aggregate
from project.evaluation.v1 import export_model
from project.training.v4 import read, validate_retriever


def main(version="v4"):
    base = version in ("v4_base", "v4_base_evidence")
    if version not in ("v4", "v4_base", "v4_base_evidence"):
        raise ValueError("Unknown evaluation stage")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-run', required=True)
    parser.add_argument('--history-mode', choices=['token-continuation'] if base else ['rerender', 'token-continuation'], default='token-continuation' if base else 'rerender')
    parser.add_argument('--expect-step', type=int, help='Fail if the selected checkpoint differs')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.train_run):
        parser.error('Invalid run ID')
    runs = resource_path('AGENTIC_RUNS_DIR')
    train = runs / version / args.train_run
    manifest, report = read(train / 'manifest.json'), read(train / 'report.json')
    if manifest['mode'] != 'main' or manifest['version'] != version or report['status'] != 'PASSED':
        raise ValueError('Use a passed V4 main run, not smoke')
    source, corpus = resource_path('AGENTIC_PROCESSED_DATA_DIR'), resource_path('AGENTIC_CORPUS_DIR')
    if model_inventory(resource_path('AGENTIC_MODEL_DIR')) != manifest['tokenizer']:
        raise ValueError('Original tokenizer/model inventory changed')
    if manifest['data']['source_manifest_sha256'] != sha256(source / 'manifest.json'):
        raise ValueError('Frozen dataset changed')
    dataset = read(source / 'manifest.json')
    for name in ('dev.jsonl', 'dev.labels.jsonl'):
        if sha256(source / name) != dataset['files'][name]:
            raise ValueError('Frozen dev file changed')
    for name in ('dev_hard_plan', 'validation_tasks'):
        if sha256(train / (name + '.json')) != manifest[name + '_sha256']:
            raise ValueError('Validation plan changed')
    retrieval = validate_retriever(Path(manifest['retrieval']['run']), source, corpus)
    if retrieval != manifest['retrieval']:
        raise ValueError('V4 training and evaluation retrieval differ')
    from project.training.v4_monitor import select_checkpoint
    selection = select_checkpoint(load_jsonl(train / 'validation_metrics.jsonl'))
    if selection != read(train / 'selection.json'):
        raise ValueError('Checkpoint selection changed')
    step = selection['selected_step']
    if args.expect_step is not None and step != args.expect_step:
        raise ValueError('Selected checkpoint differs from --expect-step')
    aligned = args.history_mode == 'token-continuation'
    if base and not aligned:
        raise ValueError('V4-base only supports token continuation')
    from project.agent.goal_anchor import system_prompt
    prompt_text = system_prompt(base)
    if base and (not manifest.get('goal_anchor') or manifest.get('system_prompt') != prompt_text
                 or manifest.get('goal_anchor_sha256') != sha256(Path(__file__).resolve().parents[1] / 'agent/goal_anchor.py')):
        raise ValueError('Goal-anchor training prompt changed')
    sampling = dict(temperature=0.0)
    prompt_limit, response_limit = 512, 8192
    if aligned:
        import yaml
        saved = yaml.safe_load((train / 'config.yaml').read_text())
        if bool(saved['agentic'].get('goal_anchor', False)) != base:
            raise ValueError('Training/evaluation goal anchor differs')
        rollout = saved['actor_rollout_ref']['rollout']
        sampling.update(temperature=rollout['val_kwargs']['temperature'],
                        top_p=rollout['val_kwargs']['top_p'], top_k=rollout['top_k'],
                        repetition_penalty=1.0)
        prompt_limit, response_limit = rollout['prompt_length'], rollout['response_length']
    policy = export_model(train, step) if step else Path(manifest['policy']['path'])
    if not step and model_inventory(policy) != manifest['policy']:
        raise ValueError('Initial policy changed')
    run_id = 'dev_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:6]
    out = runs / (version + '_eval' if base else ('v4_eval_aligned' if aligned else 'v4_eval')) / run_id
    out.mkdir(parents=True)
    summary = {'status': 'running', 'train_run': args.train_run, 'selected_step': step,
               'history_mode': args.history_mode, 'groups': {}}
    try:
        from project.retrieval.local import LocalRetriever
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
        config = manifest['data']['task']
        model_dir = Path(retrieval['run'])
        retriever = LocalRetriever(corpus, model_dir / 'index', model_dir / 'model')
        paragraphs = {r['id']: r['paragraph_id'] for r in retriever.rows}
        tokenizer_path = resource_path('AGENTIC_MODEL_DIR')
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
        save_json(out / 'manifest.json', {'code': code_info(), 'training_manifest': manifest,
                  'selection': selection, 'policy_model': model_inventory(policy),
                  'tokenizer': model_inventory(tokenizer_path), 'config': config,
                  'goal_anchor': base, 'system_prompt': prompt_text, 'protocol': 'v2', 'history_mode': args.history_mode,
                  'sampling': sampling, 'training_config_sha256': sha256(train / 'config.yaml'),
                  'scoring': 'canonical_answer_em_f1_no_aliases', 'enable_thinking': False})
        llm = LLM(model=str(policy), tokenizer=str(tokenizer_path), dtype='bfloat16', tensor_parallel_size=1,
                  max_model_len=config['max_context_tokens'], max_num_seqs=1,
                  gpu_memory_utilization=config['gpu_memory_utilization'], enforce_eager=True, seed=config['seed'])
        params = SamplingParams(**sampling, max_tokens=config['max_new_tokens'],
                                stop=['</search>', '</answer>'], include_stop_str_in_output=True)
        def generate(messages):
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if len(ids) + config['max_new_tokens'] > config['max_context_tokens']:
                raise ContextBudgetError()
            generated = llm.generate([{'prompt_token_ids': ids}], params, use_tqdm=False)[0].outputs[0]
            return {'text': generated.text, 'finish_reason': generated.finish_reason,
                    'prompt_tokens': len(ids), 'output_tokens': len(generated.token_ids)}
        tasks = load_jsonl(source / 'dev.jsonl')
        labels = {r['id']: r for r in load_jsonl(source / 'dev.labels.jsonl')}
        plan = read(train / 'dev_hard_plan.json')
        expected = read(train / 'validation_tasks.json')
        if {r['id'] for r in tasks} != set(expected['natural']):
            raise ValueError('Development IDs changed')
        for group in ('natural', 'hard'):
            chosen = [r for r in tasks if group == 'natural' or r['id'] in expected['hard']]
            results = []
            with (out / (group + '.jsonl')).open('w') as handle:
                for i, task in enumerate(chosen, 1):
                    audit = {'applied': False}
                    spec = plan[task['question']] if group == 'hard' else None
                    search = episode_search(retriever, spec, paragraphs, audit)
                    if aligned:
                        from project.evaluation.continuation import run_continuation
                        def generate_ids(ids):
                            return llm.generate([{'prompt_token_ids': ids}], params, use_tqdm=False)[0].outputs[0].token_ids
                        result = run_continuation(task['question'], generate_ids, search, config, tokenizer,
                                                  prompt_limit, response_limit, goal_anchor=base)
                    else:
                        result = run_episode(task['question'], generate, search,
                                             config, system_prompt=SYSTEM_PROMPT, action_parser=parse_action)
                    gold = labels[task['id']]
                    scores = score_answer(result['answer'], gold['answer']) if result['status'] == 'answered' else {'em': 0., 'f1': 0.}
                    result.update(id=task['id'], question=task['question'], gold=gold['answer'], perturbation=audit, **scores)
                    handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                    handle.flush()
                    results.append(result)
                    if i % 25 == 0 or i == len(chosen):
                        print(f'{version.upper()} {group}: {i}/{len(chosen)}', flush=True)
            summary['groups'][group] = aggregate(results, len(chosen))
            summary['groups'][group]['applied'] = sum(r['perturbation']['applied'] for r in results)
            if group == 'natural':
                matched = [r for r in results if r['id'] in expected['hard']]
                summary['groups']['natural_matched'] = aggregate(matched, len(matched))
        summary['status'] = 'complete'
    except Exception as exc:
        summary.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    save_json(out / 'report.json', summary)
    lines = [f"=== {version.upper()} DEV | {summary['status'].upper()} ===", f'Run: {run_id} | selected step={step}']
    for group, m in summary['groups'].items():
        lines.append(f"{group}: N={m['completed']} EM={m['em_percent']:.2f}% F1={m['f1_percent']:.2f}% "
                     f"searches={m['mean_searches']:.2f} tokens={m['mean_tokens']:.0f}")
        lines.append(f"  statuses={m['statuses']} | applied={m.get('applied', 0)}")
    lines.append(f'History: {args.history_mode}; same dev/strict scoring; engine deployment may still differ from trainer.')
    if summary.get('error'):
        lines.append(summary['error'][:240])
    show_summary(out, '\n'.join(lines))
    (out.parent / 'latest_summary.txt').write_text('\n'.join(lines) + '\n')
    save_json(out.parent / 'latest_run.json', {'run_id': run_id, 'status': summary['status']})
    return 0 if summary['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
