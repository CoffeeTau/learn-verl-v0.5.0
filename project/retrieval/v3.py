"""One train-only E5 adaptation: relation supervision, conservative mined negatives.

Uses the existing Search-R1-derived E5 adapter. Agentic-R motivates sequential
retriever optimization; no teacher sub-answer utility or alternating RL is used.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import random
import re
import unicodedata
from uuid import uuid4

from project.common import (code_info, load_jsonl, model_inventory, resource_path,
                            save_json, save_jsonl, sha256, show_summary, signature)


def normalized(text):
    return ' '.join(re.sub(r'[^\w]+', ' ', unicodedata.normalize('NFKC', text).casefold()).split())


def contains(text, phrase):
    return bool(phrase) and f' {phrase} ' in f' {normalized(text)} '


def relation_candidates(tasks, labels, corpus):
    """Positive must be an annotated support sentence, not an arbitrary chunk hit."""
    by_id = {r['id']: r for r in corpus}
    candidates, counts, seen = [], Counter(), set()
    for task in tasks:
        label = labels[task['id']]
        support = label['supporting_facts']
        excluded = sorted({by_id[f['passage_id']]['paragraph_id'] for f in support})
        for triple in label['evidences']:
            counts['triples'] += 1
            if not isinstance(triple, (list, tuple)) or len(triple) != 3 or not all(isinstance(x, str) and x.strip() for x in triple):
                counts['invalid_triple'] += 1
                continue
            subject, relation, obj = triple
            subject_key, object_key = normalized(subject), normalized(obj)
            positives = []
            for fact in support:
                passage = by_id[fact['passage_id']]
                sentence = passage['sentences'][passage['sentence_ids'].index(fact['sentence_id'])]
                # Exact normalized subject title plus object in the labeled sentence.
                # This intentionally sacrifices recall instead of guessing alias alignment.
                if normalized(passage['title']) == subject_key and contains(sentence, object_key):
                    positives.append(passage['id'])
            key = (task['id'], subject_key, normalized(relation))
            if not positives or key in seen:
                counts['unaligned_or_duplicate'] += 1
                continue
            seen.add(key)
            query = subject + ' ' + relation
            candidates.append({'task_id': task['id'], 'question': task['question'], 'query': query,
                               'positive_id': sorted(set(positives))[0], 'subject': subject, 'object': obj,
                               'excluded_paragraphs': excluded})
    return candidates, dict(counts)


def allowed_negative(row, candidate):
    # Unlabeled is NOT automatically negative. Exclude every support paragraph,
    # all occurrences of the target subject/object, including title variants.
    text = row['title'] + '\n' + row['text']
    return (row['paragraph_id'] not in candidate['excluded_paragraphs']
            and not contains(text, normalized(candidate['subject']))
            and not contains(text, normalized(candidate['object'])))


def paths():
    root = resource_path('AGENTIC_ROOT')
    return root / 'data/processed/retriever_v3', root / 'runs/v3_retriever'


def frozen_data():
    data = resource_path('AGENTIC_PROCESSED_DATA_DIR')
    corpus_dir = resource_path('AGENTIC_CORPUS_DIR')
    manifest = json.loads((data / 'manifest.json').read_text())
    for name in ('train.jsonl', 'train.labels.jsonl'):
        if sha256(data / name) != manifest['files'][name]:
            raise ValueError('Frozen training data changed: ' + name)
    if sha256(corpus_dir / 'corpus.jsonl') != manifest['corpus_sha256']:
        raise ValueError('Frozen corpus changed')
    return data, corpus_dir, manifest


def prepare(device):
    import numpy as np
    from project.retrieval.local import LocalRetriever
    data, corpus_dir, manifest = frozen_data()
    out, _ = paths()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise ValueError('Prepared V3 pairs already exist; keep the frozen artifact and run train')
    corpus = load_jsonl(corpus_dir / 'corpus.jsonl')
    tasks = load_jsonl(data / 'train.jsonl')
    labels = {r['id']: r for r in load_jsonl(data / 'train.labels.jsonl')}
    candidates, counts = relation_candidates(tasks, labels, corpus)
    random.Random(42).shuffle(candidates)
    candidates = candidates[:4000]
    retriever = LocalRetriever(corpus_dir, resource_path('AGENTIC_INDEX_DIR'), resource_path('AGENTIC_RETRIEVER_DIR'))
    retriever.encoder.device = device
    retriever.encoder.model.to(device)
    pairs = []
    for start in range(0, len(candidates), 64):
        chunk = candidates[start:start + 64]
        vectors = retriever.encoder.encode([c['question'] + ' [SEP] ' + c['query'] for c in chunk],
                                           is_query=True, batch_size=32).numpy()
        scores = vectors @ retriever.vectors.T
        for candidate, values in zip(chunk, scores):
            order = np.argsort(-values, kind='stable')[:100]
            negatives = []
            for i in order:
                row = corpus[int(i)]
                if allowed_negative(row, candidate):
                    negatives.append(row['id'])
                if len(negatives) == 3:
                    break
            if len(negatives) == 3:
                pairs.append({**candidate, 'negative_ids': negatives})
            else:
                counts['insufficient_negatives'] = counts.get('insufficient_negatives', 0) + 1
        if start % 512 == 0:
            print(f'V3 pairs: {min(start + 64, len(candidates))}/{len(candidates)}', flush=True)
    if len(pairs) < 32:
        raise ValueError(f'Only {len(pairs)} aligned pairs; inspect alignment before training')
    save_jsonl(out / 'pairs.jsonl', pairs)
    by_id = {r['id']: r for r in corpus}
    preview = [{**p, 'positive': by_id[p['positive_id']],
                'negatives': [by_id[k] for k in p['negative_ids']]} for p in pairs[:5]]
    save_json(out / 'pair_examples.json', preview)
    saved = {'status': 'passed', 'source_manifest_sha256': sha256(data / 'manifest.json'),
             'corpus_sha256': manifest['corpus_sha256'], 'pairs_sha256': sha256(out / 'pairs.jsonl'),
             'base_model': model_inventory(resource_path('AGENTIC_RETRIEVER_DIR')),
             'base_index': retriever.manifest, 'seed': 42, 'pairs': len(pairs),
             'train_tasks_used': len({p['task_id'] for p in pairs}), 'counts': counts,
             'method': 'support_sentence_aligned_relation_query_with_filtered_top100_negatives', 'code': code_info()}
    save_json(out / 'manifest.json', saved)
    lines = ['=== V3 PAIRS | PASSED ===', f'Train-only pairs={len(pairs)} | tasks={saved["train_tasks_used"]}',
             'Positive: annotated support sentence + subject title + object match',
             'Negatives: 3/query; exclude support paragraphs and subject/object mentions',
             f'Alignment: {counts}', 'Labels are weak supervision; alternative evidence may remain.']
    for p in preview[:2]:
        lines.extend(['Query: ' + p['query'][:140], 'Positive: ' + p['positive']['title'],
                      'Negatives: ' + ' | '.join(n['title'] for n in p['negatives'])[:180]])
    show_summary(out, '\n'.join(lines))


def train(device):
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from project.retrieval.e5 import E5Encoder
    data, corpus_dir, dataset = frozen_data()
    prepared, runs = paths()
    source = json.loads((prepared / 'manifest.json').read_text())
    if source['source_manifest_sha256'] != sha256(data / 'manifest.json') or source['pairs_sha256'] != sha256(prepared / 'pairs.jsonl'):
        raise ValueError('Prepared pair provenance changed')
    if source['base_model'] != model_inventory(resource_path('AGENTIC_RETRIEVER_DIR')):
        raise ValueError('Base E5 changed after mining')
    pairs = load_jsonl(prepared / 'pairs.jsonl')
    corpus = {r['id']: r for r in load_jsonl(corpus_dir / 'corpus.jsonl')}
    run_id = 'main_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid4().hex[:6]
    out = runs / run_id
    out.mkdir(parents=True)
    save_json(runs / 'latest_run.json', {'run_id': run_id, 'status': 'running'})
    torch.manual_seed(42)
    random.Random(42).shuffle(pairs)
    encoder = E5Encoder(str(resource_path('AGENTIC_RETRIEVER_DIR')), device=device)
    encoder.model.train()
    optimizer = torch.optim.AdamW(encoder.model.parameters(), lr=1e-5, weight_decay=0)
    tb_dir = resource_path('AGENTIC_ROOT') / 'tensorboard' / ('v3_retriever_' + run_id)
    writer = SummaryWriter(str(tb_dir))
    probes = []
    for parameter in encoder.model.parameters():
        if parameter.requires_grad and parameter.numel():
            indices = torch.linspace(0, parameter.numel() - 1, min(32, parameter.numel()),
                                     device=device).long()
            probes.append((parameter, indices, parameter.detach().flatten()[indices].clone()))
    manifest = {'status': 'running', 'pairs_manifest': source, 'epochs': 1, 'batch_size': 8,
                'lr': 1e-5, 'temperature': 0.02, 'in_batch_negatives': False, 'device': device,
                'loss': 'per_query_positive_vs_three_filtered_negatives_cross_entropy', 'code': code_info()}
    save_json(out / 'manifest.json', manifest)

    def encode(texts, prefix):
        inputs = encoder.tokenizer([prefix + t for t in texts], padding=True, truncation=True,
                                   max_length=512, return_tensors='pt').to(device)
        output = encoder.model(**inputs).last_hidden_state
        output = output.masked_fill(~inputs['attention_mask'][..., None].bool(), 0)
        pooled = output.sum(1) / inputs['attention_mask'].sum(1)[..., None]
        return torch.nn.functional.normalize(pooled, dim=-1)

    try:
        losses = []
        with (out / 'metrics.jsonl').open('w') as handle:
            for step, start in enumerate(range(0, len(pairs), 8), 1):
                batch = pairs[start:start + 8]
                optimizer.zero_grad(set_to_none=True)
                q = encode([p['question'] + ' [SEP] ' + p['query'] for p in batch], 'query: ')
                docs = [corpus[k] for p in batch for k in [p['positive_id']] + p['negative_ids']]
                d = encode([r['title'] + '\n' + r['text'] for r in docs], 'passage: ')
                logits = (q[:, None, :] * d.reshape(len(batch), 4, -1)).sum(-1) / 0.02
                loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device=device))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite retriever loss')
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(encoder.model.parameters(), 1.0)
                if not torch.isfinite(grad):
                    raise ValueError('Nonfinite retriever gradient')
                optimizer.step()
                value = float(loss.detach())
                losses.append(value)
                metrics = {'loss': value, 'grad_norm': float(grad),
                           'positive_top1': float((logits.argmax(-1) == 0).float().mean().detach())}
                handle.write(json.dumps({'step': step, 'metrics': metrics}) + '\n')
                handle.flush()
                for key, val in metrics.items():
                    writer.add_scalar('retriever_train/' + key, val, step)
                writer.flush()
                if step == 1 or step % 25 == 0:
                    print(f'V3 E5: {step}/{(len(pairs)+7)//8} | loss={value:.4f}', flush=True)
        delta = max(float((p.detach().flatten()[indices] - before).abs().max())
                    for p, indices, before in probes)
        if not delta > 0:
            raise ValueError('No sampled E5 parameter change')
        model_path = out / 'model'
        encoder.model.save_pretrained(model_path, safe_serialization=True)
        encoder.tokenizer.save_pretrained(model_path)
        report = {'status': 'passed', 'steps': len(losses), 'mean_loss': sum(losses)/len(losses),
                  'probe_delta': delta, 'model_path': str(model_path), 'model_inventory': model_inventory(model_path),
                  'corpus_sha256': dataset['corpus_sha256']}
        save_json(out / 'report.json', report)
        save_json(runs / 'latest_run.json', {'run_id': run_id, 'status': 'passed'})
        show_summary(out, f'=== V3 E5 TRAIN | PASSED ===\nRun: {run_id}\nPairs={len(pairs)} | epochs=1 | steps={len(losses)}\n'
                         f'Parameter delta={delta:.3g} | mean loss={report["mean_loss"]:.4f}\n'
                         'Qwen V2 frozen; no claim of development improvement.\nNext: build V3 index.')
    except Exception as exc:
        save_json(out / 'report.json', {'status': 'failed', 'error': str(exc)})
        save_json(runs / 'latest_run.json', {'run_id': run_id, 'status': 'failed'})
        raise
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'train', 'index'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.device)
    elif args.stage == 'train':
        train(args.device)
    else:
        import subprocess
        import sys
        _, runs = paths()
        latest = json.loads((runs / 'latest_run.json').read_text())
        if latest['status'] != 'passed' or not re.fullmatch(r'[A-Za-z0-9_-]+', latest['run_id']):
            raise ValueError('No completed V3 retriever run')
        run = runs / latest['run_id']
        report = json.loads((run / 'report.json').read_text())
        if report['status'] != 'passed' or report['model_inventory'] != model_inventory(run / 'model'):
            raise ValueError('Trained E5 provenance changed')
        raise SystemExit(subprocess.call([sys.executable, '-u', '-m', 'project.retrieval.build_index',
                         '--device', args.device, '--model-path', str(run / 'model'),
                         '--index-path', str(run / 'index'), '--report-path', str(run / 'index_report')]))


if __name__ == '__main__':
    main()
