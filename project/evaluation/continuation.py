"""Standalone counterpart of CorrectionAgentLoop; preserve sampled token IDs."""
import time
from project.agent.correction import parse_action
from project.training.protocol import continuation_ids
from project.agent.goal_anchor import system_prompt, observation_text


def run_continuation(question, generate_ids, search, config, tokenizer, prompt_limit=512, response_limit=8192, goal_anchor=False):
    started = time.monotonic()
    prompt = tokenizer.apply_chat_template([
        {'role': 'system', 'content': system_prompt(goal_anchor).format(max_searches=config['max_searches'])},
        {'role': 'user', 'content': question}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    if len(prompt) > prompt_limit:
        raise ValueError('Prompt exceeds training prompt budget')
    response, seen = [], set()
    result = dict(answer='', sources=[], searches=0, steps=[], status='unfinished_or_budget',
                  prompt_tokens=0, output_tokens=0)
    cap = config['max_new_tokens']
    for search_count in range(config['max_searches'] + 1):
        if len(prompt) + len(response) + cap > config['max_context_tokens']:
            break
        ids = list(generate_ids(prompt + response))
        if not ids:
            raise RuntimeError('Empty engine response')
        result['prompt_tokens'] += len(prompt) + len(response)
        result['output_tokens'] += len(ids)
        response.extend(ids)
        text = tokenizer.decode(ids, skip_special_tokens=True)
        step = dict(output=text, output_token_ids=ids, output_tokens=len(ids),
                    prompt_tokens=len(prompt) + len(response) - len(ids),
                    finish_reason='length' if len(ids) >= cap else 'stop')
        result['steps'].append(step)
        if len(ids) >= cap:
            result['status'] = 'generation_truncated'
            break
        try:
            action = parse_action(text, after_search=search_count > 0)
        except ValueError as exc:
            result['status'] = str(exc)
            break
        if 'judge' in action:
            step.update(judge=action['judge'], reason=action['reason'])
        if action['kind'] == 'answer':
            result.update(answer=action['answer'], sources=action['sources'], status='answered',
                          invalid_source_ids=sorted(set(action['sources']) - seen))
            break
        # Trainer reward counts emitted searches, including an over-budget request.
        result['searches'] += 1
        if search_count == config['max_searches']:
            break
        hits = search(question, action['query'], config['top_k'])
        step.update(query=action['query'], hits=hits)
        seen.update(h['id'] for h in hits)
        observation = observation_text(hits, config['max_searches'] - search_count - 1, question, goal_anchor)
        inserted = continuation_ids(tokenizer, observation, ids)
        if len(prompt) + len(response) + len(inserted) + cap > config['max_context_tokens']:
            break
        response.extend(inserted)
    if not response or len(response) > response_limit:
        raise ValueError('Invalid trajectory length')
    result.update(total_tokens=result['prompt_tokens'] + result['output_tokens'],
                  seconds=time.monotonic() - started)
    return result
