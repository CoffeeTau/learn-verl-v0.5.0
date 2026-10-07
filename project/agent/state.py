"""Source-preserving task state, not a learned or API-generated factual summary.

Preserve the trained assistant/action -> user/observation conversation contract.
Append a compact ledger without replacing any sampled actions or evidence.
No extra generation/search calls are added.
"""
import html
import json
from project.agent.correction import SYSTEM_PROMPT as V2_PROMPT

SYSTEM_PROMPT = V2_PROMPT + """
Tool results may include a supplementary task_state. Continue the existing search conversation.
Previous judgments are unverified model claims, NOT confirmed facts. Check them against original evidence.
Keep full entity names, titles, and relation direction: a matching name alone does not establish identity.
For each required relation, check which passage actually states it. A retrieved title is not proof.
Use queries_seen to avoid repeating searches that added no useful evidence.
If searches_remaining is zero, emit a judge and answer: sufficient only if the complete relation chain
is supported; otherwise insufficient followed by <answer>insufficient evidence</answer>.
If searches remain and evidence is insufficient, use them to check the missing relation instead of guessing.
"""


def state_messages(question, steps, remaining, system_prompt):
    evidence = {}
    queries = []
    judgments = []
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': question}]
    search_steps = [step for step in steps if 'query' in step]
    maximum = remaining + len(search_steps)
    completed = 0
    for step in steps:
        if 'query' in step:
            completed += 1
            queries.append({'query': step['query'], 'returned_ids': [h['id'] for h in step['hits']]})
            observation = '\n\n'.join(f'[{h["id"]}] {html.escape(h["title"])}\n{html.escape(h["text"])}'
                                        for h in step['hits'])
            messages.extend([{'role': 'assistant', 'content': step['output']},
                             {'role': 'user', 'content': f'<information>\n{observation}\n</information>\n'
                              f'Searches remaining: {maximum - completed}.'}])
        if 'judge' in step:
            judgments.append({'judgment': step['judge'], 'claim': step['reason'], 'verified': False})
        for hit in step.get('hits', []):
            item = {key: hit[key] for key in ('id', 'title', 'text')}
            if hit['id'] in evidence and evidence[hit['id']] != item:
                raise ValueError('Evidence ID changed content within an episode')
            evidence[hit['id']] = item
    state = {'searches_remaining': remaining, 'queries_seen': queries,
             'previous_judgments_unverified': judgments,
             'evidence_ids': list(evidence)}
    if search_steps:
        messages[-1]['content'] += '\n<task_state>\n' + html.escape(json.dumps(state, ensure_ascii=False)) + '\n</task_state>'
        messages[-1]['content'] += f'\nSearches remaining: {remaining}. Start with <judge>sufficient: ...</judge> or <judge>insufficient: ...</judge>, then one action.'
    return messages
