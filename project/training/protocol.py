"""Token-preserving Qwen3 continuation and strict terminal reward; CPU-testable."""
from project.agent.search import parse_action
from project.evaluation.metrics import score_answer


def continuation_ids(tokenizer, observation, last_ids):
    # Qwen3 ChatML only. Never re-tokenize sampled model output.
    # If generation returned EOS itself, do not insert a second im_end.
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    close = "\n" if last_ids and last_ids[-1] == end_id else "<|im_end|>\n"
    text = (close + "<|im_start|>user\n" + observation + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return tokenizer.encode(text, add_special_tokens=False)


def model_segments(ids, mask):
    if len(ids) != len(mask) or any(value not in (0, 1) for value in mask):
        raise ValueError("Invalid response mask")
    segments, current = [], []
    for token, trainable in zip(ids, mask):
        if trainable:
            current.append(token)
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def trajectory_score(texts, gold, max_searches=4, version="v1"):
    searches = 0
    for index, text in enumerate(texts):
        try:
            if version == "v2":
                from project.agent.correction import parse_action as parse_correction
                action = parse_correction(text, after_search=searches > 0)
            else:
                action = parse_action(text)
        except ValueError as exc:
            return {"score": 0.0, "em": 0.0, "status": str(exc), "searches": searches}
        if action["kind"] == "search":
            searches += 1
            if searches > max_searches:
                break
        elif index == len(texts) - 1:
            scores = score_answer(action["answer"], gold)
            return {"score": scores["f1"], "em": scores["em"], "status": "answered", "searches": searches}
        else:
            break
    return {"score": 0.0, "em": 0.0, "status": "unfinished_or_budget", "searches": searches}
