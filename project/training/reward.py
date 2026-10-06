"""Terminal F1 from model-only token spans; observations cannot earn rewards."""
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from verl.workers.reward_manager import register
from project.training.protocol import model_segments, trajectory_score


@register("agentic_f1")
class SearchRewardManager:
    def __init__(self, tokenizer, num_examine=0, compute_score=None, reward_fn_key=None,
                 audit_dir=None, max_searches=4, max_new_tokens=512):
        self.tokenizer = tokenizer
        self.audit_dir = Path(audit_dir) if audit_dir else None
        self.max_searches, self.max_new_tokens = max_searches, max_new_tokens

    def __call__(self, data, return_dict=False):
        rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        extras, groups, records = defaultdict(list), defaultdict(list), []
        prompt_len = data.batch["prompts"].shape[-1]
        for i in range(len(data)):
            length = int(data.batch["attention_mask"][i, prompt_len:].sum())
            ids = data.batch["responses"][i, :length].tolist()
            mask = data.batch["response_mask"][i, :length].tolist()
            segments = model_segments(ids, mask)
            if not segments or not mask[-1]:
                raise ValueError("Reward must land on a sampled model token")
            texts = [self.tokenizer.decode(part, skip_special_tokens=True) for part in segments]
            gold = data.non_tensor_batch["reward_model"][i]["ground_truth"]
            score = trajectory_score(texts, gold, self.max_searches)
            if any(len(part) >= self.max_new_tokens for part in segments):
                score.update(score=0.0, em=0.0, status="generation_truncated")
            rewards[i, length - 1] = score["score"]
            for key, value in score.items():
                extras[key].append(value)
            task_id = data.non_tensor_batch["extra_info"][i]["task_id"]
            groups[str(task_id)].append(score["score"])
            records.append({**score, "task_id": str(task_id), "model_tokens": sum(mask),
                            "tool_template_tokens": length - sum(mask), "model_outputs": texts})
        varying = sum(len(values) > 1 and np.ptp(values) > 0 for values in groups.values())
        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            # Per-process file: no concurrent append corruption between Ray workers.
            with (self.audit_dir / f"rewards_{os.getpid()}.jsonl").open("a") as handle:
                handle.write(json.dumps({"episodes": len(records), "groups": len(groups),
                                         "varying_groups": varying, "records": records}) + "\n")
        return {"reward_tensor": rewards, "reward_extra_info": dict(extras)} if return_dict else rewards
