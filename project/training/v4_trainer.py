"""Only extend validation reporting; optimization and checkpointing stay in veRL."""
from pathlib import Path
import json
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from project.common import load_jsonl, save_json
from project.training.v4_monitor import summarize_validation, select_checkpoint


class V4Trainer(RayPPOTrainer):
    def _validate(self):
        metrics = super()._validate()
        out = Path(self.config.trainer.default_local_dir).parent
        expected = json.loads((out / 'validation_tasks.json').read_text())
        rows = load_jsonl(Path(self.config.trainer.validation_data_dir) / f'{self.global_steps}.jsonl')
        groups = summarize_validation(rows, expected)
        path = out / 'validation_metrics.jsonl'
        history = load_jsonl(path) if path.exists() else []
        if any(int(row['step']) == self.global_steps for row in history):
            raise ValueError('Duplicate V4 validation step; do not append a restarted run')
        record = {'step': int(self.global_steps), 'groups': groups}
        with path.open('a') as handle:
            handle.write(json.dumps(record) + '\n')
            handle.flush()
        selection = select_checkpoint(history + [record])
        save_json(out / 'selection.json', selection)
        for group, values in groups.items():
            for key, value in values.items():
                metrics[f'dev/{group}/{key}'] = value
        metrics['dev/selected_step'] = selection['selected_step']
        natural, hard = groups['natural'], groups['hard']
        text = (f"V4 DEV step={self.global_steps} | natural EM={natural['em']:.2%} F1={natural['f1']:.2%} "
                f"failures={natural['protocol_failures']}/{natural['n']}\n"
                f"Matched {hard['n']}: natural EM={groups['natural_matched']['em']:.2%} "
                f"hard EM={hard['em']:.2%} | selected step={selection['selected_step']}")
        (out / 'validation_summary.txt').write_text(text + '\n')
        print(text, flush=True)
        return metrics
