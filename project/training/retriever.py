"""One CPU E5 instance shared by the single-node training job; no labels."""
import json
from pathlib import Path
import ray
from project.retrieval.local import LocalRetriever


@ray.remote(num_cpus=1)
class RetrievalActor:
    def __init__(self, corpus, index, model, hard_plan=None, audit_path=None, dev_hard_plan=None):
        self.retriever = LocalRetriever(corpus, index, model)
        self.plan = json.loads(Path(hard_plan).read_text()) if hard_plan else {}
        self.dev_plan = json.loads(Path(dev_hard_plan).read_text()) if dev_hard_plan else {}
        if set(self.plan) & set(self.dev_plan):
            raise ValueError("Train/dev perturbation questions overlap")
        self.audit_path = Path(audit_path) if audit_path else None
        self.paragraphs = {row["id"]: row["paragraph_id"] for row in self.retriever.rows}

    def ready(self):
        return self.retriever.manifest["corpus_sha256"]

    def search(self, question, query, top_k):
        return self.retriever.search(question, query, top_k)

    def search_v2(self, question, query, top_k, search_count, perturb):
        from project.training.hard_episodes import transform_hits
        plan = self.dev_plan if perturb == "dev" else self.plan
        specification = plan.get(question) if perturb and search_count == 0 else None
        candidates = self.retriever.search(question, query, max(top_k, 20) if specification else top_k)
        hits = candidates[:top_k]
        if not specification:
            return hits
        selected, applied = transform_hits(hits, candidates, specification, self.paragraphs, top_k)
        if self.audit_path:
            path = self.audit_path.with_name('dev_' + self.audit_path.name) if perturb == 'dev' else self.audit_path
            with path.open('a') as handle:
                handle.write(json.dumps({'task_id': specification['task_id'], 'mode': specification['mode'],
                                         'applied': applied, 'query': query, 'original_ids': [h['id'] for h in hits],
                                         'returned_ids': [h['id'] for h in selected]}) + '\n')
        return selected
