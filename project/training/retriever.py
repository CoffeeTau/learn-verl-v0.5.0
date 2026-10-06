"""One CPU E5 instance shared by the single-node training job; no labels."""
import ray
from project.retrieval.local import LocalRetriever


@ray.remote(num_cpus=1)
class RetrievalActor:
    def __init__(self, corpus, index, model):
        self.retriever = LocalRetriever(corpus, index, model)

    def ready(self):
        return self.retriever.manifest["corpus_sha256"]

    def search(self, question, query, top_k):
        return self.retriever.search(question, query, top_k)
