"""Read a frozen corpus/index; never receive QA labels or filter by task ID."""
import json
import numpy as np

from project.common import load_jsonl, model_inventory, sha256, signature
from project.retrieval.e5 import E5Encoder


class LocalRetriever:
    def __init__(self, corpus_dir, index_dir, model_dir):
        self.manifest = json.loads((index_dir / "manifest.json").read_text())
        if sha256(corpus_dir / "corpus.jsonl") != self.manifest["corpus_sha256"]:
            raise ValueError("Retrieval corpus/index mismatch")
        if signature(model_inventory(model_dir)) != self.manifest["model_inventory_signature"]:
            raise ValueError("E5 local file inventory changed since indexing")
        if sha256(index_dir / "embeddings.npy") != self.manifest["embeddings_sha256"]:
            raise ValueError("Index file differs from its manifest")
        self.rows = load_jsonl(corpus_dir / "corpus.jsonl")
        self.vectors = np.load(index_dir / "embeddings.npy", mmap_mode="r", allow_pickle=False)
        if list(self.vectors.shape) != self.manifest["shape"] or len(self.rows) != len(self.vectors):
            raise ValueError("Index/corpus shape mismatch")
        self.encoder = E5Encoder(str(model_dir))

    def search(self, question, query, top_k):
        vector = self.encoder.encode([question + " [SEP] " + query], is_query=True).numpy()[0]
        scores = self.vectors @ vector
        indices = np.argsort(-scores, kind="stable")[:top_k]
        return [{"id": self.rows[i]["id"], "title": self.rows[i]["title"],
                 "text": self.rows[i]["text"], "score": float(scores[i])} for i in indices]
