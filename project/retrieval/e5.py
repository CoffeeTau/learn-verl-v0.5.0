# SPDX-License-Identifier: Apache-2.0
# Adapted from Search-R1 contributors' search_r1/search/retrieval_server.py:
# pooling(), Encoder.encode(). See project/THIRD_PARTY.md.
"""Small local E5 adapter; no FAISS, HTTP server or reference-repo imports."""

class E5Encoder:
    def __init__(self, model_path):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()

    def encode(self, texts, is_query=False, batch_size=16):
        torch = self.torch
        chunks = []
        prefix = "query: " if is_query else "passage: "
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                inputs = self.tokenizer(
                    [prefix + text for text in texts[start:start + batch_size]],
                    max_length=512, padding=True, truncation=True, return_tensors="pt",
                )
                output = self.model(**inputs)
                # Search-R1 masked mean pooling, followed by L2 normalization.
                hidden = output.last_hidden_state.masked_fill(
                    ~inputs["attention_mask"][..., None].bool(), 0.0
                )
                pooled = hidden.sum(dim=1) / inputs["attention_mask"].sum(dim=1)[..., None]
                chunks.append(torch.nn.functional.normalize(pooled, dim=-1))
        embeddings = torch.cat(chunks)
        if not torch.isfinite(embeddings).all() or (embeddings.norm(dim=-1) < 0.99).any():
            raise ValueError("E5 returned invalid or zero embeddings")
        return embeddings
