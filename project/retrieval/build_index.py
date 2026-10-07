"""Build one exact inner-product E5 index for the project's modest shared corpus."""
import argparse
import json
import traceback
from pathlib import Path

from project.common import (config_from, load_jsonl, model_inventory, resource_path,
                            save_json, sha256, show_summary, signature)
from project.retrieval.e5 import E5Encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()
    if bool(args.model_path) != bool(args.index_path) or bool(args.model_path) != bool(args.report_path):
        parser.error('Custom index requires model, index and report paths together')
    out = args.report_path or resource_path("AGENTIC_RUNS_DIR") / "index_v1"
    try:
        import numpy as np
        config = config_from(args.config)
        corpus_dir = resource_path("AGENTIC_CORPUS_DIR")
        corpus_path = corpus_dir / "corpus.jsonl"
        corpus_manifest = json.loads((corpus_dir / "manifest.json").read_text())
        digest = sha256(corpus_path)
        if digest != corpus_manifest["corpus_sha256"]:
            raise ValueError("Corpus differs from frozen manifest")
        path = args.index_path or resource_path("AGENTIC_INDEX_DIR")
        path.mkdir(parents=True, exist_ok=True)
        if (path / "manifest.json").exists():
            raise ValueError("Index already exists; reuse it rather than overwriting the V0 index")
        model_path = args.model_path or resource_path("AGENTIC_RETRIEVER_DIR")
        inventory = model_inventory(model_path)
        rows = load_jsonl(corpus_path)
        encoder = E5Encoder(str(model_path), device=args.device)
        vectors = None
        for start in range(0, len(rows), 512):
            chunk = rows[start:start + 512]
            batch = encoder.encode([row["title"] + "\n" + row["text"] for row in chunk],
                                   batch_size=config["index_batch_size"]).numpy()
            if vectors is None:
                vectors = np.lib.format.open_memmap(path / "embeddings.partial.npy", mode="w+",
                                                    dtype="float32", shape=(len(rows), batch.shape[1]))
            vectors[start:start + len(chunk)] = batch
            print(f"Index: {start + len(chunk)}/{len(rows)}", flush=True)
        if vectors is None:
            raise ValueError("Empty corpus")
        shape = list(vectors.shape)
        vectors.flush()
        del vectors
        (path / "embeddings.partial.npy").replace(path / "embeddings.npy")
        manifest = {"status": "passed", "corpus_sha256": digest,
                    "model_inventory": inventory, "model_inventory_signature": signature(inventory),
                    "embeddings_sha256": sha256(path / "embeddings.npy"), "shape": shape,
                    "device": args.device, "method": "normalized_inner_product_exact",
                    "query_input": "original_question [SEP] current_query"}
        save_json(path / "manifest.json", manifest)
        save_json(out / "report.json", manifest)
        show_summary(out, f"=== INDEX | PASSED ===\nE5 device: {args.device} | shape={shape}\n"
                         f"Corpus SHA256: {digest[:16]}\nMethod: normalized exact inner product\n"
                         + ('Next: V3 state integration and development evaluation.' if args.model_path else 'Next: V0 on project dev (200 tasks).'))
        return 0
    except Exception as exc:
        save_json(out / "report.json", {"status": "failed", "traceback": traceback.format_exc()})
        show_summary(out, f"=== INDEX | FAILED ===\n{type(exc).__name__}: {str(exc)[:240]}\nDetails: runs/index_v1/report.json")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
