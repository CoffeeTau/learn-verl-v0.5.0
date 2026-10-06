# Reused reference implementations

## Search-R1 (Apache-2.0)

Source checkout: `project_resource/Search-R1/`.

- `project/retrieval/e5.py`: adapted masked mean pooling, query/passage prefixes and L2 normalization from `search_r1/search/retrieval_server.py` (`pooling`, `Encoder.encode`). Changed to local-only loading and CPU operation; removed FAISS/server dependencies for the resource smoke check.
- `project/scripts/smoke.py`: follows the `<search>` / `<information>` / `<answer>` protocol and search-then-continue flow in `infer.py`. Reimplemented generation with vLLM and Qwen3 non-thinking chat messages, bounded to one tool roundtrip. This is not a copy of its full RL rollout.

Original license is included in `licenses/Search-R1-LICENSE`. Original files above have no file-level copyright header; contributor attribution and license are retained here and in the adapted encoder.

The E5 implementation also agrees with the [official model card](https://huggingface.co/intfloat/e5-base-v2). Smoke scripts do not import another checkout's `verl` or install its dependencies.
