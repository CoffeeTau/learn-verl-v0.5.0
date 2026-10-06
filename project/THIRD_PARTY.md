# Reused reference implementations

## Search-R1 (Apache-2.0)

Source checkout: `project_resource/Search-R1/`.

- `project/retrieval/e5.py`: adapted masked mean pooling, query/passage prefixes and L2 normalization from `search_r1/search/retrieval_server.py` (`pooling`, `Encoder.encode`). Changed to local-only loading and CPU operation; removed FAISS/server dependencies for the resource smoke check.
- `project/scripts/smoke.py`: follows the `<search>` / `<information>` / `<answer>` protocol and search-then-continue flow in `infer.py`. Reimplemented generation with vLLM and Qwen3 non-thinking chat messages, bounded to one tool roundtrip. This is not a copy of its full RL rollout.

Original license is included in `licenses/Search-R1-LICENSE`. Original files above have no file-level copyright header; contributor attribution and license are retained here and in the adapted encoder.

The E5 implementation also agrees with the [official model card](https://huggingface.co/intfloat/e5-base-v2). Smoke scripts do not import another checkout's `verl` or install its dependencies.

## Mainline additions

- `project/retrieval/e5.py` now supports a configurable device: index construction uses GPU, while V0 queries use CPU. Both retain the same prefix, masked mean pooling and normalization.
- `project/agent/search.py` adapts the Search-R1 action/observation protocol into a bounded loop with a Qwen3 non-thinking backend, sentence-preserving observations and explicit stop reasons. Source: `project_resource/Search-R1/infer.py` and `search_r1/llm_agent/generation.py`. This V0 loop is not yet the veRL training rollout adapter.
- `project/evaluation/metrics.py` adapts `normalize_answer` from Search-R1's Apache-2.0 `verl/utils/reward_score/qa_em.py`, retaining its copyright notice. Token-F1 is implemented to match the answer-only behavior, including special yes/no/noanswer handling, documented in [the official 2Wiki scorer](https://github.com/Alab-NII/2wikimultihop/blob/main/2wikimultihop_evaluate.py). Entity-alias expansion and official joint scores are not implemented or claimed.
- `project/data_pipeline/prepare.py` is project-specific preparation for the downloaded original-shaped 2Wiki files, rather than reuse of Search-R1's processed question/answer-only training dataset. It preserves evidence sentence mappings required by this project's correction and retriever work.

## V1 integration

`project/training/agent_loop.py` implements this checkout’s `AgentLoopBase` / `AgentLoopOutput` interface, following the token/response-mask contract in `verl/experimental/agent_loop/agent_loop.py` and `tool_agent_loop.py` (Apache-2.0). Training uses this checkout’s FSDP, asynchronous vLLM, GRPO advantages and checkpoint machinery, rather than importing another reference repository’s veRL. Search protocol, E5 retrieval and answer scoring are reused from the project modules attributed above.
