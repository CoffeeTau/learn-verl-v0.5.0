"""veRL 0.5.0 AgentLoop adapter. Gold labels never enter this interface."""
import html
import time
import uuid
import ray
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
from project.agent.search import SYSTEM_PROMPT, parse_action
from project.training.protocol import continuation_ids


class SearchAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        # Once per worker, the actor is shared across all episode objects.
        if cls.__dict__.get("_ready", False):
            return
        cls.settings = dict(config.agentic.task)
        cls.retriever = ray.get_actor(config.agentic.retriever_name)
        cls._ready = True

    async def run(self, messages, sampling_params):
        cfg = self.settings
        version = getattr(self, "version", "v1")
        prompt_text = SYSTEM_PROMPT
        if version == "v2":
            from project.agent.correction import SYSTEM_PROMPT as prompt_text
            from project.agent.correction import parse_action as parse_correction
        messages = list(messages)
        if len(messages) != 2 or messages[0]["content"] != prompt_text.format(max_searches=cfg["max_searches"]):
            raise ValueError("Unexpected training prompt")
        question = messages[1]["content"]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    enable_thinking=False)
        if len(prompt) > self.config.actor_rollout_ref.rollout.prompt_length:
            raise ValueError("Prompt exceeds padded prompt budget")
        response, mask = [], []
        metrics = AgentLoopMetrics()
        request_id = uuid.uuid4().hex
        params = dict(sampling_params)
        params.update(top_k=self.config.actor_rollout_ref.rollout.top_k, max_tokens=cfg["max_new_tokens"], stop=["</search>", "</answer>"],
                      include_stop_str_in_output=True)
        turns = 0
        for search_count in range(cfg["max_searches"] + 1):
            if len(prompt) + len(response) + cfg["max_new_tokens"] > cfg["max_context_tokens"]:
                break
            start = time.monotonic()
            tokens = await self.server_manager.generate(request_id, prompt_ids=prompt + response,
                                                        sampling_params=params)
            metrics.generate_sequences += time.monotonic() - start
            if not tokens:
                raise RuntimeError("Empty engine response")
            response.extend(tokens)
            mask.extend([1] * len(tokens))
            turns += 1
            # The engine interface returns IDs only. At the exact cap, be conservative.
            if len(tokens) >= cfg["max_new_tokens"]:
                break
            try:
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                action = parse_correction(text, after_search=search_count > 0) if version == "v2" else parse_action(text)
            except ValueError:
                break
            if action["kind"] == "answer" or search_count == cfg["max_searches"]:
                break
            start = time.monotonic()
            if version == "v2":
                hits = await self.retriever.search_v2.remote(question, action["query"], cfg["top_k"],
                                                            search_count, self.perturb)
            else:
                hits = await self.retriever.search.remote(question, action["query"], cfg["top_k"])
            metrics.tool_calls += time.monotonic() - start
            observation = "\n\n".join(f"[{hit['id']}] {html.escape(hit['title'])}\n{html.escape(hit['text'])}"
                                          for hit in hits)
            observation = f"<information>\n{observation}\n</information>\nSearches remaining: {cfg['max_searches'] - search_count - 1}."
            inserted = continuation_ids(self.tokenizer, observation, tokens)
            # Finish on the last sampled action if another generation cannot fit.
            if len(prompt) + len(response) + len(inserted) + cfg["max_new_tokens"] > cfg["max_context_tokens"]:
                break
            response.extend(inserted)
            mask.extend([0] * len(inserted))
            turns += 1
        if not response or len(response) > self.config.actor_rollout_ref.rollout.response_length:
            raise ValueError("Invalid trajectory length")
        return AgentLoopOutput(prompt_ids=prompt, response_ids=response, response_mask=mask,
                               num_turns=turns, metrics=metrics)


class CorrectionAgentLoop(SearchAgentLoop):
    def __init__(self, trainer_config, server_manager, tokenizer, perturb=False, **kwargs):
        super().__init__(trainer_config, server_manager, tokenizer, **kwargs)
        self.version = "v2"
        # Train and validation have different registry entries. This flag is never
        # inferred from temperature or the content of the question.
        self.perturb = bool(perturb)
