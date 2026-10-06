"""Isolated training process: one named retriever plus upstream TaskRunner."""
import argparse
from pathlib import Path


def main():
    import ray
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import get_ppo_ray_runtime_env, run_ppo
    from project.training.retriever import RetrievalActor
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    ray.init(runtime_env=get_ppo_ray_runtime_env(), namespace=config.agentic.retriever_name)
    retriever = RetrievalActor.options(name=config.agentic.retriever_name).remote(
        Path(config.agentic.corpus), Path(config.agentic.index), Path(config.agentic.retriever))
    try:
        if ray.get(retriever.ready.remote()) != config.agentic.corpus_sha256:
            raise ValueError("Training/index corpus mismatch")
        run_ppo(config)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
