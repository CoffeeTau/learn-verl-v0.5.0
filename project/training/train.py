"""V1 smoke/main launcher. Full logs on disk, screenshot-sized status on stdout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from project.common import code_info, model_inventory, resource_path, save_json, sha256
from project.training.summarize import summarize


def build_config(out, data_dir, prepared, gpus, steps, smoke, version="v1", policy_path=None, overrides=None):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    repo = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(repo / "verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer")
    values = {
        "algorithm.adv_estimator": "grpo", "algorithm.use_kl_in_reward": False,
        "data.train_files": [str(data_dir / "train.parquet")], "data.val_files": [str(data_dir / "dev.parquet")],
        "data.train_batch_size": 16, "data.max_prompt_length": 512, "data.max_response_length": 8192,
        "data.seed": 42, "data.return_raw_chat": True, "data.dataloader_num_workers": 0, "data.truncation": "error",
        "actor_rollout_ref.model.path": str(policy_path or resource_path("AGENTIC_MODEL_DIR")),
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 16,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.actor.optim.weight_decay": 0.0,
        "actor_rollout_ref.actor.use_kl_loss": True,
        "actor_rollout_ref.actor.kl_loss_coef": 0.001,
        "actor_rollout_ref.actor.use_torch_compile": False,
        "actor_rollout_ref.actor.update_probe": True,
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.name": "vllm", "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.seed": 42, "actor_rollout_ref.rollout.n": 4, "actor_rollout_ref.rollout.temperature": 0.7,
        "actor_rollout_ref.rollout.top_p": 0.8, "actor_rollout_ref.rollout.top_k": 20,
        "actor_rollout_ref.rollout.max_model_len": 8192,
        "actor_rollout_ref.rollout.max_num_seqs": 8,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.45,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.multi_turn.enable": True,
        "actor_rollout_ref.rollout.agent.num_workers": gpus,
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(out / "agent_loop.yaml"),
        "reward_model.reward_manager": "agentic_f1",
        "reward_model.external_lib": "project.training.reward",
        "reward_model.reward_kwargs": {"audit_dir": str(out / "reward_audit"),
                                        "max_searches": prepared["task"]["max_searches"],
                                        "max_new_tokens": prepared["task"]["max_new_tokens"]},
        "trainer.project_name": "agentic_search", "trainer.experiment_name": out.name,
        "trainer.metrics_jsonl": str(out / "metrics.jsonl"),
        "trainer.logger": ["jsonl", "console"], "trainer.nnodes": 1, "trainer.n_gpus_per_node": gpus,
        "trainer.total_epochs": 1, "trainer.total_training_steps": steps,
        "trainer.save_freq": 1 if smoke else 25, "trainer.test_freq": -1,
        "trainer.val_before_train": False, "trainer.resume_mode": "disable",
        "trainer.default_local_dir": str(out / "checkpoints"), "trainer.max_actor_ckpt_to_keep": 2,
        "agentic": {"task": prepared["task"], "retriever_name": "retriever_" + out.name,
                    "corpus": str(resource_path("AGENTIC_CORPUS_DIR")),
                    "index": str(resource_path("AGENTIC_INDEX_DIR")),
                    "retriever": str(resource_path("AGENTIC_RETRIEVER_DIR")),
                    "corpus_sha256": prepared["corpus_sha256"]},
    }
    if version == "v2":
        values.update({"reward_model.reward_kwargs.version": "v2",
                       "trainer.logger": ["jsonl", "tensorboard", "console"],
                       "trainer.tensorboard_dir": str(resource_path("AGENTIC_ROOT") / "tensorboard" / ("v2_" + out.name)),
                       "trainer.val_before_train": not smoke, "trainer.test_freq": -1 if smoke else 25,
                       "data.val_batch_size": 32,
                       "actor_rollout_ref.rollout.val_kwargs.temperature": 0.0,
                       "actor_rollout_ref.rollout.val_kwargs.top_p": 1.0,
                       "actor_rollout_ref.rollout.val_kwargs.n": 1,
                       "actor_rollout_ref.rollout.val_kwargs.do_sample": False,
                       "agentic.hard_plan": str(out / "hard_plan.json"),
                       "agentic.retrieval_audit": str(out / "retrieval_audit.jsonl")})
    values.update(overrides or {})
    # Known upstream keys must exist; only explicit project extension keys may be new.
    new_keys = {"trainer.tensorboard_dir", "reward_model.reward_kwargs.version", "agentic.hard_plan",
                "agentic.retrieval_audit", "trainer.metrics_jsonl", "agentic", "reward_model.external_lib", "reward_model.reward_kwargs",
                "actor_rollout_ref.actor.update_probe", "data.seed", "actor_rollout_ref.rollout.seed",
                "trainer.custom_trainer"}
    for key, value in values.items():
        if key not in new_keys and OmegaConf.select(config, key, default="__missing__") == "__missing__":
            raise KeyError(f"Unexpected veRL config field: {key}")
        OmegaConf.update(config, key, value, force_add=key in new_keys, merge=False)
    loops = [{"name": "agentic_search", "_target_": "project.training.agent_loop.SearchAgentLoop"}]
    if version == "v2":
        loops = [{"name": "agentic_correction_" + split,
                  "_target_": "project.training.agent_loop.CorrectionAgentLoop", "perturb": split == "train"}
                 for split in ("train", "eval")]
    OmegaConf.save(OmegaConf.create(loops), out / "agent_loop.yaml")
    OmegaConf.save(config, out / "config.yaml", resolve=True)
    return config


def print_progress(out, steps, version):
    path = out / "metrics.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "actor/grad_norm" in row.get("metrics", {}):
                rows.append(row)
    if not rows:
        print(f"{version.upper()} running: setup/initial validation/first update; see train.log", flush=True)
        return
    last = rows[-1]
    reward = last["metrics"].get("critic/score/mean", float("nan"))
    print(f"{version.upper()} completed {last['step']}/{steps} | train reward={reward:.3f} | see TensorBoard/train.log", flush=True)


def main(version="v1"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "main"), default="smoke")
    parser.add_argument("--gpus", type=int, choices=(2, 4, 8), default=8)
    if version == "v2":
        parser.add_argument("--init-run", default="main_20261006T090840Z_c41e35")
    args = parser.parse_args()
    steps = 2 if args.mode == "smoke" else 125
    run_id = args.mode + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    out = resource_path("AGENTIC_RUNS_DIR") / version / run_id
    out.mkdir(parents=True)
    code = 1
    try:
        from project.training.prepare import prepare
        policy_path = resource_path("AGENTIC_MODEL_DIR")
        parent_run = None
        if version == "v2":
            import json
            import re
            if not re.fullmatch(r"[A-Za-z0-9_-]+", args.init_run):
                raise ValueError("Invalid V1 run ID")
            from project.evaluation.v1 import export_model
            parent_run = resource_path("AGENTIC_RUNS_DIR") / "v1" / args.init_run
            parent_report = json.loads((parent_run / "report.json").read_text())
            parent_manifest = json.loads((parent_run / "manifest.json").read_text())
            if parent_report["status"] != "PASSED" or parent_manifest["mode"] != "main":
                raise ValueError("V2 requires a completed V1 main run")
            policy_path = export_model(parent_run, int(parent_manifest["steps"]))
            from torch.utils.tensorboard import SummaryWriter  # dependency check before GPU setup
        data_dir, prepared = prepare(version=version)
        if version == "v2":
            if parent_manifest["data"]["source_manifest_sha256"] != prepared["source_manifest_sha256"]:
                raise ValueError("Frozen data differs from V1 training")
            from project.training.hard_episodes import prepare_plan
            prepared["hard_episodes"] = prepare_plan(resource_path("AGENTIC_PROCESSED_DATA_DIR"),
                                                       resource_path("AGENTIC_CORPUS_DIR"), out / "hard_plan.json")
            print("V2 hard episode plan:", prepared["hard_episodes"]["modes"], flush=True)
        import torch
        if torch.cuda.device_count() < args.gpus:
            raise ValueError(f"Need {args.gpus} visible GPUs; found {torch.cuda.device_count()}")
        build_config(out, data_dir, prepared, args.gpus, steps, args.mode == "smoke", version, policy_path)
        repo = Path(__file__).resolve().parents[2]
        core_files = ["verl/utils/tracking.py", "verl/trainer/ppo/reward.py", "verl/workers/actor/dp_actor.py",
                      "verl/workers/rollout/vllm_rollout/vllm_async_server.py"]
        save_json(out / "manifest.json", {"code": code_info(), "core_sha256": {p: sha256(repo / p) for p in core_files}, "data": prepared, "mode": args.mode,
                                          "policy": model_inventory(policy_path), "version": version,
                                          "parent_run": str(parent_run) if parent_run else None,
                                          "reward": "canonical terminal F1; KL coefficient 0.001", "steps": steps})
        print(f"{version.upper()} {args.mode}: {steps} updates | GPUs={args.gpus} | 16 questions x 4 samples\n"
              f"Log: {out.relative_to(resource_path('AGENTIC_ROOT'))}/train.log", flush=True)
        env = dict(os.environ)
        env.update(VLLM_USE_V1="1", VLLM_WORKER_MULTIPROC_METHOD="spawn", TOKENIZERS_PARALLELISM="false",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1")
        with (out / "train.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "project.training.worker", "--config", str(out / "config.yaml")],
                                       stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                while process.poll() is None:
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        print_progress(out, steps, version)
                code = process.returncode
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
    except Exception:
        import traceback
        with (out / "train.log").open("a") as log:
            traceback.print_exc(file=log)
    passed = summarize(out, steps, code, version=version)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
