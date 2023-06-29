import torch

torch.backends.cuda.matmul.allow_tf32 = True
import json
import os
import socket
from typing import Optional, Set

import hydra
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import transformers
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record

import trainers
from utils import disable_dropout, get_local_dir, get_local_run_dir, init_distributed

OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


def worker_main(rank_idx: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""

    global_rank = os.environ.get("RANK", None)
    if not global_rank:
        global_rank = rank_idx

    global_rank = int(global_rank)

    if "FSDP" in config.trainer:
        print(f"GLOBAL_RANK / WORLD = {global_rank} / {world_size}")
        init_distributed(global_rank, world_size)

    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if global_rank == 0 and config.wandb.enabled:
        os.environ["WANDB_CACHE_DIR"] = get_local_dir(config.local_dirs)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )

    TrainerClass = getattr(trainers, config.trainer)
    print(f"Creating trainer on process {global_rank} with world size {world_size}")
    trainer = TrainerClass(
        policy,
        config,
        config.seed,
        config.local_run_dir,
        reference_model=reference_model,
        rank=global_rank,
        world_size=world_size,
    )

    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print("WARNING: eval_every must be divisible by batch_size")
        print("Setting eval_every to", config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)

    print("=" * 80)
    print(f"Writing to {socket.gethostname()}:{config.local_run_dir}")
    print("=" * 80)

    os.environ["XDG_CACHE_HOME"] = get_local_dir(config.local_dirs)

    print(f"building policy on: {os.environ.get('RANK', None)}")
    model_kwargs = {"device_map": "balanced"} if config.trainer == "BasicTrainer" else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        cache_dir=get_local_dir(config.local_dirs),
        torch_dtype=policy_dtype,
        low_cpu_mem_usage=True,
        **model_kwargs,
    )
    disable_dropout(policy)
    print(f"Policy build OK: {os.environ.get('RANK', None)}")

    if config.loss.name == "dpo":
        print(f"building reference model: {os.environ.get('RANK', None)}")
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            cache_dir=get_local_dir(config.local_dirs),
            torch_dtype=reference_model_dtype,
            low_cpu_mem_usage=True,
            **model_kwargs,
        )
        disable_dropout(reference_model)
        print(f"Reference build OK: {os.environ.get('RANK', None)}")
    else:
        reference_model = None

    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location="cpu")
        step, metrics = state_dict["step_idx"], state_dict["metrics"]
        print(f"loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}")
        policy.load_state_dict(state_dict["state"])
        if config.loss.name == "dpo":
            reference_model.load_state_dict(state_dict["state"])
        print("loaded pre-trained weights")

    if "FSDP" in config.trainer:
        world_size = int(os.environ.get("WORLD_SIZE"))
        print("starting", world_size, "processes for FSDP training")
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, policy, reference_model), join=True)
    else:
        print("starting single-process worker")
        worker_main(0, 1, config, policy, reference_model)


if __name__ == "__main__":
    main()
