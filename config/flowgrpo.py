"""Paper-matched SD3.5-M FlowGRPO control configuration.

The manuscript includes one locally trained, reward-matched SD3.5-M
FlowGRPO CLIPScore run in the training-dynamics analysis and uses the same
configuration for the short efficiency profile. The main-table SD3.5-M
FlowGRPO row remains the upstream public-checkpoint reference.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import ml_collections


ROOT = Path(__file__).resolve().parents[1]


def _load_api():
    path = Path(__file__).with_name("api.py")
    spec = importlib.util.spec_from_file_location("_diffusionopsd_flowgrpo_api", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


API = _load_api()


def get_config(name: str):
    """Return the matched ``sd35_clipscore`` FlowGRPO control."""

    if name not in {"sd35_clipscore", "sd3_clipscore"}:
        raise ValueError(
            "The paper-matched SD3.5-M FlowGRPO control is sd35_clipscore; "
            f"got {name!r}"
        )
    n_gpus = int(
        os.environ.get(
            "PUBLIC_POLICY_WORLD_SIZE",
            os.environ.get("PUBLIC_N_GPUS", "8"),
        )
    )
    config = API.sd3_reward_config("clipscore", dataset="pickscore", n_gpus=n_gpus)
    config.dataset = str(ROOT / "data" / "pickapic")

    # FlowGRPO needs non-degenerate SDE transition densities and trains on all
    # transitions. Two optimizer updates are made during one replay pass.
    config.sample.solver = "flow"
    config.sample.deterministic = False
    config.sample.noise_level = 0.7
    config.train.beta = 0.0
    config.train.timestep_fraction = 1.0
    optimizer_updates_per_rollout = 2
    num_train_batches = (
        config.sample.num_batches_per_epoch
        * config.sample.train_batch_size
        // config.train.batch_size
    )
    if num_train_batches % optimizer_updates_per_rollout:
        raise ValueError(
            f"FlowGRPO rollout batches ({num_train_batches}) must divide into "
            f"{optimizer_updates_per_rollout} optimizer windows"
        )
    config.train.gradient_accumulation_steps = (
        num_train_batches // optimizer_updates_per_rollout
    )
    config.flowgrpo = ml_collections.ConfigDict(
        {
            "group_size": config.sample.num_image_per_prompt,
            "clip_range": 1e-4,
            "optimizer_updates_per_rollout": optimizer_updates_per_rollout,
        }
    )

    # Fifty rollout/replay rounds produce the paper's 100 optimizer updates.
    config.num_epochs = 50
    config.save_freq = 10
    config.eval_freq = 0
    config.run_name = "flowgrpo_sd35_clipscore"
    config.save_dir = str(ROOT / "outputs" / "flowgrpo" / "sd35_clipscore")
    config.logdir = str(ROOT / "outputs" / "wandb")
    return config
