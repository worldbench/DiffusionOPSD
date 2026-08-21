"""Paper-aligned Z-Image-Turbo configurations."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import ml_collections


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_REWARDS = (
    "hpsv2",
    "clipscore",
    "pickscore",
    "aesthetic",
    "imagereward",
    "hpsv3",
    "deqa",
)


def _load_local(name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(f"_diffusionopsd_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load configuration module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_local("base.py")
opsd_defaults = _load_local("opsd_defaults.py")


def _get_config(
    base_model: str = "zimage",
    n_gpus: int = 8,
    gradient_step_per_epoch: int = 1,
    dataset: str = "pickscore",
    reward_fn: dict[str, float] | None = None,
    name: str = "",
):
    """Build the native 9-step, guidance-free Z-Image-Turbo recipe."""

    if base_model != "zimage":
        raise ValueError(f"Only Z-Image-Turbo is supported, got {base_model!r}")
    if dataset != "pickscore":
        raise ValueError(f"Unsupported public dataset: {dataset!r}")
    if n_gpus < 1 or gradient_step_per_epoch < 1:
        raise ValueError("n_gpus and gradient_step_per_epoch must be positive")

    config = base.get_config()
    config.base_model = base_model
    config.dataset = str(ROOT / "data" / "pickapic")
    config.pretrained.model = "Tongyi-MAI/Z-Image-Turbo"
    config.pretrained.revision = ""
    config.mixed_precision = "bf16"
    config.resolution = 1024

    config.sample.num_steps = 9
    config.sample.eval_num_steps = 9
    config.sample.guidance_scale = 0.0
    config.sample.deterministic = True
    config.sample.solver = "flow_euler"
    config.sample.noise_level = 1.0
    config.sample.num_image_per_prompt = 12

    num_groups = 48
    batch_size = 6
    while batch_size >= 1:
        total = num_groups * config.sample.num_image_per_prompt
        if total % (n_gpus * batch_size) == 0 and batch_size * n_gpus % config.sample.num_image_per_prompt == 0:
            num_batches = total // (n_gpus * batch_size)
            if num_batches % gradient_step_per_epoch == 0:
                config.sample.train_batch_size = batch_size
                config.sample.num_batches_per_epoch = num_batches
                config.train.batch_size = batch_size
                config.train.gradient_accumulation_steps = num_batches // gradient_step_per_epoch
                break
        batch_size -= 1
    else:
        raise ValueError(f"Cannot satisfy 48x12 grouping with {n_gpus} policy GPUs")

    config.sample.test_batch_size = 8 if n_gpus <= 32 else 4
    config.prompt_fn = "general_ocr"
    config.reward_fn = dict(reward_fn or {})
    config.num_epochs = 100
    config.save_freq = 10
    config.eval_freq = 0
    config.run_name = f"zimg_{name or 'run'}"
    config.save_dir = str(ROOT / "outputs" / "zimage" / (name or "run"))
    config.logdir = str(ROOT / "outputs" / "wandb")

    config.decay_type = 1
    config.beta = 1.0
    config.train.beta = 1e-4
    config.train.adv_mode = "all"
    return config


def zimg_nft(reward: str = "hpsv2", n_gpus: int = 8):
    """Return the paper-matched reward-specific DiffusionNFT preset."""

    if reward not in PUBLIC_REWARDS:
        raise ValueError(f"Unsupported public Z-Image DiffusionNFT reward: {reward!r}")
    return _get_config(
        n_gpus=n_gpus,
        reward_fn={reward: 1.0},
        name=f"nft_{reward}",
    )


def zimg_nft_hps(n_gpus: int = 8):
    """Backward-compatible alias for the original HPSv2.1 preset name."""

    return zimg_nft("hpsv2", n_gpus=n_gpus)


def zimg_opsd_hps(n_gpus: int = 8):
    config = zimg_nft_hps(n_gpus=n_gpus)
    config.opsd = ml_collections.ConfigDict(opsd_defaults.zimage_turbo_opsd_default_params())
    config.opsd.opa_mb = 2
    config.train.beta = 0.0
    config.run_name = "zimg_opsd_hps"
    config.save_dir = str(ROOT / "outputs" / "zimage" / "opsd_hps")
    return config


def zimg_flowgrpo(reward: str = "hpsv2", n_gpus: int = 8):
    """Return the paper-matched reward-specific FlowGRPO preset."""

    config = zimg_nft(reward, n_gpus=n_gpus)
    config.flowgrpo = ml_collections.ConfigDict(
        {
            "group_size": config.sample.num_image_per_prompt,
            "clip_range": 1e-4,
            "optimizer_updates_per_rollout": 2,
        }
    )
    config.sample.deterministic = False
    config.sample.solver = "flow_sde"
    config.sample.noise_level = 0.7
    config.train.beta = 0.0
    config.train.timestep_fraction = 1.0
    num_train_batches = (
        config.sample.num_batches_per_epoch
        * config.sample.train_batch_size
        // config.train.batch_size
    )
    updates = config.flowgrpo.optimizer_updates_per_rollout
    if num_train_batches % updates:
        raise ValueError(
            f"FlowGRPO rollout batches ({num_train_batches}) must divide into {updates} updates"
        )
    config.train.gradient_accumulation_steps = num_train_batches // updates
    # Two optimizer updates per sampling round: 50 rounds are 100 updates.
    config.num_epochs = 50
    config.run_name = f"zimg_flowgrpo_{reward}"
    config.save_dir = str(ROOT / "outputs" / "zimage" / f"flowgrpo_{reward}")
    return config


def zimg_flowgrpo_hps(n_gpus: int = 8):
    """Backward-compatible alias for the original HPSv2.1 preset name."""

    return zimg_flowgrpo("hpsv2", n_gpus=n_gpus)


def get_config(name: str):
    """Resolve ``zimg_{nft|flowgrpo}_<reward>`` public baseline presets."""

    n_gpus = int(
        os.environ.get(
            "PUBLIC_POLICY_WORLD_SIZE",
            os.environ.get("PUBLIC_N_GPUS", "8"),
        )
    )
    aliases = {
        "zimg_nft_hps": "zimg_nft_hpsv2",
        "zimg_opsd_hps": "zimg_opsd_hpsv2",
        "zimg_flowgrpo_hps": "zimg_flowgrpo_hpsv2",
    }
    name = aliases.get(name, name)
    try:
        prefix, method, reward = name.split("_", 2)
    except ValueError as exc:
        raise ValueError(
            f"Expected zimg_{{nft|flowgrpo}}_<reward>, got {name!r}"
        ) from exc
    if prefix != "zimg" or reward not in PUBLIC_REWARDS:
        raise ValueError(f"Unsupported Z-Image preset: {name!r}")
    if method == "nft":
        return zimg_nft(reward, n_gpus=n_gpus)
    if method == "flowgrpo":
        return zimg_flowgrpo(reward, n_gpus=n_gpus)
    if method == "opsd" and reward == "hpsv2":
        return zimg_opsd_hps(n_gpus=n_gpus)
    raise ValueError(f"Unsupported Z-Image preset: {name!r}")
