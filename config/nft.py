"""Paper-aligned SD3.5-M configurations for public DiffusionNFT/OPSD runs."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_REWARDS = {
    "hpsv2",
    "clipscore",
    "pickscore",
    "aesthetic",
    "imagereward",
    "hpsv3",
    "deqa",
}
# These evaluators use internal fine-tuned weights. Their implementations are
# retained for provenance, but the public release does not provide checkpoints.
INTERNAL_REWARDS = {"altclip", "internvl_t2i", "internvl_dual"}


def _load_local(name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(f"_diffusionopsd_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load configuration module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_local("base.py")


def _get_config(
    base_model: str = "sd3",
    n_gpus: int = 8,
    gradient_step_per_epoch: int = 1,
    dataset: str = "pickscore",
    reward_fn: dict[str, float] | None = None,
    name: str = "",
    num_image_per_prompt: int = 24,
):
    """Build the SD3.5-M recipe reported in the paper.

    ``n_gpus`` is the policy world size. With eight policy GPUs the search
    below yields batch size 9 and gradient accumulation 16, i.e. 48 prompt
    groups times 24 images and one optimizer update per sampling round.
    """

    if base_model != "sd3":
        raise ValueError(f"Only the paper's SD3.5-M backbone is supported, got {base_model!r}")
    if dataset not in {"pickscore", "internvl_dual"}:
        raise ValueError(f"Unsupported public dataset: {dataset!r}")
    if n_gpus < 1 or gradient_step_per_epoch < 1:
        raise ValueError("n_gpus and gradient_step_per_epoch must be positive")

    config = base.get_config()
    config.base_model = base_model
    config.dataset = str(ROOT / "data" / ("pickapic" if dataset == "pickscore" else dataset))
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.pretrained.revision = ""
    config.resolution = 512
    config.mixed_precision = "fp16"

    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 1.0
    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    config.sample.noise_level = 0.7
    config.sample.num_image_per_prompt = int(num_image_per_prompt)

    num_groups = 48
    batch_size = 9
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
        raise ValueError(
            f"Cannot satisfy 48x{num_image_per_prompt} grouping with {n_gpus} policy GPUs"
        )

    config.sample.test_batch_size = 16 if n_gpus <= 32 else 8
    config.prompt_fn = "general_ocr"
    config.reward_fn = dict(reward_fn or {})

    config.num_epochs = 100
    config.save_freq = 10
    config.eval_freq = 0
    config.run_name = f"nft_sd3_{name or 'run'}"
    config.save_dir = str(ROOT / "outputs" / "nft" / (name or "run"))
    config.logdir = str(ROOT / "outputs" / "wandb")

    # DiffusionNFT objective and behavior-policy tracking.
    config.decay_type = 1
    config.beta = 1.0
    config.train.beta = 1e-4
    config.train.adv_mode = "all"
    return config


def get_config(name: str):
    """Return ``sd3_<reward>`` or the joint ``sd3_multi_reward`` preset."""

    n_gpus = int(
        os.environ.get(
            "PUBLIC_POLICY_WORLD_SIZE",
            os.environ.get("PUBLIC_N_GPUS", "8"),
        )
    )

    if name in {"sd3_multi_reward", "sd3_open3"}:
        config = _get_config(
            n_gpus=n_gpus,
            reward_fn={"pickscore": 1.0, "clipscore": 1.0, "hpsv2": 1.0},
            name="multi_reward",
        )
        config.beta = 0.1
        config.num_epochs = 300
        return config

    prefix = "sd3_"
    if not name.startswith(prefix):
        raise ValueError(f"Expected sd3_<reward>, got {name!r}")
    reward = name[len(prefix):]
    if reward not in PUBLIC_REWARDS | INTERNAL_REWARDS:
        raise ValueError(f"Unsupported SD3.5-M reward: {reward!r}")
    dataset = "internvl_dual" if reward == "internvl_dual" else "pickscore"
    return _get_config(
        n_gpus=n_gpus,
        dataset=dataset,
        reward_fn={reward: 1.0},
        name=reward,
    )
