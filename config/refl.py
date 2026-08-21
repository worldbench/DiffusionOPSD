"""Paper-matched ReFL configurations for the seven public evaluators.

The internal AltCLIP/VLM evaluator variants are intentionally not exposed by
the public launcher; their adapters remain annotated in ``src/diffusionopsd/``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import ml_collections


ROOT = Path(__file__).resolve().parents[1]
API_PATH = ROOT / "config" / "api.py"
PICKSCORE_DATASET = str(ROOT / "data" / "pickapic")
DRAWBENCH_PROMPTS = str(ROOT / "data" / "drawbench" / "test.txt")

REWARDS = (
    "pickscore", "clipscore", "hpsv2", "aesthetic", "imagereward",
    "hpsv3", "deqa",
)

SPEC = importlib.util.spec_from_file_location("_refl_config_api", API_PATH)
API = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(API)


def _topology(backbone: str, reward: str) -> tuple[int, int, str]:
    """Return policy ranks, launched ranks, and the required reward topology."""
    if backbone == "zimage" and reward in {"hpsv3", "deqa"}:
        if os.environ.get("ZIMAGE_HEAVY_DIFF_BRIDGE", "0") != "1":
            raise ValueError(f"zimage/{reward} requires ZIMAGE_HEAVY_DIFF_BRIDGE=1")
        return 6, 7, "zimage_heavy_diff_server"
    return 8, 8, "colocated"


def _refl_block(backbone: str, reward: str, policy_world_size: int, launch_world_size: int,
                          topology: str) -> ml_collections.ConfigDict:
    trajectories_per_prompt = 24 if backbone == "sd35m" else 12
    trajectories_per_update = 48 * trajectories_per_prompt
    micro_batch_size = 1
    denominator = policy_world_size * micro_batch_size
    if trajectories_per_update % denominator:
        raise ValueError(
            f"invalid batch arithmetic: {trajectories_per_update} / "
            f"({policy_world_size} * {micro_batch_size})"
        )
    accumulation = trajectories_per_update // denominator
    standardize = reward != "imagereward"
    actual_compute_gpu_count = launch_world_size
    return ml_collections.ConfigDict({
        "reward": reward,
        "num_updates": 100,
        "distinct_prompt_groups": 48,
        "trajectories_per_prompt": trajectories_per_prompt,
        "trajectories_per_update": trajectories_per_update,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": accumulation,
        "expected_policy_world_size": policy_world_size,
        "expected_launch_world_size": launch_world_size,
        "reward_topology": topology,
        "actual_compute_gpu_count": actual_compute_gpu_count,
        "reserved_gpu_count": 8,
        "learning_rate": 1e-5,
        "weight_decay": 1e-2,
        "grad_scale": 1e-3,
        "max_grad_norm": 1.0,
        "late_fraction": 0.25,
        "hinge_margin": 2.0,
        "standardize_reward": standardize,
        "calibration_num_prompts": 512,
        "curve_eval_num_prompts": 128,
        "final_eval_num_prompts": 1000,
        "eval_every": 10,
        "checkpoint_every": 10,
        "calibration_manifest_path": str(ROOT / "data" / "pickapic" / "train.txt"),
        "curve_manifest_path": DRAWBENCH_PROMPTS,
        "final_manifest_path": DRAWBENCH_PROMPTS,
        "pairwise_train_ref_root": "",
        "pairwise_eval_ref_root": "",
        # Every seed block is disjoint and recorded in the emitted manifests.
        "seed": 20260814,
        "train_prompt_seed": 2026081400,
        "train_shuffle_seed": 2026081500,
        "train_noise_seed": 202608160000,
        "train_late_seed": 202609160000,
        "calibration_noise_seed": 202610160000,
        "calibration_late_seed": 202611160000,
        "curve_noise_seed": 42,
        "final_noise_seed": 42,
        "eval_generation_batch_size": 16 if backbone == "sd35m" else 8,
        "eval_seed_scheme": "canonical_per_batch_seed_plus_batch_index",
        "save_only_final_checkpoint": False,
        "fail_on_nonfinite_or_zero_grad": True,
    })


def _build(backbone: str, reward: str):
    if backbone not in {"sd35m", "zimage"} or reward not in REWARDS:
        raise ValueError(f"unknown ReFL config: {backbone}/{reward}")
    policy_world, launch_world, topology = _topology(backbone, reward)
    dataset = PICKSCORE_DATASET
    if backbone == "sd35m":
        config = API.sd3_reward_config(reward, dataset="pickscore", n_gpus=policy_world)
        config.sample.num_steps = 10
        config.sample.eval_num_steps = 40
        config.sample.guidance_scale = 1.0
        config.sample.eval_guidance_scale = 1.0
        config.sample.deterministic = True
        config.sample.solver = "dpm2"
        config.mixed_precision = "fp16"
    else:
        config = API.zimage_config("nft", reward=reward, n_gpus=policy_world)
        config.sample.num_steps = 9
        config.sample.guidance_scale = 0.0
        config.mixed_precision = "bf16"
    config.dataset = dataset
    config.refl = _refl_block(
        backbone, reward, policy_world, launch_world, topology
    )
    config.num_epochs = int(os.environ.get("PUBLIC_NUM_UPDATES", "100"))
    config.sample.num_image_per_prompt = 1
    config.sample.train_batch_size = 1
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = config.refl.gradient_accumulation_steps
    config.train.learning_rate = 1e-5
    config.train.adam_weight_decay = 1e-2
    config.train.max_grad_norm = 1.0
    config.train.beta = 0.0
    config.train.ema = True
    config.save_freq = 0
    config.eval_freq = 0
    config.run_name = f"refl_{backbone}_{reward}"
    config.save_dir = str(ROOT / "outputs" / "refl" / f"{backbone}_{reward}")
    config.logdir = str(ROOT / "outputs" / "wandb")
    return config


def get_config(name: str):
    backbone, reward = name.split("_", 1)
    return _build(backbone, reward)


# Explicit names make config enumeration and launcher dry-runs independent of string magic.
def _expose(backbone: str, reward: str):
    def factory():
        return _build(backbone, reward)
    factory.__name__ = f"refl_{backbone}_{reward}"
    factory.__qualname__ = factory.__name__
    return factory


for _backbone in ("sd35m", "zimage"):
    for _reward in REWARDS:
        globals()[f"refl_{_backbone}_{_reward}"] = _expose(_backbone, _reward)
