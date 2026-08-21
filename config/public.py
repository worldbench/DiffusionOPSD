"""Downloadable-weight public configurations for DiffusionOPSD.

Examples:
    --config config/public.py:sd35_hpsv2
    --config config/public.py:zimage_clipscore

Run ``python scripts/prepare_pickapic_prompts.py`` once before loading a
configuration.  The public launchers do this automatically.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API_PATH = Path(__file__).with_name("api.py")
_SPEC = importlib.util.spec_from_file_location("_diffusionopsd_public_api", API_PATH)
API = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(API)

PUBLIC_REWARDS = {
    "hpsv2", "clipscore", "pickscore", "aesthetic", "imagereward", "hpsv3", "deqa"
}
HEAVY_ZIMAGE_REWARDS = {"hpsv3", "deqa"}


def get_config(name: str):
    """Return ``<backbone>_<reward>`` or the SD3.5 joint ``sd35_open3`` preset."""

    try:
        backbone, reward = name.split("_", 1)
    except ValueError as exc:
        raise ValueError(
            f"Expected <backbone>_<reward>, e.g. sd35_hpsv2; got {name!r}"
        ) from exc

    heavy_zimage = backbone == "zimage" and reward in HEAVY_ZIMAGE_REWARDS
    default_policy_world = "6" if heavy_zimage else "8"
    n_gpus = int(os.environ.get("PUBLIC_POLICY_WORLD_SIZE", os.environ.get("PUBLIC_N_GPUS", default_policy_world)))
    if n_gpus < 1:
        raise ValueError(f"PUBLIC_POLICY_WORLD_SIZE must be positive, got {n_gpus}")
    if heavy_zimage:
        if os.environ.get("ZIMAGE_HEAVY_DIFF_BRIDGE", "0") != "1":
            raise ValueError(
                f"zimage/{reward} requires ZIMAGE_HEAVY_DIFF_BRIDGE=1; "
                "use scripts/train_public.sh or follow README.md#reward-model-setup"
            )
        launch_world = int(os.environ.get("PUBLIC_LAUNCH_WORLD_SIZE", str(n_gpus + 1)))
        if launch_world != n_gpus + 1:
            raise ValueError("heavy Z-Image needs one reward-server rank in addition to policy ranks")

    if backbone == "sd35":
        if reward == "open3":
            base_reward = "pickscore"
        elif reward in PUBLIC_REWARDS:
            base_reward = reward
        else:
            raise ValueError(f"Unsupported public SD3.5 reward: {reward}")

        config = API.sd3_reward_config(base_reward, dataset="pickscore", n_gpus=n_gpus)
        params = API.sd35m_opsd_default_params()
        if reward == "open3":
            config.reward_fn = {"pickscore": 1.0, "clipscore": 1.0, "hpsv2": 1.0}
            # PickScore is normalized by /26 in its scorer, matching the paper's
            # PickScore/26 + CLIPScore + HPSv2.1 objective.
            config.beta = 0.1
            params["opa_reward_kind"] = "open3"
        elif reward == "imagereward":
            # Paper recipe for the memory-heavy SD3.5-M ImageReward target pass.
            params["opa_mb"] = 1
        elif reward == "deqa":
            # DeQA is an 8B mPLUG-Owl2 scorer; the paper uses one target per
            # differentiable reward micro-batch on SD3.5-M.
            params["opa_mb"] = 1
        API.attach_opsd(config, params)

    elif backbone == "zimage":
        if reward not in PUBLIC_REWARDS:
            raise ValueError(f"Unsupported public Z-Image reward: {reward}")
        config = API.zimage_config("opsd", reward=reward, n_gpus=n_gpus)

    else:
        raise ValueError(f"Unsupported backbone {backbone!r}; choose sd35 or zimage")

    config.dataset = str(ROOT / "data" / "pickapic")
    default_updates = 300 if name == "sd35_open3" else 100
    config.num_epochs = int(os.environ.get("PUBLIC_NUM_UPDATES", str(default_updates)))
    config.save_freq = int(os.environ.get("PUBLIC_SAVE_FREQ", "10"))
    config.eval_freq = 0
    config.run_name = f"public_{name}"
    config.save_dir = str(ROOT / "outputs" / name)
    config.logdir = str(ROOT / "outputs" / "wandb")
    return config
