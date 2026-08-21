"""Public configuration interface.

Paper-matched hyperparameters live in the modules under ``config/``. Public
trainers consume the returned ConfigDict without external experiment manifests.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import ml_collections


def _load(name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(f"_dopsd_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sd3_reward_config(reward: str, *, dataset: str = "pickscore", n_gpus: int = 8):
    nft = _load("nft.py")
    if reward == "internvl_dual":
        dataset = "internvl_dual"
    return nft._get_config(
        base_model="sd3", n_gpus=n_gpus, gradient_step_per_epoch=1,
        dataset=dataset, reward_fn={reward: 1.0}, name=reward,
    )


def attach_opsd(config, params: dict):
    config.opsd = ml_collections.ConfigDict(params)
    # Canonical OPSD omits frozen-base prediction MSE.  A FlowGRPO-style
    # reference regularizer remains an optional experiment, not the default.
    config.train.beta = 0.0
    return config


def opsd_default_params(backbone: str = "sd35m", **overrides):
    defaults = _load("opsd_defaults.py")
    return defaults.opsd_default_params(backbone, **overrides)


def sd35m_opsd_default_params(**overrides):
    return opsd_default_params("sd35m", **overrides)


def zimage_turbo_opsd_default_params(**overrides):
    return opsd_default_params("zimage_turbo", **overrides)


def zimage_config(method: str, *, reward: str = "hpsv2", n_gpus: int = 8):
    # n_gpus is the POLICY world size and drives the sampler/batch math. It is 8 for a co-located
    # run and 6 under ZIMAGE_HEAVY_BRIDGE (NPROC=7 = 6 policy ranks + 1 heavy-reward server rank).
    zimage = _load("zimage.py")
    if method == "nft":
        return zimage.zimg_nft(reward, n_gpus=n_gpus)
    if method == "opsd":
        config = zimage.zimg_opsd_hps(n_gpus=n_gpus)
        config.reward_fn = {reward: 1.0}
        return config
    if method == "flowgrpo":
        return zimage.zimg_flowgrpo(reward, n_gpus=n_gpus)
    raise ValueError(f"Unsupported Z-Image method: {method}")
