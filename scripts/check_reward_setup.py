#!/usr/bin/env python3
"""Fail-fast dependency/asset checks for the seven public reward adapters.

This intentionally does not load multi-gigabyte model weights.  It verifies the
Python interfaces and local files needed before ``torchrun`` starts eight ranks.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
from pathlib import Path


PUBLIC_REWARDS = {
    "hpsv2", "clipscore", "pickscore", "aesthetic", "imagereward", "hpsv3", "deqa", "open3"
}


def _require_module(name: str, hint: str) -> object:
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise RuntimeError(f"cannot import {name!r}. {hint}. Original error: {exc}") from exc


def _require_assets(names: tuple[str, ...]) -> None:
    root = Path(os.environ.get("REWARD_CKPT_PATH", Path.cwd() / "reward_ckpts"))
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise RuntimeError(
            f"missing reward assets under {root}: {', '.join(missing)}; "
            "run bash scripts/download_reward_weights.sh and export REWARD_CKPT_PATH"
        )


def check(reward: str, backbone: str) -> None:
    if reward not in PUBLIC_REWARDS:
        raise ValueError(f"unknown public reward: {reward}")
    if reward in {"hpsv2", "open3"}:
        _require_module("open_clip", 'install the standard extras: pip install -e ".[rewards]"')
        _require_assets(("open_clip_pytorch_model.bin", "HPS_v2.1_compressed.pt"))
    if reward == "aesthetic":
        _require_assets(("sac+logos+ava1-l14-linearMSE.pth",))
    if reward == "imagereward":
        _require_module("ImageReward", "follow README.md#reward-model-setup (install ImageReward without dependencies)")

    if reward == "hpsv3":
        module = _require_module("hpsv3", "follow README.md#reward-model-setup (heavy HPSv3 setup)")
        inferencer = getattr(module, "HPSv3RewardInferencer", None)
        if inferencer is None or "differentiable" not in inspect.signature(inferencer).parameters:
            raise RuntimeError("installed hpsv3 lacks the differentiable=True inference interface")
        _require_module("qwen_vl_utils", "follow README.md#reward-model-setup")

    if reward == "deqa":
        _require_module("sentencepiece", "follow README.md#reward-model-setup (heavy DeQA setup)")
        _require_module("icecream", "follow README.md#reward-model-setup (heavy DeQA setup)")

    if reward in {"hpsv3", "deqa"}:
        transformers = _require_module("transformers", "install the project dependencies")
        if not str(transformers.__version__).startswith("4.51."):
            raise RuntimeError(
                f"heavy scorers require the validated Transformers 4.51.x compatibility stack; "
                f"found {transformers.__version__}. Follow README.md#reward-model-setup"
            )
        if backbone == "zimage":
            diff_bridge = os.environ.get("ZIMAGE_HEAVY_DIFF_BRIDGE", "0") == "1"
            scalar_bridge = os.environ.get("ZIMAGE_HEAVY_BRIDGE", "0") == "1"
            if diff_bridge == scalar_bridge:
                raise RuntimeError(
                    "Z-Image HPSv3/DeQA requires exactly one heavy-reward bridge: "
                    "ZIMAGE_HEAVY_DIFF_BRIDGE=1 for DiffusionOPSD/ReFL or "
                    "ZIMAGE_HEAVY_BRIDGE=1 for DiffusionNFT/FlowGRPO"
                )
            policy = int(os.environ.get("PUBLIC_POLICY_WORLD_SIZE", "0"))
            launch = int(os.environ.get("PUBLIC_LAUNCH_WORLD_SIZE", "0"))
            if launch != policy + 1:
                raise RuntimeError(
                    f"heavy Z-Image topology must be policy+server; got policy={policy}, launch={launch}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward", required=True)
    parser.add_argument("--backbone", choices=("sd35", "zimage"), required=True)
    args = parser.parse_args()
    check(args.reward, args.backbone)
    print(f"[ok] reward setup: {args.backbone}/{args.reward}")


if __name__ == "__main__":
    main()
