"""Configurable SD3.5-M mixed-reward training presets.

Set ``PUBLIC_MIXED_REWARDS`` to a comma-separated weighted objective, for
example ``pickscore=1,clipscore=1,hpsv2=1``. PickScore is already divided by
26 inside its public scorer, so unit weights reproduce the paper's Open3
objective.
"""

from __future__ import annotations

import importlib.util
import math
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
PAPER_OPEN3 = "pickscore=1,clipscore=1,hpsv2=1"


def _load_local(name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(f"_diffusionopsd_mixed_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load configuration module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


API = _load_local("api.py")
NFT = _load_local("nft.py")


def parse_reward_spec(spec: str | None = None) -> dict[str, float]:
    """Parse ``reward[=weight],...`` while preserving the user's order."""

    raw = spec if spec is not None else os.environ.get("PUBLIC_MIXED_REWARDS", PAPER_OPEN3)
    terms = [term.strip() for term in raw.split(",") if term.strip()]
    if len(terms) < 2:
        raise ValueError("Mixed-reward training requires at least two rewards")

    rewards: dict[str, float] = {}
    for term in terms:
        if "=" in term:
            name, value = (part.strip() for part in term.split("=", 1))
        else:
            name, value = term, "1"
        if name not in PUBLIC_REWARDS:
            raise ValueError(f"Unsupported public mixed reward: {name!r}")
        if name in rewards:
            raise ValueError(f"Duplicate mixed reward: {name!r}")
        try:
            weight = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid weight for {name!r}: {value!r}") from exc
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Mixed-reward weight for {name!r} must be finite and positive")
        rewards[name] = weight
    return rewards


def _slug(rewards: dict[str, float]) -> str:
    return "_".join(rewards)


def _opsd_config(rewards: dict[str, float], n_gpus: int):
    first_reward = next(iter(rewards))
    config = API.sd3_reward_config(first_reward, dataset="pickscore", n_gpus=n_gpus)
    config.reward_fn = dict(rewards)
    params = API.sd35m_opsd_default_params()
    params["opa_reward_kind"] = "mixed"
    if set(rewards) & {"imagereward", "deqa"}:
        params["opa_mb"] = 1
    API.attach_opsd(config, params)
    return config


def _nft_config(rewards: dict[str, float], n_gpus: int):
    return NFT._get_config(
        n_gpus=n_gpus,
        dataset="pickscore",
        reward_fn=rewards,
        name=f"mixed_{_slug(rewards)}",
    )


def get_config(name: str):
    """Return ``sd35_opsd`` or ``sd35_nft`` for an arbitrary weighted sum."""

    aliases = {"sd3_opsd": "sd35_opsd", "sd3_nft": "sd35_nft"}
    name = aliases.get(name, name)
    if name not in {"sd35_opsd", "sd35_nft"}:
        raise ValueError(f"Expected sd35_{{opsd|nft}}, got {name!r}")

    n_gpus = int(
        os.environ.get(
            "PUBLIC_POLICY_WORLD_SIZE",
            os.environ.get("PUBLIC_N_GPUS", "8"),
        )
    )
    if n_gpus < 1:
        raise ValueError("PUBLIC_POLICY_WORLD_SIZE must be positive")

    rewards = parse_reward_spec()
    method = name.rsplit("_", 1)[-1]
    config = _opsd_config(rewards, n_gpus) if method == "opsd" else _nft_config(rewards, n_gpus)

    # The paper uses 300 updates and the same behavior-policy coefficient for
    # its directly trained Open3 DiffusionOPSD and DiffusionNFT policies.
    config.beta = 0.1
    config.num_epochs = int(os.environ.get("PUBLIC_NUM_UPDATES", "300"))
    config.save_freq = int(os.environ.get("PUBLIC_SAVE_FREQ", "10"))
    config.eval_freq = 0
    config.run_name = f"mixed_{method}_{_slug(rewards)}"
    config.save_dir = str(ROOT / "outputs" / "mixed" / method / _slug(rewards))
    config.logdir = str(ROOT / "outputs" / "wandb")
    return config
