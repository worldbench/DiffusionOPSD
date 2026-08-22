"""Generic experiment I/O used by the public training entry points.

Schedules and output paths are supplied through public configs. This module
provides reusable checkpoint/resume and raw-record interfaces for trainers.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


_STEP_RE = re.compile(r"checkpoint-(\d+)$")


def resolve_resume_checkpoint(path: str) -> str:
    """Resolve an exact checkpoint or the newest checkpoint below a run directory."""
    if not path:
        return ""
    root = Path(path)
    if _STEP_RE.search(root.name):
        return str(root)
    candidates = []
    for parent in (root, root / "checkpoints", root / "save" / "checkpoints"):
        if not parent.is_dir():
            continue
        for child in parent.glob("checkpoint-*"):
            match = _STEP_RE.search(child.name)
            if match and child.is_dir():
                candidates.append((int(match.group(1)), child))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directory found under {path}")
    return str(max(candidates, key=lambda item: item[0])[1])


def updates_per_outer_epoch(config: Any, world_size: int) -> int:
    samples = int(config.sample.train_batch_size) * world_size * int(config.sample.num_batches_per_epoch)
    effective_batch = (
        int(config.train.batch_size) * world_size * int(config.train.gradient_accumulation_steps)
    )
    return max(1, samples // effective_batch) * int(config.train.num_inner_epochs)


def resume_position(checkpoint: str, config: Any, world_size: int) -> tuple[int, int]:
    """Return ``(first_epoch, global_step)`` for target-total-epoch semantics."""
    state_path = Path(checkpoint) / "trainer_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        return int(state["epoch_completed"]), int(state["global_step"])
    match = _STEP_RE.search(Path(checkpoint).name)
    if not match:
        raise ValueError(f"Cannot infer global step from checkpoint path: {checkpoint}")
    step = int(match.group(1))
    return step // updates_per_outer_epoch(config, world_size), step


def save_trainer_state(
    checkpoint: str,
    *,
    epoch_completed: int,
    global_step: int,
    ema: Any | None,
) -> None:
    """Save continuation metadata, EMA, and a safe rank-0 RNG snapshot."""
    root = Path(checkpoint)
    root.mkdir(parents=True, exist_ok=True)
    numpy_state = np.random.get_state()
    state = {
        "epoch_completed": int(epoch_completed),
        "global_step": int(global_step),
        "python_random_state": random.getstate(),
        # Store NumPy's uint32 array as a tensor so PyTorch's safe
        # ``weights_only=True`` loader can restore it without unpickling
        # arbitrary Python objects.
        "numpy_random_state": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    torch.save(state, root / "rng_state.pt")
    (root / "trainer_state.json").write_text(
        json.dumps({"epoch_completed": int(epoch_completed), "global_step": int(global_step)}, indent=2)
    )
    if ema is not None:
        torch.save(ema.state_dict(), root / "ema.pt")


def restore_ema_and_rng(checkpoint: str, ema: Any | None, *, restore_rng: bool = False) -> None:
    root = Path(checkpoint)
    ema_path = root / "ema.pt"
    if ema is not None and ema_path.is_file():
        ema.load_state_dict(torch.load(ema_path, map_location="cpu", weights_only=True))
    # A checkpoint is written by rank 0. Replaying that one RNG state on every
    # distributed rank would duplicate rollout noise, so restoration is opt-in.
    rng_path = root / "rng_state.pt"
    if not restore_rng:
        return
    if not rng_path.is_file():
        return
    state = torch.load(rng_path, map_location="cpu", weights_only=True)
    random.setstate(state["python_random_state"])
    numpy_state = state["numpy_random_state"]
    np.random.set_state((
        numpy_state["bit_generator"],
        numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=False),
        int(numpy_state["pos"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])


def write_raw_reward_jsonl(
    save_dir: str,
    *,
    epoch: int,
    global_step: int,
    prompts: Sequence[str],
    rewards: Mapping[str, np.ndarray],
) -> str:
    """Persist unnormalized per-rollout reward values for later curves/distributions."""
    out_dir = Path(save_dir) / "train_rewards"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"epoch_{epoch + 1:04d}.jsonl"
    arrays = {name: np.asarray(value) for name, value in rewards.items()}
    with out.open("w") as handle:
        for idx, prompt in enumerate(prompts):
            values = {}
            for name, array in arrays.items():
                item = np.asarray(array[idx]).reshape(-1)
                values[name] = float(item[0]) if item.size else None
            handle.write(json.dumps({
                "epoch": int(epoch + 1),
                "global_step_before_update": int(global_step),
                "sample_index": idx,
                "prompt": prompt,
                "rewards_raw": values,
            }, ensure_ascii=False) + "\n")
    return str(out)
