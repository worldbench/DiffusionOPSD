"""Shared, fail-closed utilities for the canonical ReFL trainers.

This module intentionally has no dependency on a diffusion backbone.  The pure
batch-planning helpers are unit-testable on CPU and make the experiment's two
important counting invariants explicit:

* exactly 48 distinct prompt groups are selected for every optimizer update;
* every group contributes exactly K fresh-noise trajectories (K=24/12).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import errno
import random
import shutil
import stat
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.distributed as dist

from diffusionopsd.stat_tracking import calculate_prompt_group_dispersion


REWARDS = (
    "pickscore",
    "clipscore",
    "hpsv2",
    "aesthetic",
    "imagereward",
    "hpsv3",
    "deqa",
)
ZIMAGE_DIFF_BRIDGE_REWARDS = {"hpsv3", "deqa"}
EVAL_SCORE_BATCH_ONE_REWARDS = {"hpsv3", "deqa"}
REFL_IMPLEMENTATION_REVISION = "truncated_prefix_pre_eval_ckpt_migrated_v3"
_TRANSIENT_IO_ERRNOS = {
    errno.EAGAIN,
    errno.EBUSY,
    errno.EDQUOT,
    errno.EIO,
    errno.ENOSPC,
    errno.ESTALE,
    errno.ETIMEDOUT,
}
_TRANSIENT_SPACE_ERRNOS = {errno.EDQUOT, errno.ENOSPC}


def policy_rank0(rank: int) -> bool:
    return rank == 0


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_prompt_line(line: str) -> dict[str, str]:
    line = line.rstrip("\n")
    if not line:
        raise ValueError("empty prompt in manifest")
    return {"prompt": line}


def load_prompt_records(path: str | os.PathLike[str], limit: int = 0) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        records = [parse_prompt_line(line) for line in handle if line.strip()]
    if limit > 0:
        records = records[:limit]
    return records


def derive_gradient_accumulation_steps(
    trajectories_per_update: int,
    policy_world_size: int,
    micro_batch_size: int,
) -> int:
    denominator = int(policy_world_size) * int(micro_batch_size)
    if denominator <= 0 or int(trajectories_per_update) % denominator:
        raise ValueError(
            "trajectories_per_update must divide exactly by policy_world_size * "
            f"micro_batch_size: {trajectories_per_update} / "
            f"({policy_world_size} * {micro_batch_size})"
        )
    return int(trajectories_per_update) // denominator


def _randperm(n: int, seed: int) -> list[int]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(n, generator=generator).tolist()


def build_update_records(
    dataset_records: Sequence[dict[str, str]],
    *,
    update_index: int,
    distinct_prompt_groups: int,
    trajectories_per_prompt: int,
    prompt_seed: int,
    shuffle_seed: int,
    noise_seed: int,
    late_seed: int,
) -> list[dict[str, Any]]:
    """Construct the global, deterministically shuffled trajectory plan for one update."""
    n = len(dataset_records)
    if n < distinct_prompt_groups:
        raise ValueError(f"dataset has {n} records but {distinct_prompt_groups} groups are required")
    chosen = _randperm(n, prompt_seed + update_index)[:distinct_prompt_groups]
    total = distinct_prompt_groups * trajectories_per_prompt
    records: list[dict[str, Any]] = []
    for group_slot, dataset_index in enumerate(chosen):
        source = dataset_records[dataset_index]
        for repeat_index in range(trajectories_per_prompt):
            canonical_index = group_slot * trajectories_per_prompt + repeat_index
            records.append({
                "group_slot": group_slot,
                "repeat_index": repeat_index,
                "dataset_index": dataset_index,
                "prompt": source["prompt"],
                "noise_seed": int(noise_seed + update_index * total + canonical_index),
                "late_seed": int(late_seed + update_index * total + canonical_index),
                "canonical_trajectory_index": canonical_index,
            })
    order = _randperm(total, shuffle_seed + update_index)
    shuffled = [records[i] for i in order]
    for global_position, record in enumerate(shuffled):
        record["global_position"] = global_position
    validate_global_update_records(
        shuffled,
        distinct_prompt_groups=distinct_prompt_groups,
        trajectories_per_prompt=trajectories_per_prompt,
    )
    return shuffled


def validate_global_update_records(
    records: Sequence[dict[str, Any]],
    *,
    distinct_prompt_groups: int,
    trajectories_per_prompt: int,
) -> None:
    expected = distinct_prompt_groups * trajectories_per_prompt
    if len(records) != expected:
        raise AssertionError(f"trajectory count {len(records)} != {expected}")
    counts = Counter(int(x["group_slot"]) for x in records)
    if set(counts) != set(range(distinct_prompt_groups)):
        raise AssertionError("prompt group slots are incomplete")
    if any(value != trajectories_per_prompt for value in counts.values()):
        raise AssertionError(f"per-prompt trajectory counts are not all {trajectories_per_prompt}: {counts}")
    noise_seeds = [int(x["noise_seed"]) for x in records]
    late_seeds = [int(x["late_seed"]) for x in records]
    if len(set(noise_seeds)) != expected or len(set(late_seeds)) != expected:
        raise AssertionError("trajectory noise/late seeds are not globally unique")


def shard_update_records(
    records: Sequence[dict[str, Any]], rank: int, world_size: int, micro_batch_size: int
) -> list[list[dict[str, Any]]]:
    if len(records) % world_size:
        raise ValueError(f"global trajectories {len(records)} not divisible by policy world {world_size}")
    per_rank = len(records) // world_size
    if per_rank % micro_batch_size:
        raise ValueError(f"per-rank trajectories {per_rank} not divisible by micro batch {micro_batch_size}")
    local = list(records[rank * per_rank : (rank + 1) * per_rank])
    return [local[i : i + micro_batch_size] for i in range(0, len(local), micro_batch_size)]


def choose_late_index(num_steps: int, late_fraction: float, seed: int) -> int:
    if num_steps <= 0 or not 0.0 < late_fraction <= 1.0:
        raise ValueError(f"invalid late-state request: num_steps={num_steps}, fraction={late_fraction}")
    late_count = max(1, math.ceil(num_steps * late_fraction))
    late_start = num_steps - late_count
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return late_start + int(torch.randint(late_count, (1,), generator=generator).item())


def fixed_manifest_records(
    records: Sequence[dict[str, str]],
    *,
    count: int,
    noise_seed: int,
    late_seed: int | None = None,
) -> list[dict[str, Any]]:
    if len(records) < count:
        raise ValueError(f"manifest needs {count} records but source has {len(records)}")
    result = []
    for index, source in enumerate(records[:count]):
        item: dict[str, Any] = {
            "index": index,
            "prompt": source["prompt"],
            "noise_seed": int(noise_seed + index),
        }
        if late_seed is not None:
            item["late_seed"] = int(late_seed + index)
        result.append(item)
    return result


def fixed_eval_manifest_records(
    records: Sequence[dict[str, str]],
    *,
    count: int,
    base_seed: int,
    generation_batch_size: int,
) -> list[dict[str, Any]]:
    """Freeze the paper evaluator's per-batch RNG scheme as per-image metadata.

    The existing SD evaluator samples batches of 16 with ``seed + batch_index``;
    the existing Z-Image evaluator does the same with batches of 8. ReFL
    reconstructs those exact generation batches, while allowing the large
    reward scorers to consume the resulting images one at a time.  Every record
    stores both the canonical batch seed and its offset within that batch.
    """
    if len(records) < count:
        raise ValueError(f"eval manifest needs {count} records but source has {len(records)}")
    if generation_batch_size <= 0:
        raise ValueError(f"invalid eval generation batch size: {generation_batch_size}")
    result = []
    for index, source in enumerate(records[:count]):
        batch_index, offset = divmod(index, generation_batch_size)
        batch_start = batch_index * generation_batch_size
        actual_batch_size = min(generation_batch_size, count - batch_start)
        result.append({
            "index": index,
            "prompt": source["prompt"],
            "noise_seed": int(base_seed + batch_index),
            "noise_offset": int(offset),
            "noise_batch_size": int(actual_batch_size),
            "canonical_generation_batch_size": int(generation_batch_size),
            "base_seed": int(base_seed),
            "seed_scheme": "canonical_per_batch_seed_plus_batch_index",
        })
    return result


def sync_mean(value: torch.Tensor, group=None) -> torch.Tensor:
    out = value.detach().float().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
    out /= dist.get_world_size(group=group)
    return out


def distributed_moments(local_values: Iterable[torch.Tensor] | torch.Tensor, group=None) -> dict[str, float]:
    if isinstance(local_values, torch.Tensor):
        values = local_values.detach().float().reshape(-1)
    else:
        pieces = [value.detach().float().reshape(-1) for value in local_values]
        if not pieces:
            raise ValueError("cannot summarize an empty local value list")
        values = torch.cat(pieces)
    packed = torch.stack([
        torch.tensor(float(values.numel()), device=values.device),
        values.sum(),
        values.square().sum(),
        values.min(),
        values.max(),
    ])
    # SUM the first three; MIN/MAX need their own reductions.
    count_sum_squares = packed[:3].clone()
    dist.all_reduce(count_sum_squares, op=dist.ReduceOp.SUM, group=group)
    minimum = packed[3].clone()
    maximum = packed[4].clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    count, total, total2 = count_sum_squares
    if int(count.item()) <= 0:
        raise ValueError("distributed summary received zero samples")
    mean = total / count
    if int(count.item()) > 1:
        variance = (total2 - count * mean.square()) / (count - 1)
    else:
        variance = total2 * 0
    std = variance.clamp_min(0).sqrt()
    return {
        "mean": float(mean),
        "std": float(std),
        "min": float(minimum),
        "max": float(maximum),
        "n": int(count.item()),
    }


def gather_indexed_scores(
    indices: torch.Tensor, scores: torch.Tensor, group=None, *, rank: int
) -> list[tuple[int, float]] | None:
    """Gather variable-length indexed score shards, returning sorted pairs on policy rank 0."""
    indices = indices.detach().to(dtype=torch.long).reshape(-1)
    scores = scores.detach().to(dtype=torch.float32).reshape(-1)
    if indices.numel() != scores.numel():
        raise ValueError("indices/scores length mismatch")
    world = dist.get_world_size(group=group)
    local_n = torch.tensor([indices.numel()], device=scores.device, dtype=torch.long)
    counts = [torch.zeros_like(local_n) for _ in range(world)]
    dist.all_gather(counts, local_n, group=group)
    max_n = max(int(x.item()) for x in counts)
    pad_i = torch.full((max_n,), -1, device=scores.device, dtype=torch.long)
    pad_s = torch.full((max_n,), float("nan"), device=scores.device, dtype=torch.float32)
    pad_i[: indices.numel()] = indices
    pad_s[: scores.numel()] = scores
    all_i = [torch.empty_like(pad_i) for _ in range(world)]
    all_s = [torch.empty_like(pad_s) for _ in range(world)]
    dist.all_gather(all_i, pad_i, group=group)
    dist.all_gather(all_s, pad_s, group=group)
    if rank != 0:
        return None
    merged: list[tuple[int, float]] = []
    for count, shard_i, shard_s in zip(counts, all_i, all_s):
        n = int(count.item())
        merged.extend((int(i), float(s)) for i, s in zip(shard_i[:n].cpu(), shard_s[:n].cpu()))
    merged.sort(key=lambda pair: pair[0])
    return merged


def distributed_prompt_group_dispersion(
    group_keys: torch.Tensor,
    rewards: torch.Tensor,
    group=None,
    *,
    rank: int,
) -> dict[str, float]:
    """Gather a ReFL update and apply the trainer-wide dispersion statistic."""
    indexed_rewards = gather_indexed_scores(group_keys, rewards, group, rank=rank)
    if rank != 0:
        return {}
    merged_group_keys, merged_rewards = zip(*indexed_rewards)
    return calculate_prompt_group_dispersion(merged_group_keys, merged_rewards)


def summarize_scores(scores: Sequence[float]) -> dict[str, float | int]:
    values = torch.tensor(list(scores), dtype=torch.float64)
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("score vector is empty or non-finite")
    std = values.std(unbiased=True) if values.numel() > 1 else values.new_zeros(())
    return {
        "mean": float(values.mean()),
        "std": float(std),
        "se": float(std / math.sqrt(values.numel())),
        "n": int(values.numel()),
    }


def _retry_io(label: str, operation, *, attempts: int | None = None):
    """Retry bounded transient shared-filesystem failures.

    ByteFS occasionally returns EIO/ESTALE, or transient EDQUOT/ENOSPC while
    concurrent checkpoint writes settle. A single failure must not kill a
    multi-hour distributed run. Non-transient failures still fail immediately
    and exhausted retries still fail closed.
    """
    attempts = int(os.environ.get("REFL_IO_RETRIES", "8")) if attempts is None else int(attempts)
    space_attempts = int(os.environ.get("REFL_SPACE_RETRIES", "60"))
    base_delay = float(os.environ.get("REFL_IO_RETRY_BASE_SECONDS", "0.5"))
    if attempts <= 0 or space_attempts <= 0:
        raise ValueError(
            f"invalid I/O retry count: attempts={attempts} space_attempts={space_attempts}"
        )
    for attempt in range(1, max(attempts, space_attempts) + 1):
        try:
            return operation()
        except OSError as error:
            retry_limit = space_attempts if error.errno in _TRANSIENT_SPACE_ERRNOS else attempts
            if error.errno not in _TRANSIENT_IO_ERRNOS or attempt >= retry_limit:
                raise
            delay = min(8.0, base_delay * (2 ** (attempt - 1)))
            print(
                f"REFL_IO_RETRY label={label!r} attempt={attempt}/{retry_limit} "
                f"errno={error.errno} delay={delay:.2f}s",
                flush=True,
            )
            time.sleep(delay)


def _read_bytes(path: str) -> bytes:
    def operation():
        with open(path, "rb") as handle:
            return handle.read()
    return _retry_io(f"read {path}", operation)


def _path_stat(path: str):
    try:
        return _retry_io(f"stat {path}", lambda: os.stat(path))
    except FileNotFoundError:
        return None


def _path_exists(path: str) -> bool:
    return _path_stat(path) is not None


def _is_file(path: str) -> bool:
    stat_result = _path_stat(path)
    return stat_result is not None and stat.S_ISREG(stat_result.st_mode)


def _atomic_write_bytes(path: str, payload: bytes) -> None:
    directory = os.path.dirname(path)

    def operation():
        os.makedirs(directory, exist_ok=True)
        partial = os.path.join(
            directory,
            f".{os.path.basename(path)}.{os.getpid()}.{time.time_ns()}.partial",
        )
        try:
            with open(partial, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, path)
        finally:
            try:
                if os.path.exists(partial):
                    os.unlink(partial)
            except OSError:
                pass

    _retry_io(f"atomic write {path}", operation)


def append_jsonl(path: str, record: dict[str, Any], *, rank: int) -> None:
    """Durably append one JSON record without exposing a partial final file.

    There is only one writer (policy rank 0), so rewriting the small manifest
    or metrics file through an atomic rename is both simple and idempotent.
    If a rename succeeded but its acknowledgement was lost, the tail equality
    check prevents a duplicate record on retry.
    """
    if not policy_rank0(rank):
        return
    line = (json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")

    def operation():
        try:
            existing = _read_bytes(path)
        except FileNotFoundError:
            existing = b""
        if existing and not existing.endswith(b"\n"):
            raise RuntimeError(f"refusing to append to truncated JSONL: {path}")
        if existing and existing.rsplit(b"\n", 2)[-2] + b"\n" == line:
            return
        _atomic_write_bytes(path, existing + line)

    _retry_io(f"append JSONL {path}", operation)


def rewrite_jsonl(path: str, records: Sequence[dict[str, Any]], *, rank: int) -> None:
    if not policy_rank0(rank):
        return
    payload = b"".join(
        (json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        for record in records
    )
    _atomic_write_bytes(path, payload)


def read_json(path: str) -> Any:
    return json.loads(_read_bytes(path).decode("utf-8"))


def write_json(path: str, payload: Any, *, rank: int) -> None:
    if not policy_rank0(rank):
        return
    data = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    _atomic_write_bytes(path, data)


def prepare_resume_artifacts(
    save_dir: str,
    global_step: int,
    *,
    rank: int,
    replay_online_eval_step: int | None = None,
) -> tuple[float, float, bool]:
    """Roll append-only artifacts back to the durable checkpoint boundary.

    Metrics/manifests may be newer than the latest checkpoint when a node dies.
    Recomputing those updates without truncation would create duplicate steps.
    A checkpoint is deliberately committed before its same-step online eval.
    When resuming an eval boundary, remove any prior same-step eval metric and
    replay that eval before the next optimizer update.  Replaying even a
    previously completed eval restores the exact post-eval RNG path rather than
    silently continuing from the checkpoint's pre-eval RNG state.

    Returns retained cumulative ``(training_seconds, online_eval_seconds,
    replay_online_eval)``.
    """
    if not policy_rank0(rank):
        return 0.0, 0.0, False
    metrics_path = os.path.join(save_dir, "metrics.jsonl")
    retained_metrics: list[dict[str, Any]] = []
    if _is_file(metrics_path):
        for line in _read_bytes(metrics_path).decode("utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            step = record.get("step")
            if isinstance(step, int) and step <= global_step:
                retained_metrics.append(record)
    replay_online_eval = replay_online_eval_step is not None
    if replay_online_eval:
        if int(replay_online_eval_step) != int(global_step):
            raise ValueError(
                "replayed online eval must be the durable checkpoint boundary: "
                f"{replay_online_eval_step} != {global_step}"
            )
        retained_metrics = [
            record
            for record in retained_metrics
            if not (
                record.get("kind") == "online_eval"
                and int(record.get("step", -1)) == int(replay_online_eval_step)
            )
        ]
    if _is_file(metrics_path):
        rewrite_jsonl(metrics_path, retained_metrics, rank=rank)
    manifest_path = os.path.join(save_dir, "manifests", "train_updates.jsonl")
    retained_manifests: list[dict[str, Any]] = []
    if _is_file(manifest_path):
        for line in _read_bytes(manifest_path).decode("utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("update", -1)) <= global_step:
                retained_manifests.append(record)
        rewrite_jsonl(manifest_path, retained_manifests, rank=rank)
    train_steps = [record for record in retained_metrics if record.get("kind") == "train"]
    actual_steps = [int(record.get("step", -1)) for record in train_steps]
    if actual_steps != list(range(1, global_step + 1)):
        raise RuntimeError(
            f"resume metrics do not cover checkpoint steps 1..{global_step}: {actual_steps[-8:]}"
        )
    eval_steps = [record for record in retained_metrics if record.get("kind") == "online_eval"]
    return (
        float(sum(float(record.get("step_seconds", 0.0)) for record in train_steps)),
        float(sum(float(record.get("eval_seconds", 0.0)) for record in eval_steps)),
        replay_online_eval,
    )


def save_refl_checkpoint(
    save_dir: str,
    *,
    global_step: int,
    rank: int,
    model,
    trainable_parameters,
    ema,
    config,
    optimizer,
    scaler,
    provenance: dict[str, str],
    group=None,
) -> str:
    """Save a complete resumable checkpoint through a staging directory.

    ``lora`` remains the EMA adapter used for evaluation compatibility;
    ``resume_lora`` stores the raw optimizer policy.  The final checkpoint
    directory only appears after every file and the completion marker exist.
    """
    final = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    marker = os.path.join(final, "checkpoint_complete.json")
    import numpy as np
    local_rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(torch.cuda.current_device()),
    }
    rng_states = [None] * dist.get_world_size(group=group) if policy_rank0(rank) else None
    dist.gather_object(local_rng_state, rng_states, dst=0, group=group)
    if not policy_rank0(rank):
        return final
    if _is_file(marker):
        recorded = read_json(marker)
        if int(recorded.get("global_step", -1)) != int(global_step):
            raise RuntimeError(f"checkpoint marker step mismatch: {marker}")
        return final

    checkpoints_root = os.path.dirname(final)

    def operation():
        if _is_file(marker):
            recorded = read_json(marker)
            if int(recorded.get("global_step", -1)) != int(global_step):
                raise RuntimeError(f"checkpoint marker step mismatch: {marker}")
            return
        os.makedirs(checkpoints_root, exist_ok=True)
        staging = os.path.join(
            checkpoints_root,
            f".checkpoint-{global_step}.{os.getpid()}.{time.time_ns()}.partial",
        )
        try:
            os.makedirs(staging, exist_ok=False)
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(os.path.join(staging, "resume_lora"))
            ema.copy_ema_to(trainable_parameters, store_temp=True)
            try:
                model_to_save.save_pretrained(os.path.join(staging, "lora"))
            finally:
                ema.copy_temp_to(trainable_parameters)
            torch.save(optimizer.state_dict(), os.path.join(staging, "optimizer.pt"))
            if scaler is not None:
                torch.save(scaler.state_dict(), os.path.join(staging, "scaler.pt"))
            torch.save(ema.state_dict(), os.path.join(staging, "ema.pt"))
            torch.save(rng_states, os.path.join(staging, "rng_states.pt"))
            write_json(
                os.path.join(staging, "trainer_state.json"),
                {"epoch_completed": int(global_step), "global_step": int(global_step)},
                rank=rank,
            )
            write_json(
                os.path.join(staging, "checkpoint_complete.json"),
                {
                    "global_step": int(global_step),
                    "implementation_revision": REFL_IMPLEMENTATION_REVISION,
                    "checkpoint_before_online_eval": True,
                    "completed_unix_ns": time.time_ns(),
                    "provenance": provenance,
                },
                rank=rank,
            )
            if _path_exists(final):
                raise FileExistsError(f"incomplete/conflicting checkpoint already exists: {final}")
            os.replace(staging, final)
        finally:
            try:
                if os.path.isdir(staging):
                    shutil.rmtree(staging)
            except OSError:
                pass

    _retry_io(f"save checkpoint {global_step}", operation)
    write_json(
        os.path.join(save_dir, "latest_checkpoint.json"),
        {"global_step": int(global_step), "path": final},
        rank=rank,
    )
    return final


def load_refl_checkpoint(
    checkpoint: str,
    *,
    model,
    ema,
    optimizer,
    scaler,
    device: torch.device,
    expected_provenance: dict[str, str],
    group=None,
) -> int:
    marker_path = os.path.join(checkpoint, "checkpoint_complete.json")
    marker = read_json(marker_path)
    if marker.get("implementation_revision") != REFL_IMPLEMENTATION_REVISION:
        raise RuntimeError(f"checkpoint implementation revision mismatch: {marker_path}")
    recorded_provenance = marker.get("provenance", {})
    for key in ("refl_trainer_sha256", "refl_config_sha256", "refl_matrix_sha256"):
        if recorded_provenance.get(key) != expected_provenance.get(key):
            raise RuntimeError(f"checkpoint provenance mismatch for {key}")
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    raw_lora = os.path.join(checkpoint, "resume_lora")
    state = _retry_io(
        f"load LoRA {raw_lora}",
        lambda: load_peft_weights(raw_lora, device=str(device)),
    )
    model_to_load = model.module if hasattr(model, "module") else model
    set_peft_model_state_dict(model_to_load, state, adapter_name="default")
    optimizer_state = _retry_io(
        f"load optimizer {checkpoint}",
        lambda: torch.load(os.path.join(checkpoint, "optimizer.pt"), map_location=device),
    )
    optimizer.load_state_dict(optimizer_state)
    scaler_path = os.path.join(checkpoint, "scaler.pt")
    if scaler is not None and _is_file(scaler_path):
        scaler_state = _retry_io(
            f"load scaler {checkpoint}",
            lambda: torch.load(scaler_path, map_location=device),
        )
        scaler.load_state_dict(scaler_state)
    ema_state = _retry_io(
        f"load EMA {checkpoint}",
        lambda: torch.load(os.path.join(checkpoint, "ema.pt"), map_location="cpu"),
    )
    ema.load_state_dict(ema_state)
    return int(marker["global_step"])


def restore_refl_rng_state(checkpoint: str, *, device: torch.device, group=None) -> None:
    """Restore per-policy-rank RNG after all resume-time initialization.

    Model/scorer construction and prompt pre-encoding can consume process RNG.
    Restoring only while loading model weights would therefore not reproduce the
    checkpoint boundary.  Trainers call this immediately before artifact repair,
    boundary-eval replay, or the next optimizer update.
    """
    rng_states = _retry_io(
        f"load RNG states {checkpoint}",
        lambda: torch.load(
            os.path.join(checkpoint, "rng_states.pt"), map_location="cpu", weights_only=False
        ),
    )
    policy_rank = dist.get_rank(group=group)
    if len(rng_states) != dist.get_world_size(group=group):
        raise RuntimeError("checkpoint RNG state count does not match policy world size")
    rng_state = rng_states[policy_rank]
    import numpy as np
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"])
    torch.cuda.set_rng_state(rng_state["torch_cuda"], device=device)


def maybe_no_sync(model, should_sync: bool):
    return nullcontext() if should_sync else model.no_sync()


def require_finite_nonzero(name: str, value: float, *, minimum: float = 0.0) -> None:
    if not math.isfinite(float(value)) or float(value) <= minimum:
        raise RuntimeError(f"{name} must be finite and > {minimum}, got {value}")


def provenance_from_env() -> dict[str, str]:
    keys = (
        "REFL_CODE_COMMIT",
        "REFL_TRAINER_SHA256",
        "REFL_CONFIG_SHA256",
        "REFL_MATRIX_SHA256",
        "REFL_LAUNCHER_SHA256",
        "CUDA_VISIBLE_DEVICES",
    )
    return {key.lower(): os.environ.get(key, "") for key in keys}


def assert_output_under_allowed_remote_root(path: str) -> None:
    """Optionally enforce deployment-specific shared-storage roots.

    Public users may write to a local filesystem. Cluster users can provide a
    colon-separated ``DOPSD_ALLOWED_OUTPUT_ROOTS`` list and enable the gate with
    ``DOPSD_ENFORCE_SHARED_OUTPUT=1``.
    """
    if os.environ.get("DOPSD_ENFORCE_SHARED_OUTPUT", "0") != "1":
        return
    raw_roots = os.environ.get("DOPSD_ALLOWED_OUTPUT_ROOTS", "")
    roots = [Path(root).expanduser().resolve() for root in raw_roots.split(os.pathsep) if root]
    if not roots:
        raise RuntimeError(
            "DOPSD_ENFORCE_SHARED_OUTPUT=1 requires DOPSD_ALLOWED_OUTPUT_ROOTS"
        )
    resolved = Path(path).expanduser().resolve()
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise RuntimeError(f"output {resolved} is outside allowed roots: {roots}")
