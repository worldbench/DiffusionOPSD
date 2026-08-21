"""Efficiency-profiling harness used by the paper's cost analysis.

Most DiffusionOPSD trainers make one optimizer update per outer rollout round;
faithful FlowGRPO makes two so its second replay window sees an updated policy.
The profiler derives this count from batch/accumulation settings and converts
each end-to-end round time to wall-clock per optimizer step.

PROFILE mode (env-gated) instruments the epoch/optimizer-step loop:

* ``PROFILE=1``            enable profiling.
* ``PROFILE_WARMUP=W``     warm-up optimizer steps to discard (default 3).
* ``PROFILE_OPT_STEPS=N``  measured optimizer steps (default 12).
* ``PROFILE_OUT_DIR``      output directory for ``profile.json`` / ``metrics.jsonl``.
* ``PROFILE_METHOD``       method label (nft|refl|opsd), else "unknown".
* ``PROFILE_BACKBONE``     backbone label (default "sd35m").
* ``PROFILE_SANITY_PROMPTS`` DrawBench sanity-eval prompt count (default 16, rank-0 only).

After ``W`` warm-up + ``N`` measured optimizer steps the profiler writes
``metrics.jsonl`` (one record / measured step) and ``profile.json`` (aggregates
+ sanity ClipScore), then the trainer breaks out of the epoch loop and shuts
down cleanly (no 100-epoch training, no checkpoint spam — the trainer also
forces ``config.debug=True`` so periodic/final ``save_ckpt`` and ``eval_fn`` are
skipped).

Counters are reset at the start of every outer round (``epoch_begin``) and read
at the end (``epoch_end``); per-step values are divided by the derived update
count. Only post-warm-up rounds are recorded.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist


# --------------------------------------------------------------------------- #
# Global counters (reset per optimizer step; incremented at reward/backward
# call sites in the trainers).  Guarded by a lock because rollout reward scoring
# runs inside a ThreadPoolExecutor.
# --------------------------------------------------------------------------- #
class _State:
    def __init__(self):
        self.enabled = False
        self.lock = threading.Lock()
        self.reward_fwd = 0
        self.reward_bwd = 0
        self.train_bwd = 0


_S = _State()


def profile_enabled() -> bool:
    """True iff the launcher requested PROFILE mode."""
    return os.environ.get("PROFILE", "0") == "1"


def enable() -> None:
    _S.enabled = True


def is_enabled() -> bool:
    return _S.enabled


def reward_fwd_inc(n: int = 1) -> None:
    """Count one reward-model forward pass (rollout scoring / OPA ascent / ReFL)."""
    if not _S.enabled:
        return
    with _S.lock:
        _S.reward_fwd += n


def reward_bwd_inc(n: int = 1) -> None:
    """Count one reward-gradient backward (OPSD trust-region ascent / RI refine)."""
    if not _S.enabled:
        return
    with _S.lock:
        _S.reward_bwd += n


def train_bwd_inc(n: int = 1) -> None:
    """Count one diffusion-model (training-loss) backward pass."""
    if not _S.enabled:
        return
    with _S.lock:
        _S.train_bwd += n


def _reset_counts() -> None:
    with _S.lock:
        _S.reward_fwd = 0
        _S.reward_bwd = 0
        _S.train_bwd = 0


def _snapshot_counts():
    with _S.lock:
        return _S.reward_fwd, _S.reward_bwd, _S.train_bwd


def count_reward_fn(fn):
    """Wrap a ``multi_score`` reward callable so each invocation counts as one
    reward forward pass.  Signature-transparent."""

    def _wrapped(*args, **kwargs):
        reward_fwd_inc(1)
        return fn(*args, **kwargs)

    return _wrapped


# --------------------------------------------------------------------------- #
def _dist_ready(world_size: int) -> bool:
    return world_size > 1 and dist.is_available() and dist.is_initialized()


def _percentiles(xs):
    if not xs:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    s = sorted(xs)
    n = len(s)
    mean = sum(s) / n
    median = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    var = sum((x - mean) ** 2 for x in s) / n
    return {"mean": mean, "median": median, "std": math.sqrt(var), "min": s[0], "max": s[-1]}


class Profiler:
    """Per-run efficiency profiler; one instance per trainer process.

    ``epoch_begin``/``epoch_end`` bracket one end-to-end rollout/replay round.
    ``opt_steps_per_epoch`` is derived from the config and converts outer-round
    measurements to per-optimizer-step values (a no-op when it equals 1).
    """

    def __init__(self, config, world_size: int, rank: int, device):
        self.config = config
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.device = device

        self.warmup = int(os.environ.get("PROFILE_WARMUP", "3"))
        self.measure = int(os.environ.get("PROFILE_OPT_STEPS", "12"))
        self.method = os.environ.get("PROFILE_METHOD", "unknown")
        self.backbone = os.environ.get("PROFILE_BACKBONE", "sd35m")
        self.sanity_prompts = int(os.environ.get("PROFILE_SANITY_PROMPTS", "16"))

        self.reward = list(config.reward_fn.keys())[0] if len(config.reward_fn) else "unknown"

        # ReFL defines its complete trajectory budget explicitly and performs one
        # optimizer update per outer loop. Other trainers derive the update count
        # from their epoch/micro-batch configuration.
        if hasattr(config, "refl"):
            self.opt_per_epoch = 1
        else:
            num_batches = (
                config.sample.num_batches_per_epoch * config.sample.train_batch_size
                // config.train.batch_size
            )
            gas = max(1, int(config.train.gradient_accumulation_steps))
            self.opt_per_epoch = max(1, int(config.train.num_inner_epochs) * num_batches // gas)

        # Warm-up / measured EPOCHS (converted from optimizer-step budgets).
        self.warmup_epochs = max(1, math.ceil(self.warmup / self.opt_per_epoch))
        self.measure_epochs = max(1, math.ceil(self.measure / self.opt_per_epoch))

        # Rollout images produced per epoch, across all GPUs (48 groups x 24 img
        # for the §6 protocol = 1152).  Throughput is reported on these.
        if hasattr(config, "refl"):
            self.rollout_images = int(config.refl.trajectories_per_update)
        else:
            self.rollout_images = int(
                config.sample.train_batch_size * self.world_size * config.sample.num_batches_per_epoch
            )

        self.records = []
        self._t0 = None
        self._measured = False
        self.finalized = False

        self.out_dir = os.environ.get("PROFILE_OUT_DIR", os.path.join(config.save_dir, "profile"))
        self.metrics_path = os.path.join(self.out_dir, "metrics.jsonl")
        self.profile_path = os.path.join(self.out_dir, "profile.json")
        if self.rank == 0:
            os.makedirs(self.out_dir, exist_ok=True)
            # Truncate any stale metrics from a previous run.
            open(self.metrics_path, "w").close()

    # -- window bookkeeping ------------------------------------------------- #
    @property
    def total_epochs(self) -> int:
        return self.warmup_epochs + self.measure_epochs

    def done(self, epoch_local_index: int) -> bool:
        """True once warm-up + measured epochs have completed.

        ``epoch_local_index`` counts epochs since profiling started (0-based).
        """
        return (epoch_local_index + 1) >= self.total_epochs

    # -- per optimizer-step (epoch) hooks ----------------------------------- #
    def epoch_begin(self, epoch_local_index: int) -> None:
        _reset_counts()
        self._measured = epoch_local_index >= self.warmup_epochs
        if self._measured:
            torch.cuda.reset_peak_memory_stats(self.device)
        if _dist_ready(self.world_size):
            dist.barrier()
        torch.cuda.synchronize(self.device)
        self._t0 = time.perf_counter()

    def epoch_end(self, epoch_local_index: int, global_step: int) -> None:
        torch.cuda.synchronize(self.device)
        if _dist_ready(self.world_size):
            dist.barrier()
        dt = time.perf_counter() - self._t0
        if not self._measured:
            return

        peak_alloc = float(torch.cuda.max_memory_allocated(self.device))
        peak_resv = float(torch.cuda.max_memory_reserved(self.device))
        # Reduce (MAX) time + peak-mem across ranks so the report reflects the
        # slowest / hungriest GPU.  Counts are symmetric across ranks (all ranks
        # do identical work), so the local snapshot is used directly.
        if _dist_ready(self.world_size):
            t = torch.tensor([dt, peak_alloc, peak_resv], device=self.device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            dt, peak_alloc, peak_resv = (float(x) for x in t.tolist())

        r_fwd, r_bwd, t_bwd = _snapshot_counts()
        opp = self.opt_per_epoch
        time_per_step = dt / opp
        rec = {
            "measured_index": len(self.records),
            "epoch_local_index": epoch_local_index,
            "global_step": int(global_step),
            "opt_steps_per_epoch": opp,
            "epoch_time_s": dt,
            "time_per_opt_step_s": time_per_step,
            "peak_mem_allocated_gb": peak_alloc / (1024 ** 3),
            "peak_mem_reserved_gb": peak_resv / (1024 ** 3),
            "reward_fwd_per_step": r_fwd / opp,
            "reward_bwd_per_step": r_bwd / opp,
            "train_bwd_per_step": t_bwd / opp,
            "reward_fwd_per_epoch": r_fwd,
            "reward_bwd_per_epoch": r_bwd,
            "train_bwd_per_epoch": t_bwd,
            "rollout_images": self.rollout_images,
            "images_per_s": self.rollout_images / dt if dt > 0 else None,
            "gpu_hours_per_100it": time_per_step * 100.0 * self.world_size / 3600.0,
        }
        self.records.append(rec)
        if self.rank == 0:
            with open(self.metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    # -- finalization ------------------------------------------------------- #
    def finalize(self, sanity_fn=None) -> None:
        if self.finalized:
            return
        self.finalized = True

        sanity = {}
        if sanity_fn is not None:
            try:
                sanity = sanity_fn() or {}
            except Exception as exc:  # never let a sanity-eval failure lose the profile
                sanity = {"error": repr(exc)}

        if self.rank != 0:
            return

        def col(key):
            return [r[key] for r in self.records]

        summary = {
            "method": self.method,
            "backbone": self.backbone,
            "reward": self.reward,
            "world_size": self.world_size,
            "warmup_opt_steps": self.warmup,
            "measured_opt_steps": self.measure,
            "warmup_epochs": self.warmup_epochs,
            "measured_epochs": self.measure_epochs,
            "opt_steps_per_epoch": self.opt_per_epoch,
            "note": "Outer-round measurements are divided by config-derived optimizer updates per round.",
            "protocol": {
                "resolution": int(self.config.resolution),
                "guidance_scale": float(self.config.sample.guidance_scale),
                "rollout_solver": str(self.config.sample.solver),
                "rollout_num_steps": int(self.config.sample.num_steps),
                "eval_solver": "flow",
                "eval_num_steps": int(self.config.sample.eval_num_steps),
                "num_image_per_prompt": int(self.config.sample.num_image_per_prompt),
                "num_batches_per_epoch": int(self.config.sample.num_batches_per_epoch),
                "train_batch_size": int(self.config.sample.train_batch_size),
                "gradient_accumulation_steps": int(self.config.train.gradient_accumulation_steps),
                "mixed_precision": str(self.config.mixed_precision),
                "rollout_images_per_epoch": self.rollout_images,
            },
            "time_per_opt_step_s": _percentiles(col("time_per_opt_step_s")),
            "peak_gpu_mem_gb": {
                "max_allocated": max(col("peak_mem_allocated_gb")) if self.records else None,
                "max_reserved": max(col("peak_mem_reserved_gb")) if self.records else None,
                "mean_allocated": (sum(col("peak_mem_allocated_gb")) / len(self.records)) if self.records else None,
            },
            "reward_forward_per_step": _percentiles(col("reward_fwd_per_step"))["mean"],
            "reward_backward_per_step": _percentiles(col("reward_bwd_per_step"))["mean"],
            "training_backward_per_step": _percentiles(col("train_bwd_per_step"))["mean"],
            "images_per_second": _percentiles([r["images_per_s"] for r in self.records if r["images_per_s"]])["mean"],
            "gpu_hours_per_100it": _percentiles(col("gpu_hours_per_100it"))["mean"],
            "sanity_eval": sanity,
            "num_measured_records": len(self.records),
            "metrics_file": self.metrics_path,
        }
        with open(self.profile_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[profile] wrote {self.profile_path} ({len(self.records)} measured steps)", flush=True)


# --------------------------------------------------------------------------- #
# Sanity eval: raw reward (ClipScore) of the post-profiling policy on a fixed
# DrawBench subset.  Rank-0 only (uses pipeline.transformer forward directly, no
# DDP collectives), so it never deadlocks the other ranks (which proceed to the
# trainer's post-loop dist.barrier).  Best-effort: any failure is caught by the
# caller / Profiler.finalize.
# --------------------------------------------------------------------------- #
def run_sanity_eval(
    *,
    pipeline,
    reward_fn,
    compute_text_embeddings,
    text_encoders,
    tokenizers,
    config,
    device,
    rank: int,
    world_size: int,
    num_prompts: int | None = None,
):
    if rank != 0:
        return {}

    from diffusionopsd.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob

    n_want = int(num_prompts if num_prompts is not None else os.environ.get("PROFILE_SANITY_PROMPTS", "16"))
    reward_key = list(config.reward_fn.keys())[0] if len(config.reward_fn) else "avg"

    # Locate a DrawBench prompt subset (sibling of the train dataset dir); fall
    # back to the config dataset's own test split.
    ds = Path(config.dataset)
    cand = ds.parent / "drawbench" / "test.txt"
    if not cand.exists():
        cand = ds / "test.txt"
    with open(cand, "r") as f:
        prompts = [ln.strip() for ln in f if ln.strip()][:n_want]

    mp = str(config.mixed_precision)
    mp_dtype = torch.float16 if mp == "fp16" else (torch.bfloat16 if mp == "bf16" else None)
    enable_amp = mp_dtype is not None

    bs = int(config.sample.test_batch_size)
    pipeline.transformer.eval()
    neg_e, neg_pe = compute_text_embeddings([""], text_encoders, tokenizers, 128, device)

    scores_all = []
    used_key = reward_key
    for start in range(0, len(prompts), bs):
        chunk = prompts[start:start + bs]
        n = len(chunk)
        pe, ppe = compute_text_embeddings(chunk, text_encoders, tokenizers, 128, device)
        with torch.cuda.amp.autocast(enabled=enable_amp, dtype=mp_dtype):
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=pe,
                    pooled_prompt_embeds=ppe,
                    negative_prompt_embeds=neg_e.repeat(n, 1, 1),
                    negative_pooled_prompt_embeds=neg_pe.repeat(n, 1),
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=config.sample.noise_level,
                    deterministic=True,
                    solver="flow",
                    model_type="sd3",
                )
        details, _ = reward_fn(images, chunk, [{} for _ in range(n)], only_strict=False)
        used_key = reward_key if reward_key in details else "avg"
        scores_all.extend(float(x) for x in details[used_key])

    finite = [x for x in scores_all if math.isfinite(x)]
    mean = (sum(finite) / len(finite)) if finite else None
    return {
        "reward_key": used_key,
        "mean": mean,
        "num_prompts": len(scores_all),
        "prompt_source": str(cand),
        "solver": "flow",
        "num_steps": int(config.sample.eval_num_steps),
        "guidance_scale": float(config.sample.guidance_scale),
        "ema": False,
    }
