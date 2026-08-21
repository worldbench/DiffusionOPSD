#!/usr/bin/env python3
"""Load one public evaluator and verify a finite image-space reward gradient.

This is intentionally a one-image smoke test.  It exercises the exact scorer
loader and differentiable forward used to construct DiffusionOPSD targets,
without loading either diffusion backbone.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import train_opsd_ri_sd3 as trainer


PUBLIC_REWARDS = (
    "hpsv2",
    "clipscore",
    "pickscore",
    "aesthetic",
    "imagereward",
    "hpsv3",
    "deqa",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward", choices=PUBLIC_REWARDS, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--side", type=int, default=224)
    parser.add_argument("--prompt", default="a red cube on a blue table")
    args = parser.parse_args()

    if args.side < 32:
        raise ValueError("--side must be at least 32")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    started = time.perf_counter()
    scorer = trainer._load_reward_scorer(args.reward, args.device)
    image = torch.linspace(
        0.05,
        0.95,
        steps=3 * args.side * args.side,
        device=args.device,
        dtype=torch.float32,
    ).reshape(1, 3, args.side, args.side)
    image.requires_grad_(True)
    score = trainer._reward_scores_grad(scorer, args.reward, image, [args.prompt])
    (gradient,) = torch.autograd.grad(score.sum(), image)

    score_value = float(score.detach().float().mean().cpu())
    grad_norm = float(gradient.detach().float().norm().cpu())
    if not math.isfinite(score_value):
        raise RuntimeError(f"non-finite {args.reward} score: {score_value}")
    if not math.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError(f"invalid {args.reward} image-gradient norm: {grad_norm}")

    peak_gib = None
    if args.device.startswith("cuda"):
        peak_gib = torch.cuda.max_memory_allocated(args.device) / (1024 ** 3)
    print(
        json.dumps(
            {
                "reward": args.reward,
                "score": score_value,
                "image_grad_norm": grad_norm,
                "peak_cuda_gib": peak_gib,
                "elapsed_seconds": time.perf_counter() - started,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
