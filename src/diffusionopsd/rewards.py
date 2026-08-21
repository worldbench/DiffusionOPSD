"""Reward registry used by all public trainers and evaluators.

The registry intentionally contains exactly the seven public rewards reported
in the paper plus the three internal evaluators retained for provenance.
"""

from __future__ import annotations

import os

from PIL import Image
import torch


PUBLIC_REWARDS = {
    "hpsv2",
    "clipscore",
    "pickscore",
    "aesthetic",
    "imagereward",
    "hpsv3",
    "deqa",
}
INTERNAL_REWARDS = {"altclip", "internvl_t2i", "internvl_dual"}


def _as_tensor(images):
    if isinstance(images, torch.Tensor):
        return images
    return torch.tensor(images.transpose(0, 3, 1, 2), dtype=torch.uint8) / 255.0


def aesthetic_score(device):
    from diffusionopsd.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        del prompts, metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = torch.tensor(images.transpose(0, 3, 1, 2), dtype=torch.uint8)
        return scorer(images), {}

    return _fn


def clip_score(device):
    from diffusionopsd.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        del metadata
        return scorer(_as_tensor(images), prompts), {}

    return _fn


def hpsv2_score(device):
    from diffusionopsd.hpsv2_scorer import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        del metadata
        return scorer(_as_tensor(images), prompts), {}

    return _fn


def pickscore_score(device):
    from diffusionopsd.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            arrays = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = [Image.fromarray(image) for image in arrays.transpose(0, 2, 3, 1)]
        return scorer(prompts, images), {}

    return _fn


def imagereward_score(device):
    from diffusionopsd.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        del metadata
        return scorer(prompts, images), {}

    return _fn


def altclip_score(device):
    """Internal evaluator: requires the unreleased fine-tuned AltCLIP weights."""

    from diffusionopsd.altclip_scorer import AltCLIPScorer

    scorer = AltCLIPScorer(device=device, dtype=torch.float32)

    def _fn(images, prompts, metadata):
        del metadata
        return scorer(_as_tensor(images), prompts), {}

    return _fn


def internvl_t2i_score(device):
    """Internal 26B pointwise evaluator; checkpoint is not part of this release."""

    from diffusionopsd.internvl_bridge import bridge_enabled

    if bridge_enabled():
        from diffusionopsd.internvl_bridge import remote_reward_scores_forward

        def _fn(images, prompts, metadata):
            del metadata
            scores = remote_reward_scores_forward(_as_tensor(images).to(device).float(), prompts)
            return scores, {}

        return _fn

    from diffusionopsd.internvl_t2i_scorer import get_internvl_t2i_scorer

    scorer = get_internvl_t2i_scorer(device=device)

    def _fn(images, prompts, metadata):
        del metadata
        return scorer(_as_tensor(images), prompts), {}

    return _fn


def internvl_dual_score(device):
    """Internal pairwise evaluator P(generated > reference).

    References are read from ``INTERNVL_DUAL_REF_ROOT`` using the ``ref_path``
    entries in prompt metadata. Missing references fail closed. The optional
    gray image fallback exists only for explicitly enabled plumbing tests and
    must never be used for reported results.
    """

    import torchvision.transforms.functional as transforms

    from diffusionopsd.internvl_bridge import bridge_enabled

    bridged = bridge_enabled()
    if bridged:
        from diffusionopsd.internvl_bridge import remote_reward_scores_pair_forward

        scorer = None
    else:
        from diffusionopsd.internvl_dual_scorer import get_internvl_dual_scorer

        scorer = get_internvl_dual_scorer(device=device)

    ref_root = os.environ.get("INTERNVL_DUAL_REF_ROOT", "")
    allow_gray = os.environ.get("INTERNVL_DUAL_ALLOW_GRAY", "0") == "1"

    def _load_ref(ref_path: str, side: int):
        full = os.path.join(ref_root, ref_path) if ref_root and ref_path else ""
        if full and os.path.exists(full):
            return transforms.to_tensor(Image.open(full).convert("RGB"))
        if not allow_gray:
            raise RuntimeError(
                "Internal pairwise reference image is missing. Provide a prompt file with "
                "prompt<TAB>ref_path entries and set INTERNVL_DUAL_REF_ROOT. "
                "INTERNVL_DUAL_ALLOW_GRAY=1 is for plumbing tests only."
            )
        return torch.full((3, side, side), 0.5)

    def _fn(images, prompts, metadata):
        images = _as_tensor(images).float()
        side = int(images.shape[-1])
        scores = []
        for index, prompt in enumerate(prompts):
            item = metadata[index] if metadata and isinstance(metadata[index], dict) else {}
            ref = _load_ref(item.get("ref_path", ""), side).unsqueeze(0).to(images.device)
            if bridged:
                score = remote_reward_scores_pair_forward(
                    images[index:index + 1].to(device).float(), ref.to(device).float(), [prompt]
                )
            else:
                score = scorer(images[index:index + 1], ref, [prompt])
            scores.append(score)
        return torch.cat(scores).cpu(), {}

    return _fn


def hpsv3_score(device):
    from diffusionopsd.zimage_reward_bridge import zimage_bridge_enabled

    if zimage_bridge_enabled():
        from diffusionopsd.zimage_reward_bridge import bridge_reward

        def _fn(images, prompts, metadata):
            del metadata
            return bridge_reward(_as_tensor(images).to(device).float(), prompts), {}

        return _fn

    from diffusionopsd.zimage_heavy_diff_bridge import zimage_heavy_diff_bridge_enabled

    if zimage_heavy_diff_bridge_enabled():
        from diffusionopsd.zimage_heavy_diff_bridge import remote_heavy_reward_forward

        def _fn(images, prompts, metadata):
            del metadata
            return remote_heavy_reward_forward(_as_tensor(images).to(device).float(), prompts), {}

        return _fn

    from diffusionopsd.hpsv3_scorer import get_hpsv3_scorer

    scorer = get_hpsv3_scorer(device=device)

    def _fn(images, prompts, metadata):
        del metadata
        return scorer(_as_tensor(images), prompts), {}

    return _fn


def deqa_score(device):
    from diffusionopsd.zimage_reward_bridge import zimage_bridge_enabled

    if zimage_bridge_enabled():
        from diffusionopsd.zimage_reward_bridge import bridge_reward

        def _fn(images, prompts, metadata):
            del metadata
            return bridge_reward(_as_tensor(images).to(device).float(), prompts), {}

        return _fn

    from diffusionopsd.zimage_heavy_diff_bridge import zimage_heavy_diff_bridge_enabled

    if zimage_heavy_diff_bridge_enabled():
        from diffusionopsd.zimage_heavy_diff_bridge import remote_heavy_reward_forward

        def _fn(images, prompts, metadata):
            del metadata
            return remote_heavy_reward_forward(_as_tensor(images).to(device).float(), prompts), {}

        return _fn

    from diffusionopsd.deqa_scorer import get_deqa_scorer

    scorer = get_deqa_scorer(device=device)

    def _fn(images, prompts, metadata):
        del metadata
        return scorer(_as_tensor(images), prompts), {}

    return _fn


def multi_score(device, score_dict):
    """Create the shared weighted reward function used across all code paths."""

    factories = {
        "hpsv2": hpsv2_score,
        "clipscore": clip_score,
        "pickscore": pickscore_score,
        "aesthetic": aesthetic_score,
        "imagereward": imagereward_score,
        "hpsv3": hpsv3_score,
        "deqa": deqa_score,
        "altclip": altclip_score,
        "internvl_t2i": internvl_t2i_score,
        "internvl_dual": internvl_dual_score,
    }
    unknown = sorted(set(score_dict) - set(factories))
    if unknown:
        raise ValueError(f"Unsupported reward(s): {', '.join(unknown)}")
    if not score_dict:
        raise ValueError("At least one reward must be configured")

    scorers = {name: factories[name](device) for name in score_dict}

    def _fn(images, prompts, metadata, only_strict=True):
        del only_strict
        totals = None
        details = {}
        for name, weight in score_dict.items():
            scores, _ = scorers[name](images, prompts, metadata)
            details[name] = scores
            weighted = [weight * score for score in scores]
            totals = weighted if totals is None else [
                current + update for current, update in zip(totals, weighted)
            ]
        details["avg"] = totals
        return details, {}

    return _fn
