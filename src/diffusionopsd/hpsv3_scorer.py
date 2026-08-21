"""Differentiable HPSv3 reward scorer (MizzenAI/HPSv3 = Qwen2-VL-7B + ranknet head).

HPSv3 is the VLM successor of HPSv2.1: pointwise, rewards prompt-alignment AND quality, scalar = mu
(the first of the [mu, sigma] ranknet logits). The `hpsv3` package ships its OWN differentiable image
processor (`_preprocess_differentiable`: smart_resize + F.interpolate + patchify, all torch ops), so the
reward is differentiable w.r.t. the input image and can drive OPA's reward-gradient ascent.

Design:
- Load `HPSv3RewardInferencer(differentiable=True)` (Qwen2-VL-7B + ranknet, output_dim=2, use_special_tokens).
- Image resolution is FIXED by the model (max_pixels=min_pixels=256*28*28 -> 448x448 -> grid [1,32,32], 256
  image tokens) regardless of content, so the tokenized text (input_ids/attention_mask/image_grid_thw) is
  identical for a given prompt set. We therefore build the text batch via the official `prepare_batch` with a
  dummy PIL image (correct tokenization), then OVERRIDE `pixel_values` with the differentiable per-image
  `_preprocess(img01, do_rescale=False)` output (same shape). Text correctness = official; image grad = ours.
- reward = model(**batch)["logits"][:, 0]  (mu).
"""

import os
from typing import Sequence

import torch
from PIL import Image

DEFAULT_CHECKPOINT = os.environ.get("HPSV3_CHECKPOINT") or None
DEFAULT_REVISION = os.environ.get(
    "HPSV3_REVISION", "4f81e3e09edd82fe3c5f636444c721b592a735ca"
)
_FIXED_HW = 448  # 256*28*28 pixels, square -> smart_resize gives 448x448


class HPSv3Scorer(torch.nn.Module):
    def __init__(self, checkpoint_path: str | None = DEFAULT_CHECKPOINT, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        from hpsv3 import HPSv3RewardInferencer
        self.device = device
        self.dtype = dtype
        if checkpoint_path is None:
            from huggingface_hub import hf_hub_download

            checkpoint_path = hf_hub_download(
                "MizzenAI/HPSv3",
                "HPSv3.safetensors",
                revision=DEFAULT_REVISION,
            )
        self.inf = HPSv3RewardInferencer(config_path=None, checkpoint_path=checkpoint_path,
                                         device=device, differentiable=True)
        self.inf.model.requires_grad_(False)
        # --- MEMORY: gradient-checkpoint the 7B backbone. The OPA reward-ascent needs grad w.r.t. the
        # input IMAGE through the frozen 7B, so without checkpointing the full 7B forward activations are
        # retained for the backward -> CUDA OOM next to SD3 (opa_mb=1 alone is insufficient for 7B).
        # Params are frozen and this model has attention_dropout=0 + no BN, so train() (needed to activate
        # HF's checkpointing gate) is numerically equivalent to eval() for the reward. ---
        try:
            self.inf.model.config.use_cache = False
            self.inf.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.inf.model.train()  # activate ckpt gate; params frozen + dropout=0 -> reward unchanged
        except Exception:
            self.inf.model.eval()
        self.ip = self.inf.processor.image_processor
        self._dummy = Image.new("RGB", (_FIXED_HW, _FIXED_HW))

    def _diff_pixel_values(self, images01: torch.Tensor):
        """Per-image differentiable preprocess -> (pixel_values, image_grid_thw). images01: [B,3,H,W] in [0,1].
        _preprocess_differentiable conflates batch with the temporal axis, so call it PER image and concat."""
        # The model FIXES the input grid at 256*28*28 px -> 448x448 (14-px patch, 2x2 merge -> 16x16 = 256
        # image tokens, matching the dummy-image tokenization in prepare_batch). Feeding the raw rollout
        # resolution (e.g. 512) makes the differentiable smart_resize compute a mismatched grid (504) that the
        # patchify view can't reshape -> RuntimeError. Resize to the fixed 448 here (differentiable bicubic).
        if images01.shape[-1] != _FIXED_HW or images01.shape[-2] != _FIXED_HW:
            images01 = torch.nn.functional.interpolate(
                images01, size=(_FIXED_HW, _FIXED_HW), mode="bicubic", align_corners=False, antialias=True
            ).clamp(0, 1)
        fps, grids = [], []
        for i in range(images01.shape[0]):
            fp, (gt, gh, gw) = self.ip._preprocess(images01[i:i + 1], do_rescale=False)  # tensor -> diff path
            fps.append(fp)
            grids.append([gt, gh, gw])
        pixel_values = torch.cat(fps, dim=0).to(self.dtype)
        image_grid_thw = torch.tensor(grids, device=images01.device)
        return pixel_values, image_grid_thw

    def _scores(self, images01: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = list(prompts)
        if images01.dim() == 3:
            images01 = images01.unsqueeze(0)
        images01 = images01.to(self.device)
        B = images01.shape[0]
        # official text/tokenization with dummy images (image content irrelevant -> replaced below)
        batch = self.inf.prepare_batch([self._dummy] * B, prompts)
        pv, grid = self._diff_pixel_values(images01)
        batch["pixel_values"] = pv.to(self.device)
        batch["image_grid_thw"] = grid.to(self.device)
        rewards = self.inf.model(return_dict=True, **batch)["logits"]  # [B, 2] = (mu, sigma)
        return rewards[:, 0].float()

    # Chunk the 7B reward FORWARD (no-grad scoring path used by the NFT baseline + eval). Scoring a full
    # K-repeat rollout batch (e.g. 24 imgs) through the 7B VLM at once OOMs next to SD3; process CHUNK at a
    # time. (The OPA ascent uses opa_mb=1 + grad-checkpointing for the differentiable path separately.)
    HPSV3_FWD_CHUNK = int(os.environ.get("HPSV3_FWD_CHUNK", "4"))

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = list(prompts)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        outs = []
        for i in range(0, images.shape[0], self.HPSV3_FWD_CHUNK):
            outs.append(self._scores(images[i:i + self.HPSV3_FWD_CHUNK], prompts[i:i + self.HPSV3_FWD_CHUNK]).float())
        return torch.cat(outs, dim=0)


# --- process-level singleton -------------------------------------------------------------
# The 7B Qwen2-VL reward weighs ~14 GB. Training instantiates it THREE times per rank
# (reward_fn + eval_reward_fn in multi_score, plus the OPA-ascent ri_scorer) -> ~42 GB ->
# OOM next to SD3. All three want the SAME frozen model, so hand out one shared instance per
# (device, checkpoint). Safe: params frozen, scoring functional (no per-call state).
_HPSV3_CACHE: dict = {}


def get_hpsv3_scorer(device: str = "cuda", **kwargs) -> "HPSv3Scorer":
    key = (str(device), kwargs.get("checkpoint_path", DEFAULT_CHECKPOINT))
    s = _HPSV3_CACHE.get(key)
    if s is None:
        s = HPSv3Scorer(device=device, **kwargs)
        _HPSV3_CACHE[key] = s
    return s
