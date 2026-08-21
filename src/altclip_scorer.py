"""AltCLIP T2I reward scorer (differentiable) for the DiffusionOPSD cross-reward study.

Loads the BAAI/AltCLIP architecture + the paper's fine-tuned checkpoint (keys are 'clip.'-prefixed) and
scores an image/prompt pair by normalized text-image cosine similarity. The core `_scores` keeps the
autograd graph (differentiable CLIP-style preprocessing via get_image_transform, model params frozen
but input grad flows), so it can drive OPA's reward-gradient target ascent exactly like clipscore/
pickscore. `__call__` wraps it in no_grad for rollout/eval scoring.

Ref: MaskEdit src/awm_rm/altclip_scorer.py (the original eval-only version). The base architecture
is downloaded from Hugging Face. The fine-tuned evaluator is internal and is not distributed;
``ALTCLIP_MODEL_PATH`` is therefore required. The scorer fails closed rather than silently using
randomly initialized weights.
"""

import os

import torch
from transformers import AltCLIPConfig, AltCLIPModel, AltCLIPProcessor

from diffusionopsd.clip_scorer import get_image_transform

DEFAULT_MODEL_PATH = os.environ.get("ALTCLIP_MODEL_PATH", "")
DEFAULT_BASE_MODEL = "BAAI/AltCLIP"


class AltCLIPScorer(torch.nn.Module):
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str = "cuda", dtype=torch.float32):
        super().__init__()
        if not model_path:
            raise RuntimeError(
                "ALTCLIP_MODEL_PATH is required. The paper's AltCLIP-architecture evaluator uses "
                "an internal fine-tuned checkpoint that is not distributed."
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ALTCLIP_MODEL_PATH does not exist: {model_path}")
        self.device = device
        self.dtype = dtype
        self.processor = AltCLIPProcessor.from_pretrained(DEFAULT_BASE_MODEL)
        if model_path and os.path.isdir(model_path):
            self.model = AltCLIPModel.from_pretrained(model_path)
        elif model_path.endswith((".pt", ".pth")):
            self.model = AltCLIPModel(AltCLIPConfig.from_pretrained(DEFAULT_BASE_MODEL))
            sd = torch.load(model_path, map_location="cpu")
            sd = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
            cleaned = {(k[len("clip."):] if k.startswith("clip.") else k): v for k, v in sd.items()}
            missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
            print(f"[AltCLIP] loaded {os.path.basename(model_path)} "
                  f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        else:
            raise ValueError(
                "ALTCLIP_MODEL_PATH must be a Transformers model directory or a .pt/.pth checkpoint"
            )
        self.model = self.model.eval().to(device, dtype=dtype)
        self.model.requires_grad_(False)
        self.tform = get_image_transform(self.processor.image_processor)  # differentiable Resize/Crop/Normalize

    def _process(self, pixels):
        dtype = pixels.dtype
        return self.tform(pixels).to(dtype=dtype)

    def _scores(self, images01, prompts):
        """Differentiable per-pair cosine score. images01: [B,3,H,W] in [0,1]."""
        text = self.processor(text=list(prompts), padding=True, truncation=True,
                              max_length=77, return_tensors="pt")
        text = {"input_ids": text["input_ids"][:, :512].to(self.device),
                "attention_mask": text["attention_mask"][:, :512].to(self.device)}
        pixels = self._process(images01).to(device=self.device, dtype=self.dtype)
        txt_e = self.model.get_text_features(**text)
        img_e = self.model.get_image_features(pixel_values=pixels)
        txt_e = txt_e / txt_e.norm(p=2, dim=-1, keepdim=True)
        img_e = img_e / img_e.norm(p=2, dim=-1, keepdim=True)
        return (txt_e * img_e).sum(-1)

    @torch.no_grad()
    def __call__(self, images, prompts):
        return self._scores(images, prompts).float()
