import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import ImageReward as RM

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except Exception:
    BICUBIC = Image.BICUBIC


class ImageRewardScorer(nn.Module):
    """
    Differentiable ImageReward scorer.

    _scores(images01, prompts):
      images01: torch Tensor [B,3,H,W] in [0,1]
      returns:  torch Tensor [B], gradient flows to images01

    __call__(prompts, images):
      no_grad eval wrapper for NFT sampling/eval.
    """

    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        root = os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "ImageReward",
        )
        # Offline-first: prefer pre-downloaded local weights. RM.load accepts a file path as `name` and an explicit
        # med_config, avoiding any hub download for the reward weights (only BLIP's
        # bert-base-uncased still resolves via the HF cache / mirror).
        local_pt = os.path.join(root, "ImageReward.pt")
        local_med = os.path.join(root, "med_config.json")
        if os.path.isfile(local_pt) and os.path.isfile(local_med):
            model = RM.load(local_pt, device=device, download_root=root, med_config=local_med)
        else:
            model = RM.load("ImageReward-v1.0", device=device, download_root=root)
        self.model = model.eval().to(dtype=dtype)
        self.model.requires_grad_(False)

        self.resize = T.Resize(224, interpolation=BICUBIC, antialias=True)
        self.crop = T.CenterCrop(224)
        self.norm = T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )

    def _pil_list_to_tensor(self, images):
        xs = []
        for im in images:
            if isinstance(im, str):
                im = Image.open(im).convert("RGB")
            elif isinstance(im, Image.Image):
                im = im.convert("RGB")
            else:
                raise TypeError(f"Unsupported image type: {type(im)}")
            xs.append(TF.to_tensor(im))
        return torch.stack(xs, dim=0)

    def _process(self, images01):
        if not isinstance(images01, torch.Tensor):
            images01 = self._pil_list_to_tensor(images01)
        x = images01.to(device=self.device, dtype=self.dtype)
        x = x.clamp(0, 1)
        x = self.resize(x)
        x = self.crop(x)
        x = self.norm(x)
        return x

    def _tokenize(self, prompts):
        return self.model.blip.tokenizer(
            list(prompts),
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(self.device)

    def _scores(self, images01, prompts):
        # Differentiable path. Do NOT wrap in no_grad.
        pixels = self._process(images01)
        text = self._tokenize(prompts)
        rewards = self.model.score_gard(
            text.input_ids,
            text.attention_mask,
            pixels,
        )
        return rewards.view(-1).float()

    @torch.no_grad()
    def __call__(self, prompts, images):
        return self._scores(images, prompts).contiguous()
