"""Differentiable DeQA-Score image-quality reward via 5-token MOS readout (Q-Align's sibling).

`zhiyuanyou/DeQA-Score-Mix3` shares Q-Align's backbone and class — a mPLUG-Owl2 VLM
(CLIP-ViT-L/14@448 + a 64-query visual abstractor + LLaMA-2-7B, ``MPLUGOwl2LlamaForCausalLM``) —
but is trained with SOFT distribution labels (DeQA-Score) rather than the hard best-token target.
The scalar READOUT is IDENTICAL to Q-Align: the EXPECTED MOS over 5 rating-level tokens
{excellent, good, fair, poor, bad} at the final position,
E[MOS] = softmax(logits[-1, preferential_ids_]) @ [5,4,3,2,1], in [1,5].

We reproduce the model's own ``score(task_="quality")`` in a SINGLE forward — no ``.generate()``,
no ``torch.inference_mode``, no PIL — so the reward is differentiable w.r.t. the input image and
can drive OPA's reward-gradient target ascent, exactly like CLIPScore and HPSv3.

Verified against zhiyuanyou/DeQA-Score-Mix3 ``modeling_mplug_owl2_huggingface.py`` (``score``):
- prompt (task_="quality", input_="image"), verbatim:
    "USER: How would you rate the quality of this image?\n<|image|>\nASSISTANT: The quality of the image is"
  The rating word is the VERY NEXT token, so read ``logits[:, -1, preferential_ids_]``.
- ``preferential_ids_ = [tok(w)["input_ids"][1] for w in ["excellent","good","fair","poor","bad"]]``
  (the model exposes this list); weights = [5.,4.,3.,2.,1.]; readout = ``softmax(.) @ weights``.
  DeQA's soft-label training only changes the learned distribution, NOT this expected-value readout
  (README: ``model.score([img])`` -> MOS in [1,5], higher is better; e.g. 1.9404).
- The single ``<|image|>`` (IMAGE_TOKEN_INDEX=-200) placeholder is expanded to the 64 abstractor
  query tokens INSIDE ``forward``'s ``prepare_inputs_labels_for_multimodal`` (differentiable) — we
  do NOT pre-expand it. We only swap the PIL preprocess for a torch one.
- CLIP preprocess (preprocessor_config.json, identical to Q-Align): size=crop_size=448, bicubic
  (resample=3), center-crop, normalize mean=[0.48145466,0.4578275,0.40821073]
  std=[0.26862954,0.26130258,0.27577711]. For a square input (SD3 rollouts are square)
  ``expand2square`` is identity, so resize->448 reproduces the PIL resize->center-crop path.

DeQA quality is NO-REFERENCE: the diffusion ``prompts`` argument is accepted for API parity but
unused. The class follows the same differentiable scorer interface as HPSv3.
"""

from typing import List, Sequence
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

DEFAULT_MODEL_PATH = os.environ.get("DEQA_MODEL_PATH", "zhiyuanyou/DeQA-Score-Mix3")
DEFAULT_MODEL_REVISION = os.environ.get(
    "DEQA_MODEL_REVISION", "246e219b7e3bc9b5dff5269c1a8564087aefca3f"
)

# mPLUG-Owl2 image placeholder + out-of-vocab index (from the modeling code). -200 is routed
# through prepare_inputs_labels_for_multimodal BEFORE embedding, so ``images`` MUST be passed.
IMAGE_TOKEN = "<|image|>"
IMAGE_TOKEN_INDEX = -200
# DeQA / Q-Align quality task prompt (task_="quality", input_="image"), verbatim from score().
QUALITY_PROMPT = (
    "USER: How would you rate the quality of this image?\n"
    f"{IMAGE_TOKEN}\n"
    "ASSISTANT: The quality of the image is"
)
# CLIP preprocess fallbacks (== preprocessor_config.json); read from the model when available.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_IMAGE_SIZE = 448
# MOS rating weights, aligned to preferential_ids_ = [excellent, good, fair, poor, bad].
_MOS_WEIGHTS = (5.0, 4.0, 3.0, 2.0, 1.0)


def _patch_deqa_remote_code_compat() -> None:
    """Expose legacy LLaMA symbols used by DeQA's pinned remote module.

    Transformers 4.51 narrows ``modeling_llama.__all__`` to public model
    classes.  DeQA's older module uses a star import and still expects these
    helpers, so make that import explicit without editing the downloaded Hub
    source or downgrading the Z-Image-compatible Transformers stack.
    """

    from transformers.cache_utils import Cache
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
    from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
    from transformers.models.llama import modeling_llama
    from transformers.utils import is_flash_attn_greater_or_equal_2_10

    exported = {
        "Cache": Cache,
        "BaseModelOutputWithPast": BaseModelOutputWithPast,
        "CausalLMOutputWithPast": CausalLMOutputWithPast,
        "_flash_attention_forward": _flash_attention_forward,
        "is_flash_attn_greater_or_equal_2_10": is_flash_attn_greater_or_equal_2_10,
        "apply_rotary_pos_emb": modeling_llama.apply_rotary_pos_emb,
        "repeat_kv": modeling_llama.repeat_kv,
    }
    for name, value in exported.items():
        setattr(modeling_llama, name, value)
    names = list(getattr(modeling_llama, "__all__", ()))
    modeling_llama.__all__ = names + [name for name in exported if name not in names]


def tokenizer_image_token(
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    image_token_index: int = IMAGE_TOKEN_INDEX,
) -> torch.Tensor:
    """Port of Q-Align/LLaVA ``tokenizer_image_token``.

    Tokenizes the text around each ``<|image|>``, inserting ``image_token_index`` (-200) at the
    placeholder and de-duplicating the LLaMA BOS. Returns a 1-D LongTensor. Pure-int / no grad
    (the text is constant across a rollout).
    """
    chunks = [
        tokenizer(chunk).input_ids if len(chunk) > 0 else []
        for chunk in prompt.split(IMAGE_TOKEN)
    ]

    def insert_separator(seq: List[list], sep: list) -> list:
        return [ele for pair in zip(seq, [sep] * len(seq)) for ele in pair][:-1]

    input_ids: List[int] = []
    offset = 0
    if len(chunks) > 0 and len(chunks[0]) > 0 and chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(chunks[0][0])
    for x in insert_separator(chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])
    return torch.tensor(input_ids, dtype=torch.long)


class DeQAScorer(torch.nn.Module):
    """Differentiable no-reference image-quality reward from zhiyuanyou/DeQA-Score-Mix3."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        model_revision: str = DEFAULT_MODEL_REVISION,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        _patch_deqa_remote_code_compat()
        # mPLUG-Owl2 custom modeling via trust_remote_code
        # (auto_map -> modeling_mplug_owl2_huggingface.MPLUGOwl2LlamaForCausalLM).
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=model_revision,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        self.model = self.model.eval().to(device)
        self.model.requires_grad_(False)  # frozen; grad still flows THROUGH to the input image.

        # transformers>=4.49 removed the module-level ``_use_flash_attention_2`` / ``_use_sdpa`` flags
        # that this 2023-era vendored ``modeling_llama2.py`` still reads in ``model_forward`` (a bare
        # ``self._use_flash_attention_2`` -> AttributeError at the first forward). Default both to False
        # on every submodule so the eager 4d-causal-mask path (``_prepare_4d_causal_attention_mask``,
        # fully differentiable) is taken — we never want flash/sdpa-specific masking for the reward.
        for _m in self.model.modules():
            if not hasattr(_m, "_use_flash_attention_2"):
                _m._use_flash_attention_2 = False
            if not hasattr(_m, "_use_sdpa"):
                _m._use_sdpa = False

        # The model attaches its own tokenizer / CLIP image processor / preferential_ids_ in
        # __init__ (DeQA uses config._name_or_path -> offline-safe); fall back if absent.
        self.tokenizer = getattr(self.model, "tokenizer", None)
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, revision=model_revision, trust_remote_code=True
            )
        image_processor = getattr(self.model, "image_processor", None)
        if image_processor is None:
            from transformers import CLIPImageProcessor

            image_processor = CLIPImageProcessor.from_pretrained(
                model_path, revision=model_revision
            )

        # 5 rating-level token ids [excellent, good, fair, poor, bad], aligned to _MOS_WEIGHTS.
        pref = getattr(self.model, "preferential_ids_", None)
        if pref is None:
            pref = [
                ids[1]
                for ids in self.tokenizer(["excellent", "good", "fair", "poor", "bad"])["input_ids"]
            ]
        self.preferential_ids_ = torch.tensor(list(pref), dtype=torch.long, device=device)
        self.weight_tensor = torch.tensor(_MOS_WEIGHTS, device=device)

        # CLIP preprocess geometry / normalization from the processor (fallback to constants).
        mean = tuple(getattr(image_processor, "image_mean", _CLIP_MEAN))
        std = tuple(getattr(image_processor, "image_std", _CLIP_STD))
        self.res = self._crop_size(image_processor)
        self.register_buffer("_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        # Text is CONSTANT (no-reference quality -> the diffusion prompt is unused), so tokenize
        # the -200-placeholder prompt ONCE and reuse it for every image.
        self.input_ids = tokenizer_image_token(QUALITY_PROMPT, self.tokenizer).to(device)

    @staticmethod
    def _crop_size(image_processor: object) -> int:
        cs = getattr(image_processor, "crop_size", None)
        if isinstance(cs, dict):
            return int(cs.get("height", _IMAGE_SIZE))
        if isinstance(cs, int):
            return cs
        return _IMAGE_SIZE

    def _preprocess(self, images01: torch.Tensor) -> torch.Tensor:
        """``[B,3,H,W]`` in [0,1] -> CLIP pixel_values ``[B,3,res,res]``, differentiable (no PIL).

        Assumes square inputs (SD3 rollouts): ``expand2square`` is then identity, so a direct
        bicubic resize to ``res`` reproduces the PIL resize->center-crop path.
        [grad-gate] non-square inputs would need a mean-color pad-to-square first.
        Resize in float32 for a stable bicubic, then the (differentiable, affine) CLIP normalize;
        images01 is already rescaled to [0,1] so no /255 is applied.
        """
        x = images01.to(torch.float32)
        x = F.interpolate(
            x, size=(self.res, self.res), mode="bicubic", align_corners=False, antialias=True
        )
        x = (x - self._mean.to(x)) / self._std.to(x)
        return x.to(self.dtype)

    def _scores(self, images01: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        """Differentiable per-image E[MOS] in [1,5]. ``images01``: ``[B,3,H,W]`` in [0,1] -> ``[B]``.

        ``prompts`` is accepted for API parity but UNUSED — DeQA quality is no-reference.
        """
        del prompts  # no-reference metric: the diffusion prompt does not enter the score.
        if images01.dim() == 3:
            images01 = images01.unsqueeze(0)
        images01 = images01.to(self.device)
        b = images01.shape[0]
        pixel_values = self._preprocess(images01)                       # [B,3,res,res], grad intact
        input_ids = self.input_ids.unsqueeze(0).repeat(b, 1)            # [B,L], constant text

        # forward() expands each sample's single -200 into 64 abstractor tokens
        # (prepare_inputs_labels_for_multimodal); identical prompt + image-token counts across the
        # batch -> non-ragged sequences, so a plain batched forward is safe (matches score()).
        out = self.model(
            input_ids=input_ids, images=pixel_values, use_cache=False, return_dict=True
        )
        logits = out.logits[:, -1, self.preferential_ids_]              # [B,5], next-token dist
        p = torch.softmax(logits.float(), dim=-1)
        return (p * self.weight_tensor.to(p)).sum(dim=-1)               # E[MOS] in [1,5]

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        return self._scores(images, prompts).float()


# --- Process-wide singleton (memory) ---------------------------------------------------------
# The main-table refl/opsd path builds THREE DeQAScorers per rank — reward_fn + eval_reward_fn (both
# via diffusionopsd.rewards.deqa_score) and the differentiable ri_scorer (train_opsd_ri_sd3._load_reward_scorer)
# — i.e. ~3x ~16 GB of frozen mPLUG-Owl2-7B weights before any activations, which alone OOMs an 80 GB card
# on the ReFL/OPSD backprop path (nft only forward-scores, so it never hit this). All three want the SAME
# frozen model, so hand out ONE shared instance per (device, model_path). Safe: params are frozen and
# scoring is functional (no per-call state); the no-grad __call__ and the grad _scores share the weights.
# Mirrors diffusionopsd.internvl_t2i_scorer.get_internvl_t2i_scorer.
_SCORER_CACHE: dict = {}


def get_deqa_scorer(
    device: str = "cuda",
    model_path: str = DEFAULT_MODEL_PATH,
    model_revision: str = DEFAULT_MODEL_REVISION,
    **kwargs,
) -> "DeQAScorer":
    key = (str(device), model_path, model_revision)
    scorer = _SCORER_CACHE.get(key)
    if scorer is None:
        scorer = DeQAScorer(
            model_path=model_path,
            model_revision=model_revision,
            device=device,
            **kwargs,
        )
        _SCORER_CACHE[key] = scorer
    return scorer
