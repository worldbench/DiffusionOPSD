"""INTERNAL EVALUATOR: differentiable InternVL2-26B T2I pointwise reward.

The paper checkpoint is not distributed and this evaluator is not one of the
seven public reward presets. ``INTERNVL_T2I_CKPT`` must name the internal
fine-tuned checkpoint; the adapter fails closed before loading the 26B base.

This wraps a pointwise InternVL judge checkpoint supplied through ``INTERNVL_T2I_CKPT``
(an ``InternVLChatModel`` = InternViT-6B + mlp1 + InternLM2-20B, i.e. the OpenGVLab
**InternVL2-26B** architecture — verified 25.5 B params, LM hidden 6144 / 48 layers, ViT 45
layers, all keys prefixed ``model.`` — finetuned to emit a 1-5 quality score per dimension)
and turns it into a text-to-image reward that is DIFFERENTIABLE w.r.t. the input image pixels,
so it can drive OPA's reward-gradient target ascent exactly like clipscore/pickscore/hpsv3.

MEMORY: 26 B in bf16 is approximately 51 GB/GPU. Forward-only scoring fits on one 80 GB GPU with
chunking; differentiable target ascent backpropagates through the 26 B model to the image, so
gradient checkpointing is enabled by default.

The checkpoint was trained on **edit** pointwise annotations (9 dimensions, source+target+
instruction). For the paper protocol it is repurposed for **T2I** with a SINGLE dimension —
``指令跟随`` (instruction following) — by swapping in the T2I prompt template and feeding
ONLY the generated image (no source image). The scalar reward is read from the model's own
score-token logits (no external head): teacher-force ``指令跟随：`` and take a softmax over
the five rating-digit tokens (1..5) at the predicting position. By default the reward is
``P(score ∈ {5})`` — the "high-score utility" the checkpoint name refers to — matching the
original scorer's ``select_reward / whole_reward`` readout with ``select_scores=[5]``.

Differentiability is achieved by (1) reimplementing InternVL's pad→resize→normalize
preprocessing with pure torch ops (no PIL, no ``.numpy()``), and (2) running the InternVL
forward MANUALLY — ``extract_feature`` → scatter ViT embeds into the ``<IMG_CONTEXT>``
positions → ``language_model`` — inside the autograd graph, never ``.generate()`` and never
``torch.no_grad`` in the scoring path.

The implementation uses a score-token readout with InternVL tiling/scatter.
"""

import os
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

# --- paths (environment-overridable for local or shared model storage) ---
DEFAULT_CHECKPOINT = os.environ.get(
    "INTERNVL_T2I_CKPT",
    "",
)
# InternVL2-26B *architecture* dir: needs config.json (with auto_map + the 26B dims) +
# modeling_*.py + tokenizer files. Weights are irrelevant (overwritten by the .bin), so a
# config+code+tokenizer-only snapshot suffices. Falls back to the origin path if present.
DEFAULT_BASE_ARCH = os.environ.get(
    "INTERNVL_T2I_BASE",
    "OpenGVLab/InternVL2-26B",
)
_ORIGIN_BASE_ARCH = "OpenGVLab/InternVL2-26B"

# --- InternVL2-2B constants ---
TILE_SIZE = 448                      # single-tile input (r_size=448 in the reference config)
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"  # ViT embeds are scattered onto these positions
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# InternLM2 (non-4b) chat wrapper; <user_prompt> is filled with the T2I judge instruction.
PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, "
    "是一个有用无害的人工智能助手。<|im_end|>"
    "<|im_start|>user\n<user_prompt><|im_end|><|im_start|>assistant\n"
)

# --- T2I pointwise prompt ---
TEXTIMG_POINTWISE_DIM_NAME = "指令跟随"
PREFILL_SCORE = 3                     # placeholder digit used only to locate the read position


def build_textimg_user_prompt_head() -> str:
    """Prefix before ``<image>`` (same text as the T2I point-SFT builder)."""
    return (
        "你将扮演一名专业的图像生成质量评估师。你的任务是针对一张根据指定提示词生成的图片，"
        "评估其对生成指令的跟随程度。\n"
        f"评估维度：{TEXTIMG_POINTWISE_DIM_NAME}。\n\n"
        "评分标准（1-5分，质量由差到好）：\n"
        f"- {TEXTIMG_POINTWISE_DIM_NAME}：评估生成图片与提示词描述的匹配程度，包括主体、场景、风格、"
        "细节等是否忠实还原指令要求。"
        "1分：生成内容与提示词严重不符，主体/场景/风格完全偏离指令；"
        "2分：仅部分元素与指令相关，整体偏离较大；"
        "3分：基本符合指令的核心要求，但部分细节有缺失或偏差；"
        "4分：较好地还原了指令描述，仅有个别细节存在轻微偏差；"
        "5分：完全忠实还原提示词的所有要求，主体/场景/风格/细节全部精准匹配。\n\n"
        "现给定如下图片："
    )


def build_textimg_user_prompt_suffix(prompt_text: str) -> str:
    """Suffix after ``<image>`` (embeds the T2I prompt)."""
    return (
        f"以及对应的提示词：{prompt_text}。\n"
        "请评估该图片对提示词指令的跟随程度，输出评估分数。"
    )


def _force_nonreentrant_checkpoint() -> None:
    """Make torch.utils.checkpoint.checkpoint default to use_reentrant=False when unspecified.

    InternVL's vendored modeling calls ``torch.utils.checkpoint.checkpoint(layer, ...)`` directly
    (modeling_internlm2.py / modeling_intern_vit.py) with NO ``use_reentrant`` -> defaults to True
    (reentrant). Reentrant checkpointing is INCOMPATIBLE with ``torch.autograd.grad(out, inputs)``,
    which is exactly what OPA's reward-gradient ascent (``_opa_tr_step``) uses -> RuntimeError. We
    default it to False (non-reentrant, the recommended mode); callers that pass use_reentrant
    explicitly (e.g. diffusers' SD3 checkpointing) are untouched. Idempotent + module-attribute
    patch (the modeling looks the symbol up at call time, so this takes effect)."""
    import torch.utils.checkpoint as _c
    if getattr(_c.checkpoint, "_opa_nonreentrant", False):
        return
    _orig = _c.checkpoint

    def _wrapped(*args, use_reentrant=None, **kwargs):
        if use_reentrant is None:
            use_reentrant = False
        return _orig(*args, use_reentrant=use_reentrant, **kwargs)

    _wrapped._opa_nonreentrant = True
    _c.checkpoint = _wrapped


def _pad_to_square(x: torch.Tensor, value: float = 1.0) -> torch.Tensor:
    """Differentiable center pad of [1,3,H,W] to a square with a constant fill (white=1.0)."""
    _, _, h, w = x.shape
    if h == w:
        return x
    m = max(h, w)
    pt, pl = (m - h) // 2, (m - w) // 2
    pb, pr = m - h - pt, m - w - pl
    return F.pad(x, (pl, pr, pt, pb), value=value)  # F.pad order = (left, right, top, bottom)


def _resolve_base_arch(base_arch: str) -> str:
    """Prefer the explicit/downloaded arch dir; fall back to the reference origin path."""
    if os.path.isdir(base_arch) and os.path.exists(os.path.join(base_arch, "config.json")):
        return base_arch
    if os.path.isdir(_ORIGIN_BASE_ARCH) and os.path.exists(os.path.join(_ORIGIN_BASE_ARCH, "config.json")):
        return _ORIGIN_BASE_ARCH
    return base_arch  # let AutoConfig raise a clear error if neither exists


def _load_finetuned_state_dict(model: nn.Module, checkpoint_path: str) -> None:
    """Load the .bin finetune onto ``model`` (the InternVLChatModel), robust to wrappers/prefixes.

    The .bin may be (a) a raw ``InternVLChatModel.state_dict()``, (b) wrapped under a
    'state_dict'/'module'/'model' key, and/or (c) prefixed with 'module.' (DDP) or 'model.'
    (if saved from the ``Internvl2_instruct_edit_point`` wrapper whose submodule is ``self.model``).
    We normalize all of these, then load with strict=False and report the match.
    """
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        for key in ("state_dict", "module", "model_state_dict", "model"):
            inner = blob.get(key)
            if isinstance(inner, dict) and any(torch.is_tensor(v) for v in inner.values()):
                blob = inner
                break
    if not isinstance(blob, dict):
        raise TypeError(f"Unexpected checkpoint object type: {type(blob)}")

    target_keys = set(model.state_dict().keys())

    def _strip(prefix: str, sd: dict) -> dict:
        return {k[len(prefix):]: v for k, v in sd.items()}

    # Try candidate prefix strips; keep the variant that best matches the model's keys.
    candidates = {"": blob}
    for pref in ("module.", "model.", "module.model.", "model.model."):
        if sum(k.startswith(pref) for k in blob) > 0.5 * len(blob):
            candidates[pref] = _strip(pref, blob)
    best_pref, best_sd, best_hit = "", blob, -1
    for pref, sd in candidates.items():
        hit = len(target_keys & set(sd.keys()))
        if hit > best_hit:
            best_pref, best_sd, best_hit = pref, sd, hit

    missing, unexpected = model.load_state_dict(best_sd, strict=False)
    print(
        f"[internvl_t2i] loaded {checkpoint_path}\n"
        f"  prefix_stripped={best_pref!r}  matched={best_hit}/{len(target_keys)} model keys "
        f"({len(best_sd)} ckpt keys)  missing={len(missing)}  unexpected={len(unexpected)}"
    )
    if best_hit < 0.5 * len(target_keys):
        raise RuntimeError(
            f"[internvl_t2i] checkpoint matched only {best_hit}/{len(target_keys)} keys — "
            f"architecture/prefix mismatch. Sample ckpt keys: {list(best_sd)[:5]}"
        )


def _balanced_layer_devices(n_layers: int, devices: List[int], primary: int) -> List[int]:
    """Contiguous per-layer device assignment balancing param mass. `primary` also carries the ViT
    (~12GB) + embeddings + lm-head, so it is weighted lighter (0.55) and gets proportionally fewer
    decoder layers. Contiguous (early layers on primary, later on the spare) => only 2 device hops."""
    weights = [(0.55 if d == primary else 1.0) for d in devices]
    tot = sum(weights)
    counts = [max(1, int(round(n_layers * w / tot))) for w in weights]
    while sum(counts) > n_layers:
        counts[counts.index(max(counts))] -= 1
    while sum(counts) < n_layers:
        counts[counts.index(min(counts))] += 1
    order: List[int] = []
    for d, c in zip(devices, counts):
        order.extend([d] * c)
    return order[:n_layers]


def _build_internvl_device_map(model, primary: int, devices: List[int]) -> dict:
    """Full module->GPU map for dispatch_model: IO modules pinned to `primary`, decoder layers
    balanced across `devices`. Built by introspection (robust to exact submodule names). A coverage
    check raises a clear error if any parameter is left unmapped (extend this for a new arch)."""
    dmap: dict = {}
    for name, _ in model.named_children():  # vision_model, mlp1, ... (everything but the LM) -> primary
        if name != "language_model":
            dmap[name] = primary
    lm = model.language_model
    order = _balanced_layer_devices(len(lm.model.layers), devices, primary)
    for i in range(len(lm.model.layers)):
        dmap[f"language_model.model.layers.{i}"] = order[i]
    for name, _ in lm.named_children():  # LM children except decoder layers -> primary
        if name == "model":
            for sub, _ in lm.model.named_children():  # tok_embeddings, norm, rotary, (layers handled above)
                if sub != "layers":
                    dmap[f"language_model.model.{sub}"] = primary
        else:
            dmap[f"language_model.{name}"] = primary  # e.g. `output` (lm head)
    covered = set(dmap.keys())
    missing = [pn for pn, _ in model.named_parameters()
               if not any(pn == k or pn.startswith(k + ".") for k in covered)]
    if missing:
        raise RuntimeError(
            f"[internvl_t2i] device_map leaves {len(missing)} params unmapped, e.g. {missing[:3]}; "
            f"extend _build_internvl_device_map for this architecture.")
    return dmap


class InternVLT2IPointScorer(torch.nn.Module):
    """Differentiable T2I instruction-following reward from the InternVL2-2B pointwise judge."""

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        base_arch: str = DEFAULT_BASE_ARCH,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        select_scores: Sequence[int] = (5,),
        readout: str = os.environ.get("INTERNVL_T2I_READOUT", "phigh"),  # "phigh" | "expected"
        device_map_devices: Optional[List[int]] = None,  # GATED (INTERNVL_BRIDGE server): shard 26B over these GPUs
    ):
        super().__init__()
        if not checkpoint_path:
            raise RuntimeError(
                "INTERNVL_T2I_CKPT is required. The paper's pointwise evaluator checkpoint is internal."
            )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"INTERNVL_T2I_CKPT does not exist: {checkpoint_path}")
        self.device = device
        self.dtype = dtype
        self.readout = readout
        # which scores count as the "high-score utility" reward (env: e.g. "4,5")
        env_sel = os.environ.get("INTERNVL_T2I_SELECT")
        if env_sel:
            select_scores = tuple(int(s) for s in env_sel.split(",") if s.strip())
        self.select_scores = tuple(select_scores)

        base_arch = _resolve_base_arch(base_arch)
        use_flash = os.environ.get("INTERNVL_T2I_FLASH", "0") == "1"

        config = AutoConfig.from_pretrained(base_arch, trust_remote_code=True)
        # Build the architecture, then overwrite with the finetune .bin. CRITICAL: wrap in
        # no_init_weights() — otherwise from_config runs _init_weights (.normal_()) over all 25B
        # params on CPU (~15 min/rank, pointless since the .bin overwrites them). no_init skips the
        # random fill but still allocates the tensors; register_buffer values (rope inv_freq, masks)
        # are computed in __init__ so remain correct.
        from transformers.modeling_utils import no_init_weights
        with no_init_weights():
            try:
                self.model = AutoModel.from_config(
                    config, torch_dtype=dtype, use_flash_attn=use_flash, trust_remote_code=True
                )
            except TypeError:
                # some modeling versions don't accept use_flash_attn in from_config
                self.model = AutoModel.from_config(config, torch_dtype=dtype, trust_remote_code=True)
        _load_finetuned_state_dict(self.model, checkpoint_path)
        if device_map_devices and len(device_map_devices) > 1:
            # GATED multi-GPU shard (INTERNVL_BRIDGE reward server only). Default single-GPU path
            # below is byte-for-byte unchanged.
            self._dispatch_multi_gpu(device, [int(d) for d in device_map_devices], dtype)
        else:
            self.model = self.model.to(device=device, dtype=dtype)
        self.model = self.model.eval()
        self.model.requires_grad_(False)
        # 26B backprop (OPA ascent through the frozen VLM to the image) retains the full forward
        # activations otherwise -> OOM. Enable InternVL's grad-checkpointing and put the model in
        # train() to activate the checkpoint gate. Params are frozen and InternVL2 RM configs have
        # dropout=0 / drop_path=0, so train() is numerically equivalent to eval() for the reward.
        if os.environ.get("INTERNVL_T2I_GRADCKPT", "1") == "1":
            self.gradient_checkpointing_enable()
            self.model.train()

        self.tokenizer = AutoTokenizer.from_pretrained(base_arch, trust_remote_code=True, use_fast=False)
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id

        # tokens contributed by one 448 tile (2B: 256). Reference divides by downsample_ratio
        # (448 // vit_res)^2, which is 1 for our single-448-tile path.
        self.num_image_token = int(self.model.num_image_token)

        # rating-digit token ids 1..5 (each a single token for the InternLM2 tokenizer)
        self.score_token_ids = torch.tensor(
            [self.tokenizer.encode(str(k), add_special_tokens=False)[0] for k in range(1, 6)],
            device=device,
        )
        self.values = torch.arange(1.0, 6.0, device=device)  # [1,2,3,4,5] for E[score]
        self.select_idx = torch.tensor(
            [s - 1 for s in self.select_scores], device=device
        )  # indices into the 5 digits
        self.prefill_token_id = self.tokenizer.encode(str(PREFILL_SCORE), add_special_tokens=False)[0]

        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def gradient_checkpointing_enable(self) -> None:
        """Enable InternVL grad-checkpointing (needed so 26B backprop-to-image fits). Forces
        non-reentrant checkpointing so OPA's torch.autograd.grad(reward, image) works."""
        _force_nonreentrant_checkpoint()
        try:
            self.model.vision_model.gradient_checkpointing = True
            self.model.language_model.config.use_cache = False
            self.model.language_model.gradient_checkpointing = True
            self.model.language_model.model.gradient_checkpointing = True
        except Exception as exc:  # pragma: no cover - best-effort
            print(f"[internvl_t2i] gradient_checkpointing_enable skipped: {exc}")

    def _dispatch_multi_gpu(self, primary_device, devices: List[int], dtype: torch.dtype) -> None:
        """GATED (INTERNVL_BRIDGE server): shard the frozen 26B across `devices` via accelerate
        dispatch_model. The default single-GPU ``.to(device)`` path is untouched.

        IO-critical modules (ViT, projector, LM token-embeddings, final norm, LM head) are PINNED to
        `primary_device` so the manual differentiable forward in ``_score_one`` needs ZERO changes:
        pixel_values / input_ids / score-token buffers all live on primary, ``extract_feature`` and
        the ``<IMG_CONTEXT>`` scatter stay on primary, and the final logits come back on primary
        (lm head pinned there) to be indexed by ``self.score_token_ids`` (also on primary). Only the
        decoder LAYERS are split; accelerate's per-layer AlignDevicesHook moves the hidden state
        across the device boundary with a differentiable ``.to()``, so the reward->image gradient
        (``autograd.grad`` in the server) flows across GPUs exactly as it would on one device."""
        from accelerate import dispatch_model
        primary = int(str(primary_device).split(":")[-1]) if isinstance(primary_device, str) else int(primary_device)
        self.model = self.model.to(dtype=dtype)  # cast on CPU before dispatch (never a single .to(cuda))
        dmap = _build_internvl_device_map(self.model, primary, devices)
        self.model = dispatch_model(self.model, device_map=dmap)
        print(f"[internvl_t2i] sharded 26B across {devices} (primary={primary}); "
              f"{sum(v != primary for v in dmap.values())} modules off-primary")

    # ------------------------------------------------------------------ preprocessing
    def _preprocess_one(self, image01: torch.Tensor) -> torch.Tensor:
        """Differentiable pad→resize(448)→ImageNet-normalize for one image.

        image01: [1,3,H,W] in [0,1]. Returns pixel_values [1,3,448,448] in model dtype.
        Matches the reference config's single-tile path (self.pad=True, r_size=448): letterbox
        to square with white, resize to 448 (differentiable bicubic), then normalize.
        """
        x = image01.to(torch.float32)
        x = _pad_to_square(x, value=1.0)
        x = F.interpolate(x, size=(TILE_SIZE, TILE_SIZE), mode="bicubic",
                          align_corners=False, antialias=True).clamp(0, 1)
        x = (x - self._mean.to(x)) / self._std.to(x)
        return x.to(self.dtype)

    # ------------------------------------------------------------------ prompt build
    def _build_prompt(self, prompt: str) -> str:
        """Full chat prompt with expanded image tokens; the reply is teacher-forced in _score_one."""
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token + IMG_END_TOKEN
        instruct = build_textimg_user_prompt_head() + image_tokens + build_textimg_user_prompt_suffix(prompt)
        return PROMPT_TEMPLATE.replace("<user_prompt>", instruct)

    # ------------------------------------------------------------------ single-image score
    def _score_one(self, image01: torch.Tensor, prompt: str) -> torch.Tensor:
        pixel_values = self._preprocess_one(image01)  # [1,3,448,448]

        cur_prompt = self._build_prompt(prompt)
        enc = self.tokenizer(cur_prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        prompt_len = input_ids.shape[1]

        # Teacher-force "指令跟随：<PREFILL>" so the model must emit the rating digit next; the
        # position that PREDICTS the placeholder digit is where we read the score distribution.
        answer_txt = f"{TEXTIMG_POINTWISE_DIM_NAME}：{PREFILL_SCORE}"
        ans = self.tokenizer(answer_txt, return_tensors="pt", add_special_tokens=False)
        answer_ids = ans["input_ids"].to(self.device)
        answer_mask = ans["attention_mask"].to(self.device)
        # index (in the concatenated seq) whose output logit predicts the digit token
        read_positions = [
            i + prompt_len - 1
            for i in range(answer_ids.shape[1])
            if answer_ids[0, i].item() == self.prefill_token_id
        ]
        assert read_positions, "prefill digit not found in the teacher-forced answer"

        input_ids = torch.cat([input_ids, answer_ids], dim=1)
        attention_mask = torch.cat([attention_mask, answer_mask], dim=1)

        # ---- manual differentiable InternVL forward (scatter ViT embeds into <IMG_CONTEXT>) ----
        vit_embeds = self.model.extract_feature(pixel_values)  # [n_tiles, num_image_token, C]
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        selected = input_ids.reshape(B * N) == self.img_context_token_id
        assert int(selected.sum().item()) == vit_embeds.reshape(-1, C).shape[0], (
            f"IMG_CONTEXT count {int(selected.sum())} != vit tokens {vit_embeds.reshape(-1, C).shape[0]}"
        )
        input_embeds = input_embeds.clone()
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.dtype)
        input_embeds = input_embeds.reshape(B, N, C).to(self.dtype)

        out = self.model.language_model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        logits = out[0] if not hasattr(out, "logits") else out.logits  # [1, N, vocab]

        # read the single T2I dimension's score distribution
        score_logits = logits[0, read_positions[0], :]                 # [vocab]
        p = torch.softmax(score_logits[self.score_token_ids].float(), dim=-1)  # over the 5 digits
        if self.readout == "expected":
            return (p * self.values).sum() / 5.0                       # E[score] normalized to (0,1]
        return p[self.select_idx].sum()                                # P(score ∈ select_scores)

    # ------------------------------------------------------------------ batch score
    def _scores(self, images01: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        """Differentiable per-sample reward. images01: [B,3,H,W] in [0,1]; returns [B].
        A genuine batched InternVL forward (right-pad variable-length prompts +
        per-sample read position) replaces the Python per-sample loop, so opa_mb>1 actually
        batches the 26B model (one padded forward instead of N). A one-time runtime parity check vs the
        per-sample path guards correctness — fails loud on drift, never silently wrong."""
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = list(prompts)
        if images01.dim() == 3:
            images01 = images01.unsqueeze(0)
        images01 = images01.to(self.device)
        B = images01.shape[0]
        assert B == len(prompts), f"batch mismatch: {B} images vs {len(prompts)} prompts"
        if B == 1:
            return self._score_one(images01, prompts[0]).reshape(1)  # [1], autograd graph intact
        if os.environ.get("INTERNVL_C_BATCHED", "0") != "1":
            # Batched _scores is disabled by default because _scores_batched diverges from the
            # exact per-sample path by up to ~0.017 on some InternVL-26B batches (> the 5e-3 parity
            # bound), and the bridge server calls _scores with B>1 unconditionally (it batches across
            # policy ranks), so the batched path is NOT gated by opa_mb. Fall back to the exact
            # per-sample loop; set INTERNVL_C_BATCHED=1 to re-enable and debug the batched forward.
            return torch.cat([self._score_one(images01[i:i + 1], prompts[i]).reshape(1) for i in range(B)])  # [B]
        out = self._scores_batched(images01, prompts)
        if not getattr(self, "_batch_parity_ok", False):
            with torch.no_grad():
                ref = torch.stack([self._score_one(images01[i:i + 1], prompts[i]) for i in range(B)])
            if not (torch.isfinite(out).all() and torch.isfinite(ref).all()):
                raise RuntimeError("[internvl_t2i C] non-finite reward in batched/per-sample parity check")
            md = (out.detach() - ref).abs().max().item()
            if not (md <= 5e-3):  # `not <=` also trips on NaN md (NaN>5e-3 is False, would silently pass)
                raise RuntimeError(f"[internvl_t2i C] batched vs per-sample reward max diff {md:.4g} > 5e-3")
            self._batch_parity_ok = True
        return out  # [B], autograd graph intact

    def _scores_batched(self, images01: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        """One right-padded InternVL forward over the whole batch (see _scores; keeps grad)."""
        B = images01.shape[0]
        pixel_values = torch.cat([self._preprocess_one(images01[i:i + 1]) for i in range(B)], dim=0)  # [B,3,448,448]
        vit_embeds = self.model.extract_feature(pixel_values)                    # [B, T, C] (one tile/image)
        T, C = vit_embeds.shape[1], vit_embeds.shape[-1]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        seqs, read_pos = [], []
        for i in range(B):
            ids = self.tokenizer(self._build_prompt(prompts[i]), return_tensors="pt")["input_ids"][0].to(self.device)
            plen = ids.shape[0]
            aids = self.tokenizer(f"{TEXTIMG_POINTWISE_DIM_NAME}：{PREFILL_SCORE}", return_tensors="pt",
                                  add_special_tokens=False)["input_ids"][0].to(self.device)
            rp = [j + plen - 1 for j in range(aids.shape[0]) if aids[j].item() == self.prefill_token_id]
            assert rp, "prefill digit not found in the teacher-forced answer"
            seqs.append(torch.cat([ids, aids])); read_pos.append(rp[0])
        Lmax = max(s.shape[0] for s in seqs)
        input_ids = torch.full((B, Lmax), pad_id, dtype=torch.long, device=self.device)
        attn = torch.zeros((B, Lmax), dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            input_ids[i, :s.shape[0]] = s; attn[i, :s.shape[0]] = 1  # RIGHT-pad: read positions stay pre-pad
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)  # [B, Lmax, C]
        selected = input_ids.reshape(-1) == self.img_context_token_id
        assert int(selected.sum().item()) == B * T, f"IMG_CONTEXT {int(selected.sum())} != {B * T}"
        flat = input_embeds.reshape(B * Lmax, C).clone()
        flat[selected] = vit_embeds.reshape(B * T, C).to(flat.dtype)  # sample-order aligned
        input_embeds = flat.reshape(B, Lmax, C).to(self.dtype)
        out = self.model.language_model(inputs_embeds=input_embeds, attention_mask=attn)
        logits = out[0] if not hasattr(out, "logits") else out.logits  # [B, Lmax, vocab]
        scores = []
        for i in range(B):
            p = torch.softmax(logits[i, read_pos[i], :][self.score_token_ids].float(), dim=-1)
            scores.append((p * self.values).sum() / 5.0 if self.readout == "expected" else p[self.select_idx].sum())
        return torch.stack(scores)  # [B]

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        return self._scores(images, prompts).float()


# --- process-level singleton -------------------------------------------------------------
# The 26B reward weighs ~52 GB on GPU. Training would otherwise instantiate it THREE times per
# rank (reward_fn + eval_reward_fn in multi_score, plus the OPA-ascent ri_scorer) -> ~156 GB ->
# OOM. All three want the SAME frozen model, so we hand out one shared instance per (device,
# checkpoint). Safe: params are frozen and scoring is functional (no per-call state); the no-grad
# __call__ and the grad _scores use the same weights.
_SCORER_CACHE: dict = {}


def get_internvl_t2i_scorer(device: str = "cuda", checkpoint_path: str = DEFAULT_CHECKPOINT,
                            **kwargs) -> "InternVLT2IPointScorer":
    key = (str(device), checkpoint_path)
    scorer = _SCORER_CACHE.get(key)
    if scorer is None:
        scorer = InternVLT2IPointScorer(device=device, checkpoint_path=checkpoint_path, **kwargs)
        _SCORER_CACHE[key] = scorer
    return scorer
