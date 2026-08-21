# SPDX-License-Identifier: Apache-2.0
"""Native-pipeline evaluation for non-SD3.5-Medium reference checkpoints.

This script generates from each model with its own **stock diffusers
pipeline and default sampler/steps/guidance** (i.e. the model's official inference
config) at the DiffusionNFT Table-1 annotated resolution, then scores the images with
the SAME reward stack as `cross_eval.py` (imported `score_all`) so the reward columns
are directly comparable to the SD3.5-M rows.

All generation settings are recorded in the output JSON. These rows are reference/context
only; the paper's table remains authoritative for its reported values.

Z-Image-Turbo (`--pipeline zimage`) is the step-distilled base row: its native inference
config is FlowMatchEuler / 9 steps / 1024 / guidance=0 (single forward, no CFG), bf16-native.
Its generation reuses `zimage_rollout` (the exact path in train_nft_zimage.py:eval_fn) rather
than a stock `__call__`, so the base row is bit-for-bit comparable to the trained Z-Image rows.
Z-Image evaluation requires a Diffusers build that provides ``ZImagePipeline``.

Not runnable on CPU. Offline (models pre-downloaded to $MODEL_ROOT). One reward scorer
is loaded at a time (peak memory = max(pipeline, largest scorer)); the pipeline is freed
before scoring.
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from typing import Any, List, Optional, Tuple

import torch

# Reuse the exact scoring / IO path from the SD3 harness so numbers are comparable.
import cross_eval

# Per-model official-default inference config. Resolution follows the DiffusionNFT
# Table-1 annotation (SD-XL / SD3.5-L at 1024; FLUX at 512). steps/guidance are the
# diffusers pipeline defaults for each model (overridable via CLI).
PIPELINE_SPECS = {
    # FLUX.1-dev is bf16-native: loading it in fp16 overflows -> black/NaN images. SD-XL /
    # SD3.5 are fp16-safe (our SD3.5-M harness uses fp16 and reproduced the paper).
    "sdxl": {"resolution": 1024, "num_steps": 50, "guidance": 5.0, "batch_size": 8, "dtype": "fp16"},
    "sd3": {"resolution": 1024, "num_steps": 40, "guidance": 4.5, "batch_size": 8, "dtype": "fp16"},
    "flux": {"resolution": 512, "num_steps": 50, "guidance": 3.5, "batch_size": 2, "dtype": "bf16"},
    # Z-Image-Turbo: step-distilled 6B S3-DiT. Native regime = FlowMatchEuler / 9 steps / 1024 /
    # guidance=0 (single forward, NO CFG / negative prompt). bf16-native like FLUX: fp16 overflows
    # the 6B DiT -> NaN from step 0 (see MEMORY diffusionopsd-zimage-bf16-fix). Needs the ISO env
    # (diffusers-from-source >=0.36 provides ZImagePipeline). Generation mirrors
    # train_nft_zimage.py:eval_fn (zimage_rollout) so the base row is directly comparable to the
    # trained Z-Image rows in the main table.
    "zimage": {"resolution": 1024, "num_steps": 9, "guidance": 0.0, "batch_size": 8, "dtype": "bf16"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Native-pipeline reference eval (SD-XL / SD3.5-L / FLUX / Z-Image-Turbo).")
    p.add_argument("--pipeline", required=True, choices=list(PIPELINE_SPECS), help="Which native pipeline to build.")
    p.add_argument("--model", required=True, help="Local diffusers model dir (offline).")
    p.add_argument("--lora", default="",
                   help="Trained PEFT LoRA dir (…/checkpoints/checkpoint-N/lora); zimage ONLY. When set, "
                        "it is loaded onto the S3-DiT AFTER the base build (mirrors train_nft_zimage.py "
                        "warm-start), so the row is scored exactly like the base row with only the LoRA "
                        "delta added. Unset => base model, path byte-identical to today.")
    p.add_argument("--out", required=True, help="Output result.json path.")
    p.add_argument("--images_dir", default="", help="Optional dir for generated PNGs + prompts.jsonl.")
    p.add_argument("--prompts", required=True, help="Prompt .txt (one per line).")
    p.add_argument("--prompt_set_name", default="drawbench", help="Prompt-set tag recorded in JSON.")
    p.add_argument("--protocol_name", default="", help="Protocol tag recorded in JSON.")
    p.add_argument("--rewards", default="pickscore,clipscore,hpsv2,aesthetic,imagereward",
                   help="Comma-separated pointwise reward scorers (no internvl_dual for single-checkpoint rows).")
    p.add_argument("--n_prompts", type=int, default=0, help="Number of prompts from --start_idx (0 => to end).")
    p.add_argument("--start_idx", type=int, default=0, help="Prompt offset — for data-parallel sharding across GPUs.")
    p.add_argument("--seed", type=int, default=42, help="Fixed per-batch latent seed.")
    p.add_argument("--resolution", type=int, default=-1, help="Override; <0 => pipeline default.")
    p.add_argument("--num_steps", type=int, default=-1, help="Override; <0 => pipeline default.")
    p.add_argument("--guidance_scale", type=float, default=-1.0, help="Override; <0 => pipeline default.")
    p.add_argument("--batch_size", type=int, default=-1, help="Override; <0 => pipeline default.")
    p.add_argument("--pickscore_scale", default="raw", choices=["training", "raw"], help="PickScore scale.")
    p.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16"],
                   help="Pipeline weight dtype. auto => per-pipeline default (FLUX=bf16, else fp16).")
    p.add_argument("--score_batch_size", type=int, default=16,
                   help="Scorer minibatch size (16 = historical default, keeps existing rows reproducible). "
                        "Lower it for the 26B InternVL scorers, which OOM at 16 on an 80 GB card.")
    p.add_argument("--per_image_json", default="",
                   help="Optional path for the per-image score dump {reward: [score|null per prompt]}, "
                        "index-aligned to the prompt shard. Written ALONGSIDE --out, which keeps its exact "
                        "schema, so rows stay byte-comparable to cells evaluated before this flag existed. "
                        "Enables paired (per-prompt) tests between two checkpoints instead of unpaired-only.")
    return p.parse_args()


def build_pipeline(kind: str, model_path: str, dtype: torch.dtype, device: torch.device, lora_path: str = "") -> Any:
    """Load the stock diffusers pipeline (default scheduler) for the model family.

    If `lora_path` is set (zimage only), a trained PEFT LoRA is loaded onto the S3-DiT
    transformer after the base build, mirroring train_nft_zimage.py's warm-start so the row
    differs from the base row ONLY by the LoRA delta.
    """
    if kind == "sdxl":
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=dtype, use_safetensors=True)
    elif kind == "sd3":
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=dtype)
    elif kind == "flux":
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype)
    elif kind == "zimage":
        # ZImagePipeline exists only in the ISO env (diffusers-from-source >=0.36). Load the 6B
        # S3-DiT + Qwen3 text encoder in bf16 (fp16 -> NaN); the FLUX VAE is upcast to fp32 below.
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        raise ValueError(f"Unknown pipeline kind: {kind}")
    pipe = pipe.to(device)
    if kind == "sdxl":
        # SD-XL's fp16 VAE overflows during decode -> NaN / crash. Run the VAE in fp32.
        pipe.upcast_vae()
    if kind == "zimage":
        # Z-Image's FLUX AutoencoderKL: keep the VAE in fp32 for decode stability (matches
        # train_nft_zimage.py, which loads vae in fp32 while the transformer stays bf16).
        pipe.vae.to(dtype=torch.float32)
    if kind == "zimage" and lora_path:
        # Trained-checkpoint path: load the PEFT LoRA onto the S3-DiT EXACTLY as the trainer's
        # warm-start does (train_nft_zimage.py:375-377 -- `PeftModel.from_pretrained(transformer,
        # config.train.lora_path); transformer.set_adapter("default")`; identical to the verified
        # SD3 cross_eval.py:130-138 load). save_ckpt (train_nft_zimage.py:255-274) writes the trained
        # adapter via PeftModel.save_pretrained to `checkpoint-<step>/lora`: PEFT stores the active
        # "default" adapter at the dir ROOT (adapter_config.json + adapter_model.safetensors) and the
        # auxiliary "old" adapter under ./old/, so loading the dir root loads precisely the trained
        # (EMA-copied) policy.
        # We do NOT reassign pipe.transformer: PeftModel injects the LoRA modules into the transformer
        # IN PLACE, so pipe.transformer stays the raw (now LoRA-injected) S3-DiT that zimage_rollout
        # expects -- byte-identical to how eval_fn sees pipeline.transformer during training-time eval.
        from peft import PeftModel
        peft_transformer = PeftModel.from_pretrained(pipe.transformer, lora_path)
        try:
            peft_transformer.set_adapter("default")
        except (ValueError, KeyError):
            adapters = list(getattr(peft_transformer, "peft_config", {}).keys())
            if adapters:
                peft_transformer.set_adapter(adapters[0])
        pipe.transformer.eval()
        pipe._dopsd_peft_transformer = peft_transformer  # hold the wrapper for the pipeline's lifetime
    pipe.set_progress_bar_config(disable=True)
    if getattr(pipe, "safety_checker", None) is not None:
        pipe.safety_checker = None
    return pipe


def generate(
    pipe: Any, kind: str, prompts: List[str], resolution: int, num_steps: int,
    guidance: float, batch_size: int, seed: int, device: torch.device,
    global_start_idx: int = 0,
) -> torch.Tensor:
    """Generate one image per prompt with canonical global per-batch seeds.

    Shards must start on a generation-batch boundary.  This preserves exactly the
    same batches and ``seed + global_batch_index`` sequence as a single-process
    run, rather than restarting the seed sequence independently on every GPU.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if global_start_idx < 0:
        raise ValueError(f"global_start_idx must be non-negative, got {global_start_idx}")
    if global_start_idx % batch_size:
        raise ValueError(
            f"shard start_idx={global_start_idx} is not aligned to generation "
            f"batch_size={batch_size}; use batch-aligned shard boundaries"
        )
    zimage_gen: Optional[Tuple[Any, Any]] = None
    if kind == "zimage":
        # Mirror train_nft_zimage.py:eval_fn EXACTLY: Qwen3 encode -> 9-step FlowMatchEuler rollout
        # (gs=0, no CFG) -> FLUX-VAE decode to [0,1]. Reused verbatim so the base row matches the
        # trained Z-Image rows' generation path. These symbols import only in the ISO env.
        from diffusionopsd.diffusers_patch.zimage_pipeline_with_rollout import (
            zimage_encode_prompt, zimage_rollout,
        )
        zimage_gen = (zimage_encode_prompt, zimage_rollout)

    images_cpu: List[torch.Tensor] = []
    for b_idx, start in enumerate(range(0, len(prompts), batch_size)):
        batch = prompts[start:start + batch_size]
        global_batch_idx = (global_start_idx + start) // batch_size
        generator = torch.Generator(device=device).manual_seed(seed + global_batch_idx)
        if kind == "zimage":
            zimage_encode_prompt, zimage_rollout = zimage_gen
            prompt_embeds_list = zimage_encode_prompt(pipe, batch, device, max_sequence_length=512)
            # bf16 autocast wraps the whole rollout+decode, exactly as eval_fn does.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    out = zimage_rollout(
                        pipe, prompt_embeds_list, num_inference_steps=num_steps,
                        height=resolution, width=resolution, device=device, generator=generator,
                        guidance_scale=guidance, decode=True,
                    )
            imgs = out["images"]  # (N,3,H,W) in [0,1]
            images_cpu.append(imgs.detach().float().cpu())
            continue
        kwargs = dict(
            prompt=batch, num_inference_steps=num_steps, guidance_scale=guidance,
            height=resolution, width=resolution, output_type="pt", generator=generator,
        )
        if kind == "flux":
            kwargs["max_sequence_length"] = 512  # FLUX T5 cap
        with torch.no_grad():
            out = pipe(**kwargs)
        imgs = out.images  # FloatTensor (N,3,H,W) in [0,1] for output_type="pt"
        images_cpu.append(imgs.detach().float().cpu())
    return torch.cat(images_cpu, dim=0)


def main() -> None:
    args = parse_args()
    if args.lora:
        # LoRA loading is wired for the Z-Image S3-DiT only (the SD-XL / SD3.5-L / FLUX rows are
        # base-model reference rows). Fail loud rather than silently ignoring the LoRA.
        if args.pipeline != "zimage":
            raise SystemExit(f"--lora is only supported for --pipeline zimage (got {args.pipeline}).")
        # Mirror cross_eval.py's guard: the dir must be a saved PEFT adapter (…/checkpoint-N/lora).
        if not os.path.exists(os.path.join(args.lora, "adapter_config.json")):
            raise FileNotFoundError(
                f"No adapter_config.json under --lora {args.lora}; expected a trained LoRA dir "
                "(.../checkpoints/checkpoint-<step>/lora).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = PIPELINE_SPECS[args.pipeline]
    resolution = args.resolution if args.resolution > 0 else spec["resolution"]
    num_steps = args.num_steps if args.num_steps > 0 else spec["num_steps"]
    guidance = args.guidance_scale if args.guidance_scale >= 0 else spec["guidance"]
    batch_size = args.batch_size if args.batch_size > 0 else spec["batch_size"]
    dtype_name = args.dtype if args.dtype != "auto" else spec["dtype"]
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    reward_names = [r.strip() for r in args.rewards.split(",") if r.strip()]

    # internvl_dual (pairwise) scores each generated image against a per-prompt reference carried as
    # "prompt<TAB>ref_path"; load_prompt_records keeps that metadata (plain lines -> empty dict). Every
    # other reward needs only the prompt strings, so keep the light load_prompts path unchanged there.
    _needs_meta = "internvl_dual" in reward_names
    if _needs_meta:
        all_prompts, all_meta = cross_eval.load_prompt_records(args.prompts, 0)
    else:
        all_prompts = cross_eval.load_prompts(args.prompts, 0)
        all_meta = None
    end = args.start_idx + args.n_prompts if args.n_prompts > 0 else len(all_prompts)
    prompts = all_prompts[args.start_idx:end]  # shard slice (start_idx..end) for data-parallel eval
    metadata = all_meta[args.start_idx:end] if all_meta is not None else None  # ref_path per prompt (internvl_dual)
    print(
        f"[native_eval] pipeline={args.pipeline} model={args.model} lora={args.lora or '-'} "
        f"shard[{args.start_idx}:{end}]\n"
        f"[native_eval] res={resolution} steps={num_steps} guidance={guidance} bs={batch_size} "
        f"n={len(prompts)} rewards={reward_names}",
        flush=True,
    )

    pipe = build_pipeline(args.pipeline, args.model, dtype, device, lora_path=args.lora)
    t_gen = time.time()
    images_cpu = generate(
        pipe, args.pipeline, prompts, resolution, num_steps, guidance, batch_size,
        args.seed, device, global_start_idx=args.start_idx,
    )
    cross_eval.save_generated_images(
        images_cpu, prompts, args.images_dir, global_start_idx=args.start_idx
    )
    print(f"[native_eval] generated {images_cpu.shape[0]} images in {time.time() - t_gen:.1f}s", flush=True)

    # Free the diffusion stack before loading reward scorers (peak memory control).
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    per_image: Optional[dict] = {} if args.per_image_json else None
    scores, score_stats = cross_eval.score_all(
        images_cpu, prompts, reward_names, device, batch_size=args.score_batch_size,
        pickscore_scale=args.pickscore_scale,
        metadata=metadata,  # None => score_all defaults to empty dicts (byte-identical to before)
        per_image_out=per_image,  # None => score_all skips the per-image column (unchanged path)
    )

    record = {
        "pipeline": args.pipeline,
        "model": args.model,
        "checkpoint_protocol": "zimage_lora" if args.lora else "native_base_model",
        "images_dir": args.images_dir,
        "eval_protocol": args.protocol_name,
        "prompt_set": args.prompt_set_name,
        "prompts_path": args.prompts,
        "num_prompts": len(prompts),
        "start_idx": args.start_idx,
        "end_idx": args.start_idx + len(prompts),
        "resolution": resolution,
        "num_steps": num_steps,
        "guidance_scale": guidance,
        "sampler": "flow_euler_9step" if args.pipeline == "zimage" else "pipeline_default_scheduler",
        "seed": args.seed,
        "seed_scheme": "canonical_per_batch_seed_plus_batch_index",
        "generation_batch_size": batch_size,
        "dtype": dtype_name,
        "pickscore_scale": args.pickscore_scale,
        "scores": scores,
        "score_stats": score_stats,
    }
    if args.lora:
        # Add the LoRA path ONLY for trained rows so the base (no-LoRA) JSON stays byte-identical.
        record["lora"] = args.lora
    if args.score_batch_size != 16:
        # Only recorded when it deviates from the historical 16, so rows evaluated before this flag
        # existed stay schema-identical while a non-default (26B OOM workaround) stays auditable.
        record["score_batch_size"] = args.score_batch_size
    cross_eval.atomic_json_dump(record, args.out)
    print(f"[native_eval] wrote {args.out}", flush=True)
    if per_image is not None:
        cross_eval.atomic_json_dump(
            {"prompts": prompts, "start_idx": args.start_idx, "per_image": per_image},
            args.per_image_json,
        )
        print(f"[native_eval] wrote {args.per_image_json}", flush=True)


if __name__ == "__main__":
    main()
