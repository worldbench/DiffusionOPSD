# SPDX-License-Identifier: Apache-2.0
"""Cross-reward evaluation harness (eval-only).

Given a trained SD3.5 LoRA checkpoint that was optimized for a single reward X, this:
  1. builds the base SD3.5 pipeline for the training config and loads the LoRA,
  2. regenerates a fixed held-out prompt set ONCE with the SAME sampler the training
     scripts use at eval time, and
  3. scores that one image set with every reward model loadable in the current env,
     writing {reward_name: mean_score}.

Running it over all trained checkpoints yields a train-X x eval-Y matrix.

Faithfulness: the pipeline build, LoRA load, `compute_text_embeddings`, and the
`pipeline_with_logprob` generation call are copied verbatim from `eval_fn` in
`scripts/train_nft_sd3.py` / `scripts/train_opsd_ri_sd3.py` so the numbers are directly
comparable to the training-time `eval_reward_<r>` curve. Both eval_fns hardcode
`solver="flow"`, `deterministic=True`, `num_inference_steps=config.sample.eval_num_steps`
(=40) and `guidance_scale=config.sample.guidance_scale` (=1.0, CFG off) -- so does this
harness (overridable via CLI). NOTE: config.sample.solver is "dpm2" but that governs the
TRAINING rollout only; the eval path uses flow. We match the eval path.

Not runnable on CPU: it needs a GPU plus the selected base-model and reward weights.
See the evaluation commands in README.md.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
from torch.cuda.amp import autocast as torch_autocast

import diffusionopsd.rewards
from diffusionopsd.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from diffusionopsd.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt

# Self-contained public scorers used by the default pass. ImageReward and the heavyweight
# public scorers can be requested explicitly. The paper's AltCLIP evaluator is internal and
# requires ALTCLIP_MODEL_PATH, so it must never be selected implicitly.
DEFAULT_Y_REWARDS: Tuple[str, ...] = ("hpsv2", "clipscore", "aesthetic", "pickscore")

# Sentinel used by reward functions for failed items.  Table evaluation rejects it
# fail-closed instead of silently reducing the sample count.
FAILED_SENTINEL = -10.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-reward eval for a trained SD3.5 LoRA checkpoint.")
    p.add_argument("--ckpt", default="", help="Saved LoRA dir. Empty evaluates the base checkpoint directly.")
    p.add_argument("--config", required=True, help="Config function name, e.g. sd3_clipscore_opa.")
    p.add_argument("--config_file", default="config/nft.py", help="Config module file (file:function convention).")
    p.add_argument("--out", required=True, help="Output JSON path.")
    p.add_argument("--images_dir", default="", help="Optional directory for generated PNGs and prompts.jsonl.")
    p.add_argument("--model", default=os.environ.get("MODEL_PATH", ""),
                   help="Base SD3.5 pipeline path. Defaults to $MODEL_PATH, else config.pretrained.model.")
    p.add_argument("--prompts", default="", help="Held-out prompts .txt (one per line). Default: <config.dataset>/test.txt.")
    p.add_argument("--prompt_set_name", default="", help="Human-readable prompt-set tag recorded in the JSON, e.g. pickscore_test or drawbench_test.")
    p.add_argument("--protocol_name", default="", help="Evaluation protocol tag recorded in the JSON, e.g. current_flow40 or table1_drawbench_flow40.")
    p.add_argument("--rewards", default="", help="Comma-separated Y scorers. Default: the standard SD3-env set.")
    p.add_argument("--num_steps", type=int, default=40, help="Eval sampler steps (training eval uses 40).")
    p.add_argument("--sampler", default="flow", choices=["flow", "dpm2", "dpm1", "ddim", "dance"],
                   help="Sampler. Training eval_fn hardcodes 'flow' (deterministic); default matches it.")
    p.add_argument("--guidance_scale", type=float, default=-1.0, help="CFG scale. <0 => use config.sample.guidance_scale.")
    p.add_argument("--n_prompts", type=int, default=0, help="Use the first N held-out prompts (0 => all).")
    p.add_argument("--seed", type=int, default=42, help="Fixed seed for the initial latents (reproducible across ckpts).")
    p.add_argument("--batch_size", type=int, default=16, help="Per-batch prompt count for generation & scoring.")
    p.add_argument(
        "--score_batch_size",
        type=int,
        default=0,
        help="Scoring micro-batch. <=0 reuses --batch_size. This lets generation stay batched "
             "while memory-heavy reward models are scored one image at a time.",
    )
    p.add_argument("--pickscore_scale", default="training", choices=["training", "raw"],
                   help="PickScore output scale. training = repo scorer output (/26); raw = multiply by 26 for Table-1 scale.")
    p.add_argument("--max_sequence_length", type=int, default=128, help="Text-encoder max length (training uses 128).")
    p.add_argument("--mixed_precision", default="fp16", choices=["fp16", "bf16", "no"],
                   help="Autocast dtype for generation (training default fp16).")
    p.add_argument("--dump_per_image", action="store_true",
                   help="Also dump per-image/per-prompt scores (index+prompt+<reward>) into the "
                        "output JSON under `per_image` for paired-diff / paired-CI analysis. "
                        "Default off; aggregate fields are unchanged.")
    return p.parse_args()


def load_config(config_file: str, config_name: str) -> Any:
    """Load config/nft.py by file path and call get_config(name) -- mirrors absl config_flags."""
    spec = importlib.util.spec_from_file_location("_dopsd_cross_eval_config", config_file)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load config file: {config_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config(config_name)


def compute_text_embeddings(
    prompt: List[str], text_encoders: List[Any], tokenizers: List[Any], max_sequence_length: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Copied verbatim from train_nft_sd3.py:compute_text_embeddings."""
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def build_pipeline(
    model_path: str, lora_path: str, device: torch.device, te_dtype: torch.dtype
) -> Tuple[StableDiffusion3Pipeline, List[Any], List[Any]]:
    """Mirror the pipeline build + LoRA load in train_nft_sd3.py main() (eval-relevant subset)."""
    pipeline = StableDiffusion3Pipeline.from_pretrained(model_path)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)
    pipeline.safety_checker = None

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    pipeline.vae.to(device, dtype=torch.float32)  # VAE fp32 (matches training)
    pipeline.text_encoder.to(device, dtype=te_dtype)
    pipeline.text_encoder_2.to(device, dtype=te_dtype)
    pipeline.text_encoder_3.to(device, dtype=te_dtype)

    transformer = pipeline.transformer.to(device)
    if lora_path:
        # Load the trained LoRA exactly as the training resume path does (config.train.lora_path).
        transformer = PeftModel.from_pretrained(transformer, lora_path)
        try:
            transformer.set_adapter("default")
        except (ValueError, KeyError):
            adapters = list(getattr(transformer, "peft_config", {}).keys())
            if adapters:
                transformer.set_adapter(adapters[0])
    pipeline.transformer = transformer
    pipeline.transformer.eval()
    pipeline.set_progress_bar_config(disable=True)
    return pipeline, text_encoders, tokenizers


def load_prompt_records(path: str, n_prompts: int) -> Tuple[List[str], List[Dict[str, Any]]]:
    with open(path, "r") as f:
        rows = [line.strip() for line in f.readlines() if line.strip()]
    if n_prompts and n_prompts > 0:
        rows = rows[:n_prompts]  # deterministic subset (first N) so every ckpt uses the same set
    prompts: List[str] = []
    metadata: List[Dict[str, Any]] = []
    for line in rows:
        if "\t" in line:
            prompt, ref_path = line.split("\t", 1)
            prompts.append(prompt)
            metadata.append({"ref_path": ref_path})
        else:
            prompts.append(line)
            metadata.append({})
    return prompts, metadata


def load_prompts(path: str, n_prompts: int) -> List[str]:
    prompts, _ = load_prompt_records(path, n_prompts)
    return prompts


def generate_images(
    pipeline: StableDiffusion3Pipeline,
    text_encoders: List[Any],
    tokenizers: List[Any],
    prompts: List[str],
    config: Any,
    args: argparse.Namespace,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    """Generate the held-out set ONCE, mirroring eval_fn's pipeline_with_logprob call.

    Returns a CPU float tensor of shape (N, 3, H, W) in [0, 1] (same format eval_fn feeds
    to the reward fn). Batching keeps GPU memory bounded; latents are seeded per batch for
    reproducibility across checkpoints.
    """
    resolution = int(config.resolution)
    guidance_scale = args.guidance_scale if args.guidance_scale >= 0 else float(config.sample.guidance_scale)
    noise_level = float(config.sample.noise_level)
    bs = args.batch_size

    # Empty-prompt negative embeds (unused when guidance_scale<=1, but mirrors eval_fn).
    neg_embed, neg_pooled = compute_text_embeddings([""], text_encoders, tokenizers, args.max_sequence_length, device)

    images_cpu: List[torch.Tensor] = []
    for b_idx, start in enumerate(range(0, len(prompts), bs)):
        batch_prompts = prompts[start:start + bs]
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            batch_prompts, text_encoders, tokenizers, args.max_sequence_length, device
        )
        cur = len(batch_prompts)
        cur_neg = neg_embed.repeat(cur, 1, 1)
        cur_neg_pooled = neg_pooled.repeat(cur, 1)
        # Per-batch generator -> initial latents are reproducible and independent of prior RNG use.
        generator = torch.Generator(device=device).manual_seed(args.seed + b_idx)

        with torch_autocast(enabled=autocast_dtype is not None, dtype=autocast_dtype or torch.float16):
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=cur_neg,
                    negative_pooled_prompt_embeds=cur_neg_pooled,
                    num_inference_steps=args.num_steps,
                    guidance_scale=guidance_scale,
                    output_type="pt",
                    height=resolution,
                    width=resolution,
                    noise_level=noise_level,
                    deterministic=True,
                    solver=args.sampler,
                    model_type="sd3",
                    generator=generator,
                )
        images_cpu.append(images.detach().float().cpu())
    return torch.cat(images_cpu, dim=0)


def score_all(
    images_cpu: torch.Tensor,
    prompts: List[str],
    reward_names: List[str],
    device: torch.device,
    batch_size: int,
    pickscore_scale: str = "training",
    per_image_out: Optional[Dict[str, List[Optional[float]]]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Score the SAME image set with each Y reward, loading one scorer at a time and freeing it.

    Reuses diffusionopsd.rewards.multi_score (the exact factory the training eval uses).  For
    Table-1 reporting, PickScore can be converted back to the raw scale because the repo
    scorer returns the training-normalized value (raw / 26).

    If `per_image_out` is provided (a dict), it is populated in place with
    `{reward_name: [score per image, aligned to prompt order]}`. Failed, sentinel,
    non-finite, or wrong-length outputs abort the whole evaluation. The return
    signature is unchanged (results, stats) for existing callers.
    """
    n_imgs = int(images_cpu.shape[0])
    if n_imgs <= 0:
        raise ValueError("cannot score an empty image set")
    if len(prompts) != n_imgs:
        raise ValueError(f"image/prompt count mismatch: {n_imgs} images, {len(prompts)} prompts")
    if batch_size <= 0:
        raise ValueError(f"score batch_size must be positive, got {batch_size}")
    metadata = metadata if metadata is not None else [{} for _ in prompts]
    if len(metadata) != n_imgs:
        raise ValueError(f"image/metadata count mismatch: {n_imgs} images, {len(metadata)} rows")
    results: Dict[str, float] = {}
    stats: Dict[str, Dict[str, float]] = {}
    for name in reward_names:
        t0 = time.time()
        scoring_fn = None
        # Keep one scorer resident at a time, but fail the whole evaluation if any
        # requested scorer fails.  A NaN/partial matrix cell must never look like a
        # successfully completed experiment to the queue or shard merger.
        try:
            scoring_fn = diffusionopsd.rewards.multi_score(device, {name: 1.0})
            chunks: List[np.ndarray] = []
            for start in range(0, images_cpu.shape[0], batch_size):
                b_imgs = images_cpu[start:start + batch_size].to(device)
                b_prompts = prompts[start:start + batch_size]
                b_meta = metadata[start:start + batch_size]
                with torch.no_grad():
                    details, _ = scoring_fn(b_imgs, b_prompts, b_meta, only_strict=True)
                vals = details[name]
                if isinstance(vals, torch.Tensor):
                    vals = vals.detach().float().cpu().numpy()
                else:
                    vals = np.asarray(vals, dtype=np.float32)
                chunks.append(vals.reshape(-1))
            all_vals = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
            if all_vals.shape != (n_imgs,):
                raise RuntimeError(
                    f"scorer returned {all_vals.shape[0]} values for {n_imgs} images"
                )
            invalid = (~np.isfinite(all_vals)) | (all_vals == FAILED_SENTINEL)
            if np.any(invalid):
                bad = np.flatnonzero(invalid)[:8].tolist()
                raise RuntimeError(
                    f"scorer returned {int(invalid.sum())} invalid/sentinel values; "
                    f"first indices={bad}"
                )
            kept = all_vals.astype(np.float64)
            if name == "pickscore" and pickscore_scale == "raw":
                kept = kept * 26.0
            if per_image_out is not None:
                # Full per-image column aligned to prompt order for paired comparisons.
                # Invalid values have already caused a fail-closed abort above.
                scale = 26.0 if (name == "pickscore" and pickscore_scale == "raw") else 1.0
                col: List[Optional[float]] = [float(value) * scale for value in all_vals]
                per_image_out[name] = col
            if kept.size:
                mean = float(np.mean(kept))
                std = float(np.std(kept, ddof=1)) if kept.size > 1 else 0.0
                se = float(std / np.sqrt(kept.size)) if kept.size > 0 else float("nan")
            else:
                mean = std = se = float("nan")
            results[name] = mean
            stats[name] = {"mean": mean, "std": std, "se": se, "n": int(kept.size)}
            scale_note = " raw" if (name == "pickscore" and pickscore_scale == "raw") else ""
            print(f"[cross_eval] {name}{scale_note}: mean={mean:.4f} (n={kept.size}, {time.time() - t0:.1f}s)", flush=True)
        except Exception as exc:
            print(f"[cross_eval] scorer '{name}' FAILED: {exc}", flush=True)
            raise RuntimeError(f"requested scorer '{name}' failed; evaluation aborted") from exc
        finally:
            del scoring_fn
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results, stats


def atomic_json_dump(record: Dict[str, Any], path: str) -> None:
    """Publish a JSON artifact only after a complete write succeeds."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    partial = f"{path}.{os.getpid()}.partial"
    try:
        with open(partial, "w") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.unlink(partial)


def save_generated_images(
    images_cpu: torch.Tensor, prompts: List[str], images_dir: str, *, global_start_idx: int = 0
) -> None:
    """Save a deterministic visualization artifact tree owned by the experiment launcher."""
    if not images_dir:
        return
    from PIL import Image

    os.makedirs(images_dir, exist_ok=True)
    with open(os.path.join(images_dir, "prompts.jsonl"), "w") as handle:
        for idx, (image, prompt) in enumerate(zip(images_cpu, prompts)):
            array = (image.clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
            filename = f"{idx:04d}.png"
            Image.fromarray(array).save(os.path.join(images_dir, filename))
            handle.write(json.dumps({"index": global_start_idx + idx, "file": filename, "prompt": prompt}, ensure_ascii=False) + "\n")

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config_file, args.config)
    model_path = args.model or config.pretrained.model
    prompts_path = args.prompts or os.path.join(config.dataset, "test.txt")
    reward_names = [r.strip() for r in args.rewards.split(",") if r.strip()] or list(DEFAULT_Y_REWARDS)
    train_reward = ",".join(config.reward_fn.keys())

    autocast_dtype: Optional[torch.dtype] = None
    if args.mixed_precision == "fp16":
        autocast_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        autocast_dtype = torch.bfloat16
    te_dtype = autocast_dtype if autocast_dtype is not None else torch.float32

    if args.ckpt and not os.path.exists(os.path.join(args.ckpt, "adapter_config.json")):
        raise FileNotFoundError(
            f"No adapter_config.json under --ckpt {args.ckpt}; expected a saved LoRA dir "
            "(.../checkpoints/checkpoint-<step>/lora)."
        )

    prompts, metadata = load_prompt_records(prompts_path, args.n_prompts)
    print(
        f"[cross_eval] config={args.config} train_reward={train_reward} ckpt={args.ckpt}\n"
        f"[cross_eval] model={model_path} prompts={prompts_path} n={len(prompts)} "
        f"sampler={args.sampler} steps={args.num_steps} rewards={reward_names}",
        flush=True,
    )

    pipeline, text_encoders, tokenizers = build_pipeline(model_path, args.ckpt, device, te_dtype)

    t_gen = time.time()
    images_cpu = generate_images(pipeline, text_encoders, tokenizers, prompts, config, args, device, autocast_dtype)
    save_generated_images(images_cpu, prompts, args.images_dir)
    print(f"[cross_eval] generated {images_cpu.shape[0]} images in {time.time() - t_gen:.1f}s", flush=True)

    # Free the diffusion stack before loading reward scorers to keep peak GPU memory low.
    del pipeline, text_encoders, tokenizers
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    per_image_out: Optional[Dict[str, List[Optional[float]]]] = {} if args.dump_per_image else None
    score_batch_size = args.score_batch_size if args.score_batch_size > 0 else args.batch_size
    scores, score_stats = score_all(
        images_cpu, prompts, reward_names, device, score_batch_size, args.pickscore_scale,
        per_image_out=per_image_out, metadata=metadata,
    )

    guidance_scale = args.guidance_scale if args.guidance_scale >= 0 else float(config.sample.guidance_scale)
    record = {
        "ckpt": args.ckpt,
        "checkpoint_protocol": "lora" if args.ckpt else "base_model",
        "images_dir": args.images_dir,
        "config": args.config,
        "train_reward": train_reward,
        "train_reward_fn": {k: float(v) for k, v in dict(config.reward_fn).items()},
        "eval_protocol": args.protocol_name,
        "prompt_set": args.prompt_set_name,
        "prompts_path": prompts_path,
        "num_prompts": len(prompts),
        "sampler": args.sampler,
        "num_steps": args.num_steps,
        "guidance_scale": guidance_scale,
        "resolution": int(config.resolution),
        "noise_level": float(config.sample.noise_level),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "score_batch_size": score_batch_size,
        "pickscore_scale": args.pickscore_scale,
        "model": model_path,
        "scores": scores,
        "score_stats": score_stats,
    }
    if per_image_out is not None:
        # Per-image/per-prompt scores aligned to prompt order (same fixed seed -> same initial
        # latents per index across variants, so two variants are pairable by `index`). One row per
        # image: {index, prompt, <reward>: score_or_null}. Aggregate fields above are unchanged.
        per_image_rows: List[Dict[str, Any]] = []
        for idx, prompt in enumerate(prompts):
            row: Dict[str, Any] = {"index": idx, "prompt": prompt}
            for name in reward_names:
                col = per_image_out.get(name)
                row[name] = col[idx] if (col is not None and idx < len(col)) else None
            per_image_rows.append(row)
        record["per_image"] = per_image_rows
    atomic_json_dump(record, args.out)
    print(f"[cross_eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
