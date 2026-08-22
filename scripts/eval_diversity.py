#!/usr/bin/env python3
"""Matched-prompt reward + diversity eval for DiffusionOPSD checkpoints.

Settles the decisive question the confounded rollout snapshots cannot: at matched held-out
prompts, does a checkpoint keep DIVERSITY (varied images for the same prompt = no mode
collapse) or has reward optimization concentrated it? Reports, per checkpoint:
  - hps_mean            : HPSv2.1 over all generated images (matched prompt set)
  - intra_prompt_div    : 1 - mean cosine similarity of open_clip image features WITHIN a prompt,
                          averaged over prompts. Low => same image regardless of init noise
                          => mode collapse. This is the mode-collapse detector.
  - inter_prompt_div    : same across prompt-mean features (spread across prompts).

Same fixed prompt slice + same sampler (dpm2/10/guidance=1.0, matching training rollout) across
all checkpoints -> a fair (reward, diversity) point per model. No fixed seed is used:
the n_per images per prompt use independent random init noise; diversity across them is exactly
the signal we want. One process per GPU; --lora none evaluates the base model.
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
from diffusionopsd.hpsv2_scorer import HPSv2Scorer
from diffusionopsd.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob


def decode01(pipeline, x_latent):
    lat = (x_latent / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    img = pipeline.vae.decode(lat.to(pipeline.vae.dtype), return_dict=False)[0]
    return (img / 2 + 0.5).clamp(0, 1).float()


@torch.no_grad()
def clip_features(scorer, images01):
    """Normalized open_clip image features (HPS-tuned ViT-H) for a batch of [0,1] NCHW images."""
    img = scorer.preprocess_val(images01.to(scorer.dtype).to(scorer.device))
    feats = scorer.model.encode_image(img)
    return torch.nn.functional.normalize(feats.float(), dim=-1)


@torch.no_grad()
def hps(scorer, images01, prompts):
    return scorer(images01, list(prompts)).float()


def to_pil(img01):
    arr = (img01.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", type=str, default="none", help="path to checkpoint/lora dir, or 'none'")
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_prompts", type=int, default=24)
    ap.add_argument("--n_per", type=int, default=6, help="images per prompt (diversity sample)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--prompt_file", type=str, required=True)
    ap.add_argument("--montage_rows", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES pins the physical GPU
    pipeline = StableDiffusion3Pipeline.from_pretrained(os.environ["MODEL_PATH"])
    pipeline.vae.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)
    for te in (pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3):
        te.requires_grad_(False)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(disable=True)

    if args.lora != "none":
        # checkpoint/lora holds the trained "default" adapter at top level (+ an "old/" subdir).
        pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, args.lora)
        pipeline.transformer.set_adapter("default")

    pipeline.vae.to(device, dtype=torch.float32).eval()
    pipeline.text_encoder.to(device, dtype=torch.bfloat16)
    pipeline.text_encoder_2.to(device, dtype=torch.bfloat16)
    pipeline.text_encoder_3.to(device, dtype=torch.bfloat16)
    pipeline.transformer.to(device).eval()

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)

    with open(args.prompt_file) as f:
        all_prompts = [l.strip() for l in f if l.strip()]
    prompts = all_prompts[: args.n_prompts]

    hps_all, intra_divs = [], []
    montage_rows = []
    for pi, prompt in enumerate(prompts):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, all_latents, _ = pipeline_with_logprob(
                pipeline, prompt=[prompt], num_images_per_prompt=args.n_per,
                num_inference_steps=args.steps, guidance_scale=1.0, output_type="pt",
                height=512, width=512, noise_level=0.7, deterministic=True, solver="dpm2",
                model_type="sd3",
            )
        x_end = all_latents[-1].detach().float()             # (n_per, C, H, W)
        imgs = decode01(pipeline, x_end)
        h = hps(scorer, imgs, [prompt] * args.n_per)
        feats = clip_features(scorer, imgs)                  # (n_per, D) normalized
        sim = feats @ feats.T                                # cosine sims
        n = feats.shape[0]
        off = (sim.sum() - torch.diagonal(sim).sum()) / (n * (n - 1))
        intra_divs.append(float(1.0 - off))
        hps_all.append(h.cpu())
        if pi < args.montage_rows:
            montage_rows.append((prompt, imgs[: min(args.n_per, 6)].cpu(),
                                 feats.mean(0)))             # prompt-mean feat for inter-div

    hps_cat = torch.cat(hps_all)
    # inter-prompt diversity: spread of prompt-mean features
    pmeans = torch.nn.functional.normalize(torch.stack([m for _, _, m in montage_rows]), dim=-1)
    isim = pmeans @ pmeans.T
    m = pmeans.shape[0]
    inter = float(1.0 - (isim.sum() - torch.diagonal(isim).sum()) / (m * (m - 1))) if m > 1 else 0.0

    stats = {
        "tag": args.tag, "lora": args.lora,
        "n_prompts": len(prompts), "n_per": args.n_per, "steps": args.steps,
        "hps_mean": float(hps_cat.mean()), "hps_std": float(hps_cat.std()),
        "intra_prompt_div": float(np.mean(intra_divs)),
        "intra_prompt_div_std": float(np.std(intra_divs)),
        "inter_prompt_div": inter,
    }
    with open(os.path.join(args.out_dir, f"{args.tag}.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # montage: rows = prompts, cols = n_per images (visual mode-collapse check)
    if montage_rows:
        cell = 200
        cols = min(args.n_per, 6)
        M = Image.new("RGB", (cell * cols, cell * len(montage_rows)), (15, 15, 15))
        for r, (_, imgs, _) in enumerate(montage_rows):
            for c in range(min(cols, imgs.shape[0])):
                M.paste(to_pil(imgs[c]).resize((cell, cell)), (c * cell, r * cell))
        M.save(os.path.join(args.out_dir, f"{args.tag}_montage.png"))

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
