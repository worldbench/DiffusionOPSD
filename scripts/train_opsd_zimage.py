# SPDX-License-Identifier: Apache-2.0
#
# DiffusionOPSD / OPA-X0-NFT on Z-Image-Turbo.
#
# Ports the winning SD3 OPA method (scripts/train_opsd_ri_sd3.py, opa=1) to the step-distilled
# Z-Image-Turbo sampler. The OPA MATH IS IDENTICAL; only the sampler/pipeline/scheduler/sign change.
#
#   OPA (no-cert fast path, opa_cert=0):
#     1. Old-LoRA rollout (FlowMatchEuler, 9 steps, gs=0) stores the low-noise query state z_q
#        at sigma_q = argmin|sigma - 0.27| (~=0.273, step 8), read from scheduler.sigmas.
#     2. x0 anchor:      y0 = z_q - sigma_q * v_old        (v_old = -raw transformer out)
#     3. Reward-gradient ascent/descent at y0 -> y+ / y- via differentiable VAE decoding
#     4. Train NFT implicit branches on the REAL rollout state z_q (strictly on-policy):
#            y_theta = z_q - sigma_q * v_theta
#            y_pos = z_q - sigma_q*v_pos  (-> y+) ,  y_neg = z_q - sigma_q*v_neg  (-> y-)
#        group-weighted with adaptive normalizers, followed by the behavior-policy EMA.
#
# SIGN: zimage_v returns the diffusers-convention velocity (= -raw), so y0/y_theta = z - sigma*v
# is correct. The dpm2 exact-suffix certification of SD3 is NOT ported (Z-Image uses FlowMatchEuler,
# and opa_cert=0 skips it entirely -> no solver dependency). opa_cert=1 raises NotImplementedError.

from collections import defaultdict
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
import logging
import numpy as np
import diffusionopsd.rewards
from diffusionopsd import profiling  # paper efficiency-profiling harness (env PROFILE=1)
from diffusionopsd.zimage_heavy_diff_bridge import zimage_heavy_diff_bridge_enabled  # ZIMAGE_HEAVY_DIFF_BRIDGE 1-GPU DIFFERENTIABLE heavy-reward server (gated; hpsv3/deqa)
# NOTE: get_image_transform (diffusionopsd.clip_scorer) is imported LAZILY inside the pickscore/aesthetic
# branches of _reward_scores_grad, NOT here, so an hpsv2 run's module-load import graph is unchanged
# (no new torchvision/transformers-CLIP dependency on the validated hpsv2 path).
from diffusionopsd.stat_tracking import PerPromptStatTracker, calculate_prompt_group_dispersion
from diffusionopsd.diffusers_patch.zimage_pipeline_with_rollout import (
    zimage_encode_prompt, zimage_rollout, zimage_v, zimage_decode, zimage_query_sigma_index,
)
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from diffusionopsd.ema import EMAModuleWrapper
from diffusionopsd.experiment_io import (
    resolve_resume_checkpoint, resume_position, restore_ema_and_rng,
    save_trainer_state, write_raw_reward_jsonl,
)
from diffusionopsd.metrics import install_wandb_jsonl_tee
from ml_collections import config_flags
from torch.cuda.amp import GradScaler, autocast as torch_autocast
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", "config/zimage.py:zimg_opsd_hpsv2", "Training configuration."
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# ZIMAGE_HEAVY_DIFF_BRIDGE: NCCL subgroup over policy ranks only (diff-reward server excluded). None
# => default world (bridge off), so passing group=POLICY_GROUP everywhere is byte-identical when off.
POLICY_GROUP = None

ZIMAGE_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"]
QWEN_MAX_SEQ_LEN = 512


# OPA differentiable-reward scope for Z-Image. These light rewards' frozen differentiable
# scorer fits alongside the 6B Z-Image DiT on one GPU, so the OPA target ascent can follow the
# TRAINING reward's gradient (not hardcoded HPS). HPSv3 and DeQA use the public differentiable
# reward bridge.
OPSD_ZIMAGE_DIFF_REWARDS = ("hpsv2", "clipscore", "pickscore", "aesthetic", "imagereward")


# ===================== OPA reward-improvement helpers (generic differentiable reward, Z-Image decode) ===================== #
# _reward_scores_grad / _load_reward_scorer / _OPA_TFORM_CACHE are ported VERBATIM from
# scripts/train_opsd_ri_sd3.py so every per-reward branch (hpsv2 included) is numerically identical
# to the SD3 trainer. The ONLY Z-Image change is the decode used by _reward_of_latents_grad
# (zimage_decode instead of SD3's _decode01). The reward gradient still touches ONLY the OPA target
# latent y0 (never the policy/sampler).
_OPA_TFORM_CACHE = {}


def _reward_scores_grad(scorer, kind, images01, prompts):
    """Differentiable reward score for images01 [B,3,H,W] in [0,1]."""
    if kind in ("open3", "multi_open3"):
        if not isinstance(scorer, dict):
            raise ValueError("Open3 composite OPA scorer must be a dict of sub-scorers.")
        total = None
        for sub_kind in ("pickscore", "clipscore", "hpsv2"):
            sub_scores = _reward_scores_grad(scorer[sub_kind], sub_kind, images01, prompts)
            total = sub_scores if total is None else total + sub_scores
        return total.float()
    profiling.reward_fwd_inc()  # §6: reward forward (OPA ascent / certification scoring)
    dev = scorer.device if scorer is not None else images01.device
    if kind == "hpsv2":
        image = scorer.preprocess_val(images01.to(scorer.dtype).to(dev))
        text = scorer.processor(prompts).to(dev)
        out = scorer.model(image, text)
        return torch.diagonal(out["image_features"] @ out["text_features"].T, 0).float()
    if kind == "clipscore":
        texts = scorer.processor(text=prompts, padding="max_length", truncation=True, return_tensors="pt").to(dev)
        pixels = scorer._process(images01).to(dev)
        out = scorer.model(pixel_values=pixels, **texts)
        return (out.logits_per_image.diagonal() / 100).float()
    if kind == "pickscore":
        if "pickscore" not in _OPA_TFORM_CACHE:
            from diffusionopsd.clip_scorer import get_image_transform  # lazy: keep torchvision/CLIP off the hpsv2 import path
            _OPA_TFORM_CACHE["pickscore"] = get_image_transform(scorer.processor.image_processor)
        pixels = _OPA_TFORM_CACHE["pickscore"](images01).to(dtype=scorer.dtype, device=dev)
        text_inputs = scorer.processor(text=list(prompts), padding=True, truncation=True,
                                       max_length=77, return_tensors="pt").to(dev)
        img_e = scorer.model.get_image_features(pixel_values=pixels)
        img_e = img_e / img_e.norm(p=2, dim=-1, keepdim=True)
        txt_e = scorer.model.get_text_features(**text_inputs)
        txt_e = txt_e / txt_e.norm(p=2, dim=-1, keepdim=True)
        return ((scorer.model.logit_scale.exp() * (txt_e @ img_e.T)).diag() / 26).float()
    if kind == "aesthetic":
        if "aesthetic" not in _OPA_TFORM_CACHE:
            from diffusionopsd.clip_scorer import get_image_transform  # lazy (see pickscore branch)
            _OPA_TFORM_CACHE["aesthetic"] = get_image_transform(scorer.processor.image_processor)
        pixels = _OPA_TFORM_CACHE["aesthetic"](images01).to(dtype=scorer.dtype, device=dev)
        embed = scorer.clip.get_image_features(pixel_values=pixels)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return scorer.mlp.layers(embed).squeeze(1).float()  # .layers bypasses MLP.forward's @no_grad
    if kind == "hpsv3":
        if zimage_heavy_diff_bridge_enabled():
            # 7B hpsv3 lives on the diff-bridge server; ship the decoded image, get reward+grad back.
            from diffusionopsd.zimage_heavy_diff_bridge import remote_heavy_reward_scores
            return remote_heavy_reward_scores(images01, prompts).float()
        return scorer._scores(images01, prompts).float()  # differentiable Qwen2-VL-7B ranknet mu
    if kind == "deqa":
        if zimage_heavy_diff_bridge_enabled():
            # 8B deqa lives on the diff-bridge server; ship the decoded image, get reward+grad back.
            from diffusionopsd.zimage_heavy_diff_bridge import remote_heavy_reward_scores
            return remote_heavy_reward_scores(images01, prompts).float()
        return scorer._scores(images01, prompts).float()  # differentiable mPLUG-Owl2 rating-token MOS
    if kind == "imagereward":
        return scorer._scores(images01, prompts).float()  # differentiable ImageReward BLIP score_gard
    raise ValueError(f"OPA differentiable ascent: unsupported reward kind '{kind}'")


def _load_reward_scorer(kind, device):
    """Load the differentiable scorer matching the training reward (weights frozen)."""
    if kind in ("open3", "multi_open3"):
        s = {
            "pickscore": _load_reward_scorer("pickscore", device),
            "clipscore": _load_reward_scorer("clipscore", device),
            "hpsv2": _load_reward_scorer("hpsv2", device),
        }
        return s
    if kind == "hpsv2":
        from diffusionopsd.hpsv2_scorer import HPSv2Scorer

        s = HPSv2Scorer(dtype=torch.float32, device=device)
    elif kind == "clipscore":
        from diffusionopsd.clip_scorer import ClipScorer
        s = ClipScorer(device=device); s.dtype = torch.float32
    elif kind == "pickscore":
        from diffusionopsd.pickscore_scorer import PickScoreScorer
        s = PickScoreScorer(device=device, dtype=torch.float32)
    elif kind == "aesthetic":
        from diffusionopsd.aesthetic_scorer import AestheticScorer
        s = AestheticScorer(dtype=torch.float32, device=device)
    elif kind == "hpsv3":
        from diffusionopsd.hpsv3_scorer import get_hpsv3_scorer
        s = get_hpsv3_scorer(device=device)  # shared 7B singleton (avoid 3× load -> OOM)
    elif kind == "deqa":
        from diffusionopsd.deqa_scorer import get_deqa_scorer
        s = get_deqa_scorer(device=device)  # process singleton (shares the frozen 7B across reward_fn/eval/ri_scorer)
    elif kind == "imagereward":
        from diffusionopsd.imagereward_scorer import ImageRewardScorer
        s = ImageRewardScorer(device=device, dtype=torch.float32)  # differentiable BLIP score_gard
    else:
        raise ValueError(f"OPA: no differentiable scorer for reward '{kind}'.")
    s.requires_grad_(False)
    return s


def _reward_of_latents_grad(pipeline, scorer, kind, x_latent, prompts):
    # Decode the Z-Image latent to [0,1] RGB (differentiable; VAE grad-ckpt bounds decode memory),
    # then score with the generic per-reward differentiable path. zimage_decode replaces SD3's
    # _decode01; the scoring math is identical.
    return _reward_scores_grad(scorer, kind, zimage_decode(pipeline.vae, x_latent), prompts)


@torch.no_grad()
def _reward_of_latents(pipeline, scorer, kind, x_latent, prompts):
    return scorer(zimage_decode(pipeline.vae, x_latent), list(prompts)).float()


def _opa_tr_step(pipeline, scorer, kind, y0, prompts, rho, n_ascent, eta, direction,
                 dir_mode="grad", x_end=None, first_grad=None):
    """OPA target: trust-region step (+1 ascent / -1 descent) at the low-noise x0 anchor y0.

    Reused verbatim from the SD3 OPA trainer (only _reward_of_latents_grad now decodes via the
    Z-Image VAE). dir_mode: 'grad' (the method = training-reward gradient) | 'rand' (attribution
    control) | 'residual' (denoising residual x_end-y0, ATC-style). No sampler is invoked here."""
    x0 = y0.detach().float()
    budget = (rho * x0.flatten(1).norm(dim=1)).view(-1, 1, 1, 1)
    step_len = eta * budget / max(n_ascent, 1)
    fixed_dir = None
    if dir_mode == "rand":
        fixed_dir = torch.randn_like(x0)
    elif dir_mode == "residual":
        if x_end is None:
            raise ValueError("_opa_tr_step dir_mode='residual' requires x_end (rollout endpoint).")
        fixed_dir = x_end.detach().float() - x0
    elif dir_mode != "grad":
        raise ValueError(f"_opa_tr_step dir_mode '{dir_mode}' unknown (grad|rand|residual).")
    x = x0.clone()
    for _i in range(n_ascent):
        if dir_mode == "grad":
            if _i == 0 and first_grad is not None:
                g = first_grad  # Shared y0 gradient is identical for positive/negative first steps.
            else:
                x = x.detach().requires_grad_(True)
                r = _reward_of_latents_grad(pipeline, scorer, kind, x, prompts)
                (g,) = torch.autograd.grad(r.sum(), x)
                profiling.reward_bwd_inc()  # §6: reward-gradient backward (OPA trust-region ascent/descent)
        else:
            g = fixed_dir
        gn = g.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12
        x = x.detach() + float(direction) * step_len * (g / gn)
        d = x - x0
        dn = d.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        x = (x0 + d * torch.clamp(budget / (dn + 1e-12), max=1.0)).detach()
    return x.detach()


# =============================== boilerplate (verbatim from SD3) =============================== #
def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    # Bind THIS rank's GPU BEFORE init and pass device_id so NCCL knows the rank->GPU map up front.
    # Under ZIMAGE_HEAVY_DIFF_BRIDGE the first default-world NCCL op is make_bridge_groups' eager
    # barrier across all 7 ranks (6 policy + server); on the newer Z-Image ISO-env torch a bare init
    # leaves "devices used by this process are currently unknown" and that barrier HANGS the node.
    # device_id binds the default communicator to this rank's GPU at init. Guarded so an older torch
    # lacking the kwarg still works unchanged.
    torch.cuda.set_device(lock_rank)
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size,
                                device_id=torch.device(f"cuda:{lock_rank}"), timeout=__import__("datetime").timedelta(hours=6))
    except TypeError:
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=__import__("datetime").timedelta(hours=6))


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        return [e["prompt"] for e in examples], [e["metadata"] for e in examples]


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, (
            f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}")
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                per_card_samples.append(shuffled_samples[start:start + self.batch_size])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def gather_tensor_to_all(tensor, world_size):
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=POLICY_GROUP)  # POLICY_GROUP=None => default world
    return torch.cat(gathered, dim=0).cpu()


def return_decay(step, decay_type):
    if decay_type == 0:
        flat, uprate, uphold = 0, 0.0, 0.0
    elif decay_type == 1:
        flat, uprate, uphold = 0, 0.001, 0.5
    elif decay_type == 2:
        flat, uprate, uphold = 75, 0.0075, 0.999
    else:
        assert False
    if step < flat:
        return 0.0
    return min((step - flat) * uprate, uphold)




@torch.no_grad()
def encode_prompts_list(pipeline, prompts, device):
    return zimage_encode_prompt(pipeline, prompts, device, max_sequence_length=QWEN_MAX_SEQ_LEN)


def eval_fn(pipeline, transformer_ddp, test_dataloader, config, device, rank, world_size,
            global_step, reward_fn, executor, mixed_precision_dtype, ema, transformer_trainable_parameters):
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    pipeline.transformer.eval()
    transformer_ddp.module.set_adapter("default")

    all_rewards = defaultdict(list)
    test_sampler = (DistributedSampler(test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
                    if world_size > 1 else None)
    eval_loader = DataLoader(test_dataloader.dataset, batch_size=config.sample.test_batch_size, sampler=test_sampler,
                             collate_fn=test_dataloader.collate_fn, num_workers=test_dataloader.num_workers)
    images = None
    prompts = None
    for test_batch in tqdm(eval_loader, desc="Eval: ", disable=not is_main_process(rank), position=0):
        prompts, prompt_metadata = test_batch
        prompt_embeds_list = encode_prompts_list(pipeline, prompts, device)
        with torch_autocast(enabled=(config.mixed_precision in ["fp16", "bf16"]), dtype=mixed_precision_dtype):
            with torch.no_grad():
                out = zimage_rollout(pipeline, prompt_embeds_list, num_inference_steps=config.sample.eval_num_steps,
                                     height=config.resolution, width=config.resolution, device=device,
                                     guidance_scale=config.sample.guidance_scale, decode=True)
        images = out["images"]
        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, _ = rewards_future.result()
        for key, value in rewards.items():
            all_rewards[key].append(gather_tensor_to_all(torch.as_tensor(value, device=device).float(), world_size).numpy())

    if is_main_process(rank):
        final_rewards = {key: np.concatenate(v) for key, v in all_rewards.items()}
        images_to_log = images.cpu()
        with tempfile.TemporaryDirectory() as tmpdir:
            n = min(15, len(images_to_log))
            for idx in range(n):
                Image.fromarray((images_to_log[idx].float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                                ).save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts_log = [prompts[i] for i in range(n)]
            sampled_rewards_log = [{k: final_rewards[k][i] for k in final_rewards} for i in range(n)]
            persist_dir = os.path.join(config.save_dir, "eval_samples", f"step_{global_step}")
            os.makedirs(persist_dir, exist_ok=True)
            captions = []
            for idx in range(n):
                Image.open(os.path.join(tmpdir, f"{idx}.jpg")).save(os.path.join(persist_dir, f"{idx}.jpg"))
                rw = " ".join(f"{k}:{sampled_rewards_log[idx][k]:.3f}" for k in sampled_rewards_log[idx])
                captions.append(f"{idx}\t{rw}\t{sampled_prompts_log[idx][:200]}")
            with open(os.path.join(persist_dir, "captions.txt"), "w") as cf:
                cf.write("\n".join(captions))
            wandb.log({
                "eval_images": [wandb.Image(os.path.join(tmpdir, f"{idx}.jpg"),
                                            caption=f"{prompt:.1000} | " + " | ".join(
                                                f"{k}: {v:.2f}" for k, v in reward.items() if v != -10))
                                for idx, (prompt, reward) in enumerate(zip(sampled_prompts_log, sampled_rewards_log))],
                **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in final_rewards.items()},
            }, step=global_step)

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)
    if world_size > 1:
        dist.barrier(group=POLICY_GROUP, device_ids=[torch.cuda.current_device()])


def save_ckpt(save_dir, transformer_ddp, global_step, rank, ema, transformer_trainable_parameters,
              config, optimizer, scaler, epoch_completed=None):
    if is_main_process(rank):
        save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
        save_root_lora = os.path.join(save_root, "lora")
        os.makedirs(save_root_lora, exist_ok=True)
        if config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        transformer_ddp.module.save_pretrained(save_root_lora)
        torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
        if scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(save_root, "scaler.pt"))
        save_trainer_state(
            save_root, epoch_completed=(global_step if epoch_completed is None else epoch_completed),
            global_step=global_step, ema=ema,
        )
        if config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)
        logger.info(f"Saved checkpoint to {save_root}")


def main(_):
    global POLICY_GROUP
    config = FLAGS.config
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    setup_distributed(rank, local_rank, world_size)  # inits the default NCCL world over ALL launched ranks
    device = torch.device(f"cuda:{local_rank}")

    # --- ZIMAGE_HEAVY_DIFF_BRIDGE (gated): last rank is a single-GPU DIFFERENTIABLE heavy-reward
    # server; ranks 0..N-2 are the policy. Unlike the nft/flowgrpo forward-only bridge, this server
    # returns reward AND gradient so the OPSD reward-gradient target ascent can route hpsv3 (7B) /
    # deqa (8B) remotely instead of OOMing co-located on the 6B Z-Image DiT. The server NEVER touches
    # the policy/optimizer/dataset/checkpointing and RETURNS before loading any of them. Flag off (or
    # a co-located reward) => this block is skipped and the run is byte-identical to the light path.
    # Gate on the reward too so a mis-inherited flag on imagereward degrades to a normal 8-GPU run. ---
    _diff_bridge_active = zimage_heavy_diff_bridge_enabled() and list(config.reward_fn.keys())[0] in ("hpsv3", "deqa")
    if _diff_bridge_active:
        from diffusionopsd.zimage_heavy_diff_bridge import (
            make_bridge_groups, is_server_rank, HeavyDiffRewardServer, bridge_server_devices,
            policy_group as _bridge_policy_group,
        )
        make_bridge_groups(world_size)  # ALL ranks call this (new_group is a world collective)
        if is_server_rank(rank):
            # reward_kind selects the heavy diff scorer the server loads (hpsv3 / deqa).
            server = HeavyDiffRewardServer(bridge_server_devices(local_rank)[0],
                                           reward_kind=list(config.reward_fn.keys())[0])
            server.serve(n_policy=world_size - 1)  # blocks until every policy rank sends CMD_SHUTDOWN
            cleanup_distributed()
            return
        POLICY_GROUP = _bridge_policy_group()   # DDP + all policy collectives use this subgroup
        _server_rank = world_size - 1
        world_size = world_size - 1             # policy world size drives sampler counts + batch math
        logger.info(f"[diff-bridge] policy rank={rank} policy_world_size={world_size} "
                    f"(heavy diff reward server=rank {_server_rank})")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = (config.run_name + "_" + unique_id) if config.run_name else unique_id

    if is_main_process(rank):
        os.makedirs(config.save_dir, exist_ok=True)
        log_dir = os.path.join(config.logdir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        wandb.init(project="diffusionopsd", name=config.run_name, config=config.to_dict(), dir=log_dir)
        install_wandb_jsonl_tee(wandb, os.path.join(config.save_dir, "metrics.jsonl"))
    logger.info(f"\n{config}")

    if config.seed is not None:
        run_random_nonce = int(config.seed)
        set_seed(config.seed, rank)
    else:
        nonce_tensor = torch.zeros(1, dtype=torch.long, device=device)
        if is_main_process(rank):
            nonce_tensor[0] = int.from_bytes(os.urandom(8), "little") % (2**31 - 1)
        if world_size > 1:
            dist.broadcast(nonce_tensor, src=0, group=POLICY_GROUP)  # POLICY_GROUP=None => default world
        run_random_nonce = int(nonce_tensor.item())
        logger.info(f"[seed] NO fixed seed; run_random_nonce={run_random_nonce} (prompt grouping only)")

    # ---- OPA config parse (same field names as the SD3 OPA trainer) ----
    ri = config.opsd
    ri_train_state = str(ri.train_state)
    ri_opa = bool(int(ri.get("opa", 0)))
    opa_rho = float(ri.get("opa_rho", 0.10))
    opa_n_ascent = int(ri.get("opa_n_ascent", 2))
    opa_eta = float(ri.get("opa_eta", 1.0))
    opa_query_sigma = float(ri.get("opa_query_sigma", 0.273))
    opa_mb = int(ri.get("opa_mb", 6))
    ri_opa_dual_neg = bool(int(ri.get("opa_dual_neg", 1)))
    ri_opa_cert = bool(int(ri.get("opa_cert", 0)))
    opa_dir_mode = str(ri.get("opa_dir_mode", "grad"))
    opa_kind = list(config.reward_fn.keys())[0] if ri_opa else "hpsv2"
    if ri_opa and ri_train_state != "rollout":
        raise ValueError("opsd.opa=1 requires opsd.train_state='rollout' (on-policy z_q states).")
    # hpsv3/deqa become differentiable-trainable ONLY when routed to the diff bridge; still rejected
    # co-located (they OOM next to the 6B DiT). imagereward stays co-located (NOT added here).
    _allowed_diff_rewards = OPSD_ZIMAGE_DIFF_REWARDS + (("hpsv3", "deqa") if _diff_bridge_active else ())
    if ri_opa and opa_kind not in _allowed_diff_rewards:
        raise ValueError(
            f"Z-Image OPA supports differentiable rewards {_allowed_diff_rewards}; got "
            f"'{opa_kind}'. Heavy rewards hpsv3/deqa need ZIMAGE_HEAVY_DIFF_BRIDGE=1.")
    if ri_opa and ri_opa_cert:
        # The SD3 certification used an EXACT dpm2 suffix continuation. Z-Image uses FlowMatchEuler;
        # a faithful cert would force the action at z_q then run the deterministic Euler suffix and
        # re-score the endpoint. The public preset uses the no-cert fast path (opa_cert=0).
        raise NotImplementedError(
            "opa_cert=1 is not ported for Z-Image (needs a FlowMatchEuler suffix continuation). "
            "Use the canonical no-cert path opa_cert=0.")
    if is_main_process(rank):
        logger.info(f"[OPA] opa={int(ri_opa)} reward={opa_kind} dir_mode={opa_dir_mode} cert={int(ri_opa_cert)} "
                    f"dual_neg={int(ri_opa_dual_neg)} rho={opa_rho} n_ascent={opa_n_ascent} eta={opa_eta} "
                    f"query_sigma={opa_query_sigma} mb={opa_mb}")

    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16
    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=enable_amp)

    from diffusers import ZImagePipeline
    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32
    pipeline = ZImagePipeline.from_pretrained(config.pretrained.model, torch_dtype=text_encoder_dtype)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    try:
        pipeline.set_progress_bar_config(disable=not is_main_process(rank))
    except Exception:
        pass
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    transformer = pipeline.transformer.to(device)
    try:
        transformer.enable_gradient_checkpointing()
    except Exception as e:
        logger.warning(f"transformer.enable_gradient_checkpointing() unavailable: {e}")

    if config.use_lora:
        transformer_lora_config = LoraConfig(
            r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=ZIMAGE_LORA_TARGETS)
        if config.train.lora_path:
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, transformer_lora_config)
        transformer.add_adapter("old", transformer_lora_config)
        transformer.set_adapter("default")

    transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank,
                          find_unused_parameters=False, process_group=POLICY_GROUP)  # None => default world
    transformer_ddp.module.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("old")
    old_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("default")

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        transformer_trainable_parameters, lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay, eps=config.train.adam_epsilon)

    train_dataset = TextPromptDataset(config.dataset, "train")
    test_dataset = TextPromptDataset(config.dataset, "test")
    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset, batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt, num_replicas=world_size, rank=rank, seed=run_random_nonce)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0,
                                  collate_fn=train_dataset.collate_fn, pin_memory=True)
    test_sampler = (DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
                    if world_size > 1 else None)
    test_dataloader = DataLoader(test_dataset, batch_size=config.sample.test_batch_size, sampler=test_sampler,
                                 collate_fn=test_dataset.collate_fn, num_workers=0, pin_memory=True)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)
    else:
        assert False

    executor = futures.ThreadPoolExecutor(max_workers=8)
    reward_fn = getattr(diffusionopsd.rewards, "multi_score")(device, config.reward_fn)
    eval_reward_fn = getattr(diffusionopsd.rewards, "multi_score")(device, config.reward_fn)

    # --- Paper efficiency profiling (env PROFILE=1) ---
    # Enable per-optimizer-step timing/counting, wrap the rollout reward scorer so each call counts
    # as one reward forward, and force debug mode so periodic/final save_ckpt and eval_fn are skipped.
    if profiling.profile_enabled():
        profiling.enable()
        config.debug = True
        reward_fn = profiling.count_reward_fn(reward_fn)

    # Differentiable reward scorer (opa_kind) for the OPA target ascent (weights frozen); grad-ckpt
    # the VAE so the decode->reward backprop peak memory is bounded (matches the SD3 OPA trainer).
    ri_scorer = None
    if ri_opa:
        if _diff_bridge_active and opa_kind in ("hpsv3", "deqa"):
            # 7B/8B grad reward lives on the diff-bridge server; _reward_scores_grad routes remotely,
            # so load NO local scorer here (would OOM next to the 6B DiT). scorer arg stays None.
            ri_scorer = None
            if is_main_process(rank):
                logger.info(f"[diff-bridge] {opa_kind} grad reward served remotely; no local scorer on policy ranks")
        else:
            ri_scorer = _load_reward_scorer(opa_kind, device)  # frozen differentiable scorer for the training reward
        # VAE decode still runs LOCALLY (decode->image shipped to server), so grad-ckpt it regardless.
        try:
            pipeline.vae.enable_gradient_checkpointing()
        except Exception as e:
            logger.warning(f"pipeline.vae.enable_gradient_checkpointing() unavailable: {e}")

    first_epoch = 0
    global_step = 0
    if config.resume_from:
        config.resume_from = resolve_resume_checkpoint(config.resume_from)
        lora_path = os.path.join(config.resume_from, "lora")
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
        lora_state = load_peft_weights(lora_path, device=str(device))
        set_peft_model_state_dict(transformer_ddp.module, lora_state, adapter_name="default")
        set_peft_model_state_dict(transformer_ddp.module, lora_state, adapter_name="old")
        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.isfile(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        scaler_path = os.path.join(config.resume_from, "scaler.pt")
        if os.path.isfile(scaler_path) and enable_amp:
            scaler.load_state_dict(torch.load(scaler_path, map_location=device))
        first_epoch, global_step = resume_position(config.resume_from, config, world_size)
        logger.info(f"Resume position: first_epoch={first_epoch}, global_step={global_step}")

    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=1, device=device)
    if config.resume_from:
        restore_ema_and_rng(config.resume_from, ema)
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()
    for src_param, tgt_param in zip(transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True):
        tgt_param.data.copy_(src_param.detach().data)
        assert src_param is not tgt_param

    # --- §6 profiler: 1 optimizer step == 1 epoch (gradient_step_per_epoch=1) ---
    prof = profiling.Profiler(config, world_size, rank, device) if profiling.is_enabled() else None
    prof_epoch0 = first_epoch

    for epoch in range(first_epoch, config.num_epochs):
        if prof is not None:
            prof.epoch_begin(epoch - prof_epoch0)
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # =============================== SAMPLING (old adapter) =============================== #
        pipeline.transformer.eval()
        samples_data_list = []
        epoch_prompt_embeds = []
        epoch_prompts_text = []
        images = None
        prompts = None

        for i in tqdm(range(config.sample.num_batches_per_epoch), desc=f"Epoch {epoch}: sampling",
                      disable=not is_main_process(rank), position=0):
            transformer_ddp.module.set_adapter("default")
            if isinstance(train_sampler, DistributedKRepeatSampler):
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)
            epoch_prompts_text.extend(list(prompts))

            prompt_embeds_list = encode_prompts_list(pipeline, prompts, device)
            prompt_ids = pipeline.tokenizer(prompts, padding="max_length", max_length=256,
                                            truncation=True, return_tensors="pt").input_ids.to(device)

            if i == 0 and config.eval_freq > 0 and epoch % config.eval_freq == 0 and not config.debug:
                eval_fn(pipeline, transformer_ddp, test_dataloader, config, device, rank, world_size,
                        global_step, eval_reward_fn, executor, mixed_precision_dtype, ema,
                        transformer_trainable_parameters)
            if (
                i == 0
                and global_step > 0
                and global_step % config.save_freq == 0
                and is_main_process(rank)
                and not config.debug
            ):
                save_ckpt(config.save_dir, transformer_ddp, global_step, rank, ema,
                          transformer_trainable_parameters, config, optimizer, scaler,
                          epoch_completed=epoch)

            transformer_ddp.module.set_adapter("old")
            with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    out = zimage_rollout(pipeline, prompt_embeds_list, num_inference_steps=config.sample.num_steps,
                                         height=config.resolution, width=config.resolution, device=device,
                                         guidance_scale=config.sample.guidance_scale, decode=True)
            transformer_ddp.module.set_adapter("default")

            images = out["images"]
            x0 = out["x0"]
            sigmas = out["sigmas"]                                       # (num_steps+1,)
            timesteps = out["timesteps"].repeat(len(prompts), 1).to(device)
            # OPA query state: fixed index from the scheduler sigma schedule (~=0.273 @ step 8).
            kq = zimage_query_sigma_index(sigmas, opa_query_sigma)
            z_q = out["latents"][kq].detach()                           # (B,C,H,W) real rollout state
            sig_q = float(sigmas[kq].item())

            base_idx = len(epoch_prompt_embeds)
            embed_idx = torch.arange(base_idx, base_idx + len(prompts), device=device)
            epoch_prompt_embeds.extend([e.detach() for e in prompt_embeds_list])

            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)
            samples_data_list.append({
                "prompt_ids": prompt_ids,
                "embed_idx": embed_idx,
                "timesteps": timesteps,
                "latents_clean": x0,
                "opa_z_q": z_q,
                "opa_sig_q": torch.full((len(prompts),), sig_q, device=device),
                "rewards_future": rewards_future,
            })

        for sample_item in tqdm(samples_data_list, desc="Waiting for rewards",
                                disable=not is_main_process(rank), position=0):
            rewards, _ = sample_item["rewards_future"].result()
            sample_item["rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}
            del sample_item["rewards_future"]

        collated_samples = {
            k: (torch.cat([s[k] for s in samples_data_list], dim=0)
                if not isinstance(samples_data_list[0][k], dict)
                else {sk: torch.cat([s[k][sk] for s in samples_data_list], dim=0) for sk in samples_data_list[0][k]})
            for k in samples_data_list[0].keys()
        }

        if epoch % 10 == 0 and is_main_process(rank):
            images_to_log = images.cpu()
            rewards_to_log = collated_samples["rewards"]["avg"][-len(images_to_log):].cpu()
            with tempfile.TemporaryDirectory() as tmpdir:
                num_to_log = min(15, len(images_to_log))
                for idx in range(num_to_log):
                    Image.fromarray((images_to_log[idx].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                                    ).save(os.path.join(tmpdir, f"{idx}.jpg"))
                wandb.log({"images": [wandb.Image(os.path.join(tmpdir, f"{idx}.jpg"),
                                                  caption=f"{prompts[idx]:.100} | avg: {rewards_to_log[idx]:.2f}")
                                      for idx in range(num_to_log)]}, step=global_step)

        collated_samples["rewards"]["avg"] = collated_samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        gathered_rewards_dict = {k: gather_tensor_to_all(v, world_size).numpy()
                                 for k, v in collated_samples["rewards"].items()}
        if is_main_process(rank):
            wandb.log({"epoch": epoch, **{f"reward_{k}": v.mean() for k, v in gathered_rewards_dict.items()
                                          if "_strict_accuracy" not in k and "_accuracy" not in k}}, step=global_step)

        if config.per_prompt_stat_tracking:
            prompt_ids_all = gather_tensor_to_all(collated_samples["prompt_ids"], world_size)
            prompts_all_decoded = pipeline.tokenizer.batch_decode(prompt_ids_all.cpu().numpy(), skip_special_tokens=True)
            if is_main_process(rank):
                write_raw_reward_jsonl(
                    config.save_dir, epoch=epoch, global_step=global_step,
                    prompts=prompts_all_decoded, rewards=gathered_rewards_dict,
                )
            advantages = stat_tracker.update(prompts_all_decoded, gathered_rewards_dict["avg"])
            if is_main_process(rank):
                group_size, trained_prompt_num = stat_tracker.get_stats()
                dispersion_stats = calculate_prompt_group_dispersion(
                    prompts_all_decoded, gathered_rewards_dict["avg"]
                )
                wandb.log({"group_size": group_size, "trained_prompt_num": trained_prompt_num,
                           **dispersion_stats,
                           "mean_reward_10": stat_tracker.get_mean_of_top_rewards(10)}, step=global_step)
            stat_tracker.clear()
        else:
            avg = gathered_rewards_dict["avg"]
            advantages = (avg - avg.mean()) / (avg.std() + 1e-4)

        samples_per_gpu = collated_samples["timesteps"].shape[0]
        if advantages.ndim == 1:
            advantages = advantages[:, None]
        assert advantages.shape[0] == world_size * samples_per_gpu
        collated_samples["advantages"] = torch.from_numpy(
            advantages.reshape(world_size, samples_per_gpu, -1)[rank]).to(device)

        # =============== OPA target construction: certified dual action targets at z_q =============== #
        # y0 = z_q - sig_q*v_old ; y+/y- = HPS ascent/descent at y0 (trust-region). No-cert (opa_cert=0)
        # accepts both bounded targets directly, matching the canonical paper configuration.
        latents_clean_local = collated_samples["latents_clean"]
        Bn = latents_clean_local.shape[0]
        opa_yp = latents_clean_local.clone()
        opa_ym = latents_clean_local.clone()
        opa_ap = torch.zeros(Bn, device=device)
        opa_am = torch.zeros(Bn, device=device)
        opa_res = torch.zeros(Bn, device=device)
        sig_q = float(collated_samples["opa_sig_q"][0].item())
        rs_q = collated_samples["opa_z_q"]
        for s in range(0, Bn, opa_mb):
            e = min(s + opa_mb, Bn)
            zq = rs_q[s:e].float()
            xi = latents_clean_local[s:e].float()
            emb = [epoch_prompt_embeds[int(k)] for k in collated_samples["embed_idx"][s:e].tolist()]
            prm = epoch_prompts_text[s:e]
            with torch.no_grad(), torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                transformer_ddp.module.set_adapter("old")
                v_old_q = zimage_v(pipeline.transformer, zq, sig_q, emb)
                transformer_ddp.module.set_adapter("default")
            y0 = zq - sig_q * v_old_q.float()
            # The first ascent gradient at y0 is identical for the positive/negative directions
            # under the same anchor and reward, so compute it once and share it as both
            # directions' first step -> saves one reward-grad per chunk (n_ascent=2: 4/chunk -> 3).
            # grad path only (rand/residual use a fixed direction, no reward-grad).
            opa_g0 = None
            if opa_dir_mode == "grad" and opa_n_ascent > 0:  # n_ascent=0 => _opa_tr_step does no ascent, so don't pay/count g0
                _xg = y0.detach().float().requires_grad_(True)
                _rg = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, _xg, prm)
                (opa_g0,) = torch.autograd.grad(_rg.sum(), _xg)
                profiling.reward_bwd_inc()  # §6: the shared y0 reward-gradient backward
            y_plus = _opa_tr_step(pipeline, ri_scorer, opa_kind, y0, prm, opa_rho, opa_n_ascent, opa_eta,
                                  +1.0, opa_dir_mode, xi, first_grad=opa_g0)
            y_minus = _opa_tr_step(pipeline, ri_scorer, opa_kind, y0, prm, opa_rho, opa_n_ascent, opa_eta,
                                   -1.0, opa_dir_mode, xi, first_grad=opa_g0)
            # no-cert fast path: accept both dual targets directly.
            opa_ap[s:e] = 1.0
            opa_am[s:e] = 1.0
            opa_yp[s:e] = y_plus.to(opa_yp.dtype)
            opa_ym[s:e] = y_minus.to(opa_ym.dtype)
            opa_res[s:e] = (y_plus - y0).flatten(1).norm(dim=1) / (y0.flatten(1).norm(dim=1) + 1e-8)
        transformer_ddp.module.set_adapter("default")
        collated_samples["opa_y_plus"] = opa_yp
        collated_samples["opa_y_minus"] = opa_ym
        collated_samples["opa_acc_plus"] = opa_ap
        collated_samples["opa_acc_minus"] = opa_am
        if is_main_process(rank):
            wandb.log({"opa_kq_sigma": sig_q, "opa_acc_plus": float(opa_ap.mean()),
                       "opa_acc_minus": float(opa_am.mean()), "opa_delta_res": float(opa_res.mean())}, step=global_step)

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]

        num_batches = config.sample.num_batches_per_epoch * config.sample.train_batch_size // config.train.batch_size
        filtered_samples = collated_samples
        total_batch_size_filtered = filtered_samples["timesteps"].shape[0]

        # =================================== TRAINING (single query state) =================================== #
        transformer_ddp.train()
        opa_accum = max(1, config.train.gradient_accumulation_steps)
        current_accumulated_steps = 0
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled = {k: v[perm] for k, v in filtered_samples.items()}
            training_batch_size = total_batch_size_filtered // num_batches
            samples_batched_list = [
                {key: val[k_batch * training_batch_size:(k_batch + 1) * training_batch_size]
                 for key, val in shuffled.items()}
                for k_batch in range(num_batches)]

            info_accumulated = defaultdict(list)
            for i, batch in tqdm(list(enumerate(samples_batched_list)),
                                 desc=f"Epoch {epoch}.{inner_epoch}: training",
                                 position=0, disable=not is_main_process(rank)):
                embeds_list = [epoch_prompt_embeds[int(k)] for k in batch["embed_idx"].tolist()]
                zq = batch["opa_z_q"].float()
                y_plus = batch["opa_y_plus"].float()
                y_minus = batch["opa_y_minus"].float()
                acc_p = batch["opa_acc_plus"].float()
                acc_m = batch["opa_acc_minus"].float()
                sig_q = float(batch["opa_sig_q"][0].item())
                adv = batch["advantages"]
                adv = (adv[:, 0] if adv.ndim > 1 else adv).float()

                with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                    transformer_ddp.module.set_adapter("old")
                    with torch.no_grad():
                        v_old_q = zimage_v(transformer_ddp, zq, sig_q, embeds_list).detach()
                    transformer_ddp.module.set_adapter("default")
                    v_theta_q = zimage_v(transformer_ddp, zq, sig_q, embeds_list)   # grad

                v_old_q = v_old_q.float()
                v_pos = config.beta * v_theta_q.float() + (1 - config.beta) * v_old_q
                v_neg = (1.0 + config.beta) * v_old_q - config.beta * v_theta_q.float()
                y_pos = zq - sig_q * v_pos                                          # SIGN: v=-v_raw
                y_neg = zq - sig_q * v_neg
                rd = tuple(range(1, y_pos.ndim))
                adv_clip = torch.clamp(adv, -config.train.adv_clip_max, config.train.adv_clip_max)
                r1 = torch.clamp((adv_clip / config.train.adv_clip_max) / 2.0 + 0.5, 0, 1)
                with torch.no_grad():
                    wf_p = torch.abs(y_pos.double() - y_plus.double()).mean(dim=rd, keepdim=True).clip(min=1e-5)
                    wf_n = torch.abs(y_neg.double() - y_minus.double()).mean(dim=rd, keepdim=True).clip(min=1e-5)
                accp = acc_p.view(-1, *([1] * (y_pos.ndim - 1)))
                accm = acc_m.view(-1, *([1] * (y_pos.ndim - 1)))
                pos_loss = (accp * (y_pos - y_plus) ** 2 / wf_p).mean(dim=rd)
                neg_loss = (accm * (y_neg - y_minus) ** 2 / wf_n).mean(dim=rd)
                if not ri_opa_dual_neg:
                    neg_loss = neg_loss * 0.0
                opa_policy = (r1 * pos_loss + (1.0 - r1) * neg_loss).mean() * config.train.adv_clip_max
                # Canonical OPSD uses only target fitting.  A frozen-base prediction
                # MSE can be tried as an optional FlowGRPO-style stabilizer, but the
                # tested coefficient (1e-4) had no measurable effect on the result.
                loss = opa_policy

                scaled_loss = loss / opa_accum
                if mixed_precision_dtype == torch.float16:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                profiling.train_bwd_inc()  # §6: diffusion (training) backward count
                current_accumulated_steps += 1
                for kk, vv in {"opa_policy_loss": opa_policy.detach(),
                               "total_loss": loss.detach(), "opa_pos_loss": pos_loss.mean().detach(),
                               "opa_neg_loss": neg_loss.mean().detach(),
                               "opa_acc_plus_b": acc_p.mean().detach(),
                               "opa_acc_minus_b": acc_m.mean().detach()}.items():
                    info_accumulated[kk].append(vv)

                if current_accumulated_steps % opa_accum == 0:
                    if mixed_precision_dtype == torch.float16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(transformer_ddp.module.parameters(), config.train.max_grad_norm)
                    if mixed_precision_dtype == torch.float16:
                        scaler.step(optimizer); scaler.update()
                    else:
                        optimizer.step()
                    gradient_update_times += 1
                    optimizer.zero_grad()
                    log_info = {k: torch.mean(torch.stack(v)).item() for k, v in info_accumulated.items()}
                    info_tensor = torch.tensor([log_info[k] for k in sorted(log_info)], device=device)
                    dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG, group=POLICY_GROUP)  # None => default world
                    reduced = {k: info_tensor[ki].item() for ki, k in enumerate(sorted(log_info))}
                    if is_main_process(rank):
                        wandb.log({"step": global_step, "gradient_update_times": gradient_update_times,
                                   "epoch": epoch, "inner_epoch": inner_epoch, **reduced})
                    global_step += 1
                    info_accumulated = defaultdict(list)
                    if config.train.ema and ema is not None:
                        ema.step(transformer_trainable_parameters, global_step)

        if world_size > 1:
            dist.barrier(group=POLICY_GROUP, device_ids=[torch.cuda.current_device()])
        with torch.no_grad():
            decay = return_decay(global_step, config.decay_type)
            for src_param, tgt_param in zip(transformer_trainable_parameters,
                                            old_transformer_trainable_parameters, strict=True):
                tgt_param.data.copy_(tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay))

        if prof is not None:
            prof.epoch_end(epoch - prof_epoch0, global_step=global_step)
            if prof.done(epoch - prof_epoch0):
                prof.finalize(None)
                break

    if prof is not None:
        prof.finalize(None)

    if not config.debug:
        save_ckpt(config.save_dir, transformer_ddp, global_step, rank, ema,
                  transformer_trainable_parameters, config, optimizer, scaler,
                  epoch_completed=config.num_epochs)
    if world_size > 1:
        dist.barrier(group=POLICY_GROUP, device_ids=[torch.cuda.current_device()])

    if is_main_process(rank):
        try:
            with open(os.path.join(config.save_dir, "run_done.json"), "w") as f:
                json.dump({"wall_clock_end": datetime.datetime.now().isoformat(), "global_step": global_step}, f, indent=2)
        except Exception:
            pass
        wandb.finish()
    if _diff_bridge_active:
        # Every policy rank tells the diff-reward server to exit, then all ranks (policy + server)
        # rendezvous on the gloo barrier before tearing down the process groups.
        from diffusionopsd.zimage_heavy_diff_bridge import bridge_client_shutdown
        bridge_client_shutdown()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
