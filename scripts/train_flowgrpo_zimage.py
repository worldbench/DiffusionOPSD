# SPDX-License-Identifier: Apache-2.0
#
# Flow-GRPO baseline on Z-Image-Turbo.
#
# Online group-relative PPO for flow-matching, built from the repo primitives
# (diffusionopsd.diffusers_patch.solver.flow_grpo_step + the Z-Image FlowMatchEuler pipeline).
# There is no Flow-GRPO trainer upstream in this repo; this one mirrors train_nft_zimage.py's
# DDP / LoRA / AdamW / eval / checkpoint / reward / group-normalization scaffolding and swaps
# the loss for a PPO-clipped policy gradient on per-step SDE log-probs.
#
#   Algorithm (per epoch):
#     1. STOCHASTIC rollout: for each prompt, the DistributedKRepeatSampler draws a GROUP of
#        group_size (= num_image_per_prompt) samples. Each is a stochastic SDE rollout of the
#        native 9-step FlowMatchEuler schedule (gs=0). Z-Image's sampler is deterministic
#        (eta=0); we inject per-step Gaussian noise (eta=config.sample.noise_level) and record the
#        per-step log-prob via flow_grpo_step (the SD3/MixGRPO flow-SDE transition kernel). The
#        full trajectory (z_0..z_T) and old log-probs log_pi_old are stored.
#     2. Reward + advantage: score the final image (HPSv2.1), group-normalize WITHIN each prompt
#        group to advantages a = (r - mean_group) / (std_group + eps)  (PerPromptStatTracker).
#     3. PPO update: at every stored transition (z_t -> z_{t+1}), recompute log_pi_new under the
#        policy, form ratio = exp(log_pi_new - log_pi_old), and minimize the clipped surrogate
#        -min(ratio * a, clip(ratio, 1-eps, 1+eps) * a). Optional KL-to-frozen-base (config.train.beta).
#
# SIGN: flow_grpo_step consumes the diffusers-convention velocity model_output = v = -v_raw; with
# eta=0 it reduces EXACTLY to the deterministic Z-Image Euler step z_{t+1}=z_t+(sigma_{t+1}-sigma_t)*v
# (see zimage_pipeline_with_rollout). So zimage_v(...) is fed directly as model_output.
#
# The behaviour policy is the CURRENT policy (default LoRA adapter): log_pi_old is captured during
# rollout, so there is no "old"/EMA-lagged adapter (unlike NFT). eval uses the DETERMINISTIC native
# sampler (matches the main-table cross-eval flow solver).

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
from diffusionopsd.zimage_reward_bridge import zimage_bridge_enabled  # ZIMAGE_HEAVY_BRIDGE reward server (gated)
from diffusionopsd.stat_tracking import PerPromptStatTracker, calculate_prompt_group_dispersion
from diffusionopsd.metrics import install_wandb_jsonl_tee
from diffusionopsd.diffusers_patch.zimage_pipeline_with_rollout import (
    zimage_encode_prompt, zimage_rollout, zimage_v, zimage_decode,
)
from diffusionopsd.diffusers_patch.solver import flow_grpo_step
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
from ml_collections import config_flags
from torch.cuda.amp import GradScaler, autocast as torch_autocast

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", "config/zimage.py:zimg_flowgrpo_hpsv2", "Training configuration."
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# ZIMAGE_HEAVY_BRIDGE: NCCL subgroup over policy ranks only (heavy-reward server excluded). None =>
# default world (bridge off), so passing group=POLICY_GROUP everywhere is byte-identical when off.
POLICY_GROUP = None

ZIMAGE_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"]
QWEN_MAX_SEQ_LEN = 512


# ===================== Stochastic (SDE) Z-Image rollout with per-step log-probs ===================== #
@torch.no_grad()
def zimage_rollout_with_logprob(pipe, prompt_embeds_list, num_inference_steps, height, width, device,
                                noise_level, guidance_scale=0.0, generator=None, decode=True):
    """Stochastic analog of zimage_rollout: 9-step FlowMatchEuler with per-step SDE noise + log-prob.

    Reuses flow_grpo_step (the flow-matching SDE transition kernel) with model_output = zimage_v
    (diffusers-convention velocity). eta=noise_level>0 turns the deterministic ODE into an SDE whose
    per-step Gaussian transition log-prob is returned (needed for the PPO ratio). The active LoRA
    adapter is set by the caller (default = current policy). Returns:
        latents   : list[(B,C,H,W)] length num_steps+1  (z_0 .. z_T)
        log_probs : list[(B,)]      length num_steps     (log pi_old per transition)
        sigmas    : (num_steps+1,)  scheduler sigma schedule (last ~0)
        timesteps : (num_steps,)
        x0        : (B,C,H,W) = latents[-1]
        images    : (B,3,H,W) in [0,1]  (iff decode)
    """
    assert float(guidance_scale) == 0.0, "Z-Image-Turbo is native gs=0 (single forward, no CFG)."
    transformer = pipe.transformer
    scheduler = pipe.scheduler
    B = len(prompt_embeds_list)

    num_channels_latents = transformer.config.in_channels
    latents = pipe.prepare_latents(
        B, num_channels_latents, height, width, prompt_embeds_list[0].dtype, device, generator)
    if isinstance(latents, (list, tuple)):
        latents = latents[0]

    scheduler.set_timesteps(num_inference_steps, device=device)
    sigmas = scheduler.sigmas.to(device).float()          # (num_steps+1,), last ~0
    timesteps = scheduler.timesteps.to(device).float()    # (num_steps,)
    num_steps = len(timesteps)

    z = latents.float()
    all_latents = [z]
    all_log_probs = []
    for i in range(num_steps):
        sigma = sigmas[i]
        v = zimage_v(transformer, z, sigma, prompt_embeds_list)      # diffusers-convention velocity (=-v_raw)
        z_next, _x0_pred, log_prob = flow_grpo_step(
            model_output=v.float(), latents=z.float(), eta=float(noise_level),
            sigmas=sigmas, index=i, prev_sample=None, generator=generator)
        z = z_next
        all_latents.append(z)
        all_log_probs.append(log_prob)

    out = {"latents": all_latents, "log_probs": all_log_probs, "sigmas": sigmas,
           "timesteps": timesteps, "x0": all_latents[-1]}
    if decode:
        out["images"] = zimage_decode(pipe.vae, all_latents[-1])
    return out


# =============================== boilerplate (verbatim from train_nft_zimage) =============================== #
def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    # Bind THIS rank's GPU BEFORE init and pass device_id so NCCL knows the rank->GPU map up front.
    # Under ZIMAGE_HEAVY_BRIDGE the first default-world NCCL op is make_bridge_groups' eager barrier
    # across all 7 ranks (6 policy + server); on the newer Z-Image ISO-env torch a bare init leaves
    # "devices used by this process are currently unknown" and that barrier HANGS the node. device_id
    # binds the default communicator to this rank's GPU at init. Guarded so an older torch lacking the
    # kwarg still works unchanged.
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
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


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
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor, group=POLICY_GROUP)  # POLICY_GROUP=None => default world
    return torch.cat(gathered_tensors, dim=0).cpu()




# ===================================== Z-Image helpers ===================================== #
@torch.no_grad()
def encode_prompts_list(pipeline, prompts, device):
    """Qwen3 encode -> list of (seq_i, 2560). Deterministic; cached at the epoch level."""
    return zimage_encode_prompt(pipeline, prompts, device, max_sequence_length=QWEN_MAX_SEQ_LEN)


def eval_fn(pipeline, transformer_ddp, test_dataloader, config, device, rank, world_size,
            global_step, reward_fn, executor, mixed_precision_dtype, ema,
            transformer_trainable_parameters):
    # Deterministic native sampler for eval (matches the main-table cross-eval flow solver).
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    pipeline.transformer.eval()
    transformer_ddp.module.set_adapter("default")

    all_rewards = defaultdict(list)
    test_sampler = (
        DistributedSampler(test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1 else None)
    eval_loader = DataLoader(
        test_dataloader.dataset, batch_size=config.sample.test_batch_size, sampler=test_sampler,
        collate_fn=test_dataloader.collate_fn, num_workers=test_dataloader.num_workers)

    images = None
    prompts = None
    for test_batch in tqdm(eval_loader, desc="Eval: ", disable=not is_main_process(rank), position=0):
        prompts, prompt_metadata = test_batch
        prompt_embeds_list = encode_prompts_list(pipeline, prompts, device)
        with torch_autocast(enabled=(config.mixed_precision in ["fp16", "bf16"]), dtype=mixed_precision_dtype):
            with torch.no_grad():
                out = zimage_rollout(
                    pipeline, prompt_embeds_list, num_inference_steps=config.sample.eval_num_steps,
                    height=config.resolution, width=config.resolution, device=device,
                    guidance_scale=config.sample.guidance_scale, decode=True)
        images = out["images"]
        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, _ = rewards_future.result()
        for key, value in rewards.items():
            rewards_tensor = torch.as_tensor(value, device=device).float()
            all_rewards[key].append(gather_tensor_to_all(rewards_tensor, world_size).numpy())

    if is_main_process(rank):
        final_rewards = {key: np.concatenate(v) for key, v in all_rewards.items()}
        images_to_log = images.cpu()
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples_to_log = min(15, len(images_to_log))
            for idx in range(num_samples_to_log):
                image = images_to_log[idx].float()
                pil = Image.fromarray((image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts_log = [prompts[i] for i in range(num_samples_to_log)]
            sampled_rewards_log = [{k: final_rewards[k][i] for k in final_rewards} for i in range(num_samples_to_log)]
            persist_dir = os.path.join(config.save_dir, "eval_samples", f"step_{global_step}")
            os.makedirs(persist_dir, exist_ok=True)
            captions = []
            for idx in range(num_samples_to_log):
                Image.open(os.path.join(tmpdir, f"{idx}.jpg")).save(os.path.join(persist_dir, f"{idx}.jpg"))
                rw = " ".join(f"{k}:{sampled_rewards_log[idx][k]:.3f}" for k in sampled_rewards_log[idx])
                captions.append(f"{idx}\t{rw}\t{sampled_prompts_log[idx][:200]}")
            with open(os.path.join(persist_dir, "captions.txt"), "w") as cf:
                cf.write("\n".join(captions))
            wandb.log({
                "eval_images": [
                    wandb.Image(os.path.join(tmpdir, f"{idx}.jpg"),
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
        model_to_save = transformer_ddp.module
        if config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        model_to_save.save_pretrained(save_root_lora)
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
    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    # --- ZIMAGE_HEAVY_BRIDGE (gated): last rank is a single-GPU heavy-reward server; ranks 0..N-2
    # are the policy. torchrun launches N=NPROC ranks. With the bridge, rank N-1 hosts the 7B/8B
    # scorer (hpsv3/deqa) and NEVER touches the policy/optimizer/dataloader/checkpointing; the other
    # N-1 ranks do Z-Image DDP training over a policy-only subgroup. Flag off => this block is skipped
    # entirely and the run is byte-identical to the co-located light-reward path. Flow-GRPO uses the
    # reward as a forward-only scalar (identical to nft), so the forward-only bridge ports verbatim.
    # Gate on the reward too (not the flag alone): the bridge server only serves hpsv3/deqa, so a
    # mis-inherited ZIMAGE_HEAVY_BRIDGE=1 on a co-located reward (e.g. imagereward) degrades to a
    # normal 8-GPU run rather than stealing rank N-1 as a server the scorer would reject. This matches
    # launch_one.sh / zimage_table.py, which set NPROC=7 / n_gpus=6 under the SAME (flag AND reward in
    # {hpsv3,deqa}) predicate, so launched world, config batch math, and runtime policy world agree. ---
    _bridge_active = zimage_bridge_enabled() and list(config.reward_fn.keys())[0] in ("hpsv3", "deqa")
    if _bridge_active:
        from diffusionopsd.zimage_reward_bridge import (
            make_bridge_groups, is_server_rank, HeavyRewardServer, bridge_server_devices,
            policy_group as _bridge_policy_group,
        )
        make_bridge_groups(world_size)  # ALL ranks call this (new_group is a world collective)
        if is_server_rank(rank):
            # reward_kind selects the heavy scorer the server loads (hpsv3 / deqa).
            server = HeavyRewardServer(primary_device=bridge_server_devices(local_rank)[0],
                                       reward_kind=list(config.reward_fn.keys())[0])
            server.serve(n_policy=world_size - 1)  # blocks until every policy rank sends CMD_SHUTDOWN
            cleanup_distributed()
            return
        POLICY_GROUP = _bridge_policy_group()   # DDP + all policy collectives use this subgroup
        _server_rank = world_size - 1
        world_size = world_size - 1             # policy world size drives sampler counts + batch math
        logger.info(f"[bridge] policy rank={rank} policy_world_size={world_size} "
                    f"(heavy reward server=rank {_server_rank})")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = (config.run_name + "_" + unique_id) if config.run_name else unique_id

    if is_main_process(rank):
        os.makedirs(config.save_dir, exist_ok=True)
        log_dir = os.path.join(config.logdir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        wandb.init(project="diffusionopsd", name=config.run_name, config=config.to_dict(), dir=log_dir)
        install_wandb_jsonl_tee(wandb, os.path.join(config.save_dir, "metrics.jsonl"))
    logger.info(f"\n{config}")

    # --- Seed policy: DiffusionOPSD requires NO fixed seed (naturally random runs). ---
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

    # ---- Flow-GRPO config parse ----
    fg = config.flowgrpo
    fg_clip_range = float(fg.get("clip_range", 1e-4))
    fg_group_size = int(fg.get("group_size", config.sample.num_image_per_prompt))
    fg_updates_per_rollout = int(fg.get("optimizer_updates_per_rollout", 2))
    fg_noise_level = float(config.sample.noise_level)   # SDE eta for the stochastic rollout
    num_train_batches = (
        int(config.sample.num_batches_per_epoch)
        * int(config.sample.train_batch_size)
        // int(config.train.batch_size)
    )
    if fg_group_size != int(config.sample.num_image_per_prompt):
        raise ValueError(
            f"flowgrpo.group_size ({fg_group_size}) must equal sample.num_image_per_prompt "
            f"({config.sample.num_image_per_prompt}); the K-repeat sampler defines the group.")
    if str(config.sample.solver) != "flow_sde":
        raise ValueError(
            f"Z-Image Flow-GRPO needs sample.solver='flow_sde'; got {config.sample.solver}."
        )
    if bool(config.sample.deterministic):
        raise ValueError("Flow-GRPO needs a STOCHASTIC rollout: set sample.deterministic=False.")
    if fg_noise_level <= 0.0:
        raise ValueError("Flow-GRPO needs a STOCHASTIC rollout: set sample.noise_level (eta) > 0.")
    if not 0.0 < fg_clip_range <= 1e-3:
        raise ValueError(
            "Faithful Flow-GRPO uses a small probability-ratio clip (released presets: "
            f"1e-4 or 1e-5), got {fg_clip_range}."
        )
    if fg_updates_per_rollout != 2:
        raise ValueError(
            "Faithful Flow-GRPO requires exactly two optimizer updates per rollout; "
            f"got {fg_updates_per_rollout}."
        )
    if int(config.train.num_inner_epochs) != 1:
        raise ValueError(
            "Faithful Flow-GRPO uses one replay pass per rollout; "
            f"got train.num_inner_epochs={config.train.num_inner_epochs}."
        )
    if num_train_batches % fg_updates_per_rollout:
        raise ValueError(
            f"{num_train_batches} train batches cannot be split into "
            f"{fg_updates_per_rollout} equal optimizer windows."
        )
    expected_grad_accum = num_train_batches // fg_updates_per_rollout
    if int(config.train.gradient_accumulation_steps) != expected_grad_accum:
        raise ValueError(
            "Flow-GRPO gradient accumulation would make PPO clipping degenerate: "
            f"got {config.train.gradient_accumulation_steps}, expected {expected_grad_accum} "
            f"for {fg_updates_per_rollout} updates over {num_train_batches} train batches."
        )
    # Flow-GRPO scores the rollout with the scalar reward via multi_score (NO reward gradient), the
    # identical forward-only path nft uses -> it supports the SAME reward diagonal nft trains on.
    # Light rewards co-locate on every policy GPU. Heavy rewards reuse nft's mechanisms unchanged:
    #   hpsv3 (7B) / deqa (8B): OOM co-located -> ZIMAGE_HEAVY_BRIDGE reward server (rewards.py routes
    #     these to bridge_reward when zimage_bridge_enabled(); method-agnostic). NPROC=7, policy world 6.
    #   imagereward (BLIP ~400M): co-locates fine on the 6B DiT (NPROC=8, no bridge), heavy PYTHONPATH.
    FLOWGRPO_ZIMAGE_REWARDS = {
        "hpsv2", "pickscore", "clipscore", "aesthetic", "imagereward", "hpsv3", "deqa"
    }
    primary_reward = list(config.reward_fn.keys())[0]
    if primary_reward not in FLOWGRPO_ZIMAGE_REWARDS:
        raise ValueError(
            f"Z-Image Flow-GRPO supports rewards {FLOWGRPO_ZIMAGE_REWARDS}; got "
            f"'{primary_reward}'.")
    if is_main_process(rank):
        logger.info(f"[FlowGRPO] group_size={fg_group_size} clip_range={fg_clip_range} "
                    f"noise_level(eta)={fg_noise_level} kl_beta={config.train.beta} "
                    f"optimizer_updates_per_rollout={fg_updates_per_rollout} "
                    f"grad_accum_batches={expected_grad_accum}")

    if is_main_process(rank):
        run_meta = {
            "run_name": config.run_name,
            "code_variant": os.environ.get("CODE_VARIANT", "flowgrpo_zimage"),
            "seed_policy": ("no_fixed_seed_random_run" if config.seed is None else f"fixed_seed_{config.seed}"),
            "run_random_nonce": run_random_nonce,
            "sample": {"deterministic": False, "solver": "flow_sde", "noise_level": fg_noise_level,
                       "num_steps": int(config.sample.num_steps), "eval_num_steps": int(config.sample.eval_num_steps),
                       "guidance_scale": float(config.sample.guidance_scale),
                       "num_image_per_prompt": int(config.sample.num_image_per_prompt)},
            "flowgrpo": {"group_size": fg_group_size, "clip_range": fg_clip_range,
                         "optimizer_updates_per_rollout": fg_updates_per_rollout,
                         "gradient_accumulation_batches": expected_grad_accum},
            "reward_fn": {k: float(v) for k, v in dict(config.reward_fn).items()},
            "model": config.pretrained.model, "resolution": int(config.resolution), "world_size": world_size,
            "git_commit": os.environ.get("CODE_COMMIT", "unknown"),
            "wall_clock_start": datetime.datetime.now().isoformat(),
        }
        with open(os.path.join(config.save_dir, "run_config.json"), "w") as f:
            json.dump(run_meta, f, indent=2)

    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16
    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=enable_amp)

    # --- Load Z-Image pipeline. Requires diffusers-from-source in the isolated env. ---
    from diffusers import ZImagePipeline
    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32
    pipeline = ZImagePipeline.from_pretrained(config.pretrained.model, torch_dtype=text_encoder_dtype)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)  # Qwen3
    pipeline.transformer.requires_grad_(not config.use_lora)
    try:
        pipeline.set_progress_bar_config(disable=not is_main_process(rank))
    except Exception:
        pass

    pipeline.vae.to(device, dtype=torch.float32)          # FLUX VAE in fp32 (decode stability)
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    transformer = pipeline.transformer.to(device)
    try:
        transformer.enable_gradient_checkpointing()       # spec: required for 6B @ 1024
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
        # Flow-GRPO is on-policy: log_pi_old is stored numerically, so no lagged "old" adapter.
        transformer.set_adapter("default")

    transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank,
                          find_unused_parameters=False, process_group=POLICY_GROUP)  # None => default world
    transformer_ddp.module.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))

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

    first_epoch = 0
    global_step = 0
    if config.resume_from:
        config.resume_from = resolve_resume_checkpoint(config.resume_from)
        lora_path = os.path.join(config.resume_from, "lora")
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
        lora_state = load_peft_weights(lora_path, device=str(device))
        set_peft_model_state_dict(transformer_ddp.module, lora_state, adapter_name="default")
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

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)  # steps to train on

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    # --- §6 profiler derives two optimizer steps per rollout epoch from the config. ---
    prof = profiling.Profiler(config, world_size, rank, device) if profiling.is_enabled() else None
    prof_epoch0 = first_epoch

    for epoch in range(first_epoch, config.num_epochs):
        if prof is not None:
            prof.epoch_begin(epoch - prof_epoch0)
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # =============================== SAMPLING (stochastic rollout, current policy) =============================== #
        pipeline.transformer.eval()
        samples_data_list = []
        epoch_prompt_embeds = []
        epoch_sigmas = None
        images = None
        prompts = None

        for i in tqdm(range(config.sample.num_batches_per_epoch), desc=f"Epoch {epoch}: sampling",
                      disable=not is_main_process(rank), position=0):
            transformer_ddp.module.set_adapter("default")
            if isinstance(train_sampler, DistributedKRepeatSampler):
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

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

            # Stochastic rollout under the CURRENT policy (default adapter) -> stores log_pi_old.
            with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    out = zimage_rollout_with_logprob(
                        pipeline, prompt_embeds_list, num_inference_steps=config.sample.num_steps,
                        height=config.resolution, width=config.resolution, device=device,
                        noise_level=fg_noise_level, guidance_scale=config.sample.guidance_scale, decode=True)

            images = out["images"]
            traj_latents = torch.stack(out["latents"], dim=1)      # (B, num_steps+1, C, H, W)
            traj_logprobs = torch.stack(out["log_probs"], dim=1)   # (B, num_steps)
            if epoch_sigmas is None:
                epoch_sigmas = out["sigmas"].detach()              # (num_steps+1,) shared schedule
                # The PPO loop indexes traj[:, j+1] / logp[:, j] / sigma[j] for j<num_train_timesteps;
                # fail loudly if the ISO-env scheduler produced fewer steps than config.sample.num_steps.
                assert num_train_timesteps <= traj_logprobs.shape[1], (
                    f"num_train_timesteps={num_train_timesteps} exceeds rollout steps="
                    f"{traj_logprobs.shape[1]} (check config.sample.num_steps vs scheduler).")

            base_idx = len(epoch_prompt_embeds)
            embed_idx = torch.arange(base_idx, base_idx + len(prompts), device=device)
            epoch_prompt_embeds.extend([e.detach() for e in prompt_embeds_list])

            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)
            samples_data_list.append({
                "prompt_ids": prompt_ids,
                "embed_idx": embed_idx,
                "traj_latents": traj_latents,
                "traj_logprobs": traj_logprobs,
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
                    pil = Image.fromarray((images_to_log[idx].numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
                wandb.log({"images": [wandb.Image(os.path.join(tmpdir, f"{idx}.jpg"),
                                                  caption=f"{prompts[idx]:.100} | avg: {rewards_to_log[idx]:.2f}")
                                      for idx in range(num_to_log)]}, step=global_step)

        # Group-relative advantage (per-prompt normalization); repeated across trained transitions.
        collated_samples["rewards"]["avg"] = collated_samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)

        gathered_rewards_dict = {}
        for key, value_tensor in collated_samples["rewards"].items():
            gathered_rewards_dict[key] = gather_tensor_to_all(value_tensor, world_size).numpy()

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
                           "mean_reward_100": stat_tracker.get_mean_of_top_rewards(100),
                           "mean_reward_10": stat_tracker.get_mean_of_top_rewards(10)}, step=global_step)
            stat_tracker.clear()
        else:
            avg_rewards_all = gathered_rewards_dict["avg"]
            advantages = (avg_rewards_all - avg_rewards_all.mean()) / (avg_rewards_all.std() + 1e-4)

        samples_per_gpu = collated_samples["traj_logprobs"].shape[0]
        if advantages.ndim == 1:
            advantages = advantages[:, None]
        assert advantages.shape[0] == world_size * samples_per_gpu
        collated_samples["advantages"] = torch.from_numpy(
            advantages.reshape(world_size, samples_per_gpu, -1)[rank]).to(device)

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]

        num_batches = num_train_batches
        filtered_samples = collated_samples
        total_batch_size_filtered = filtered_samples["traj_logprobs"].shape[0]

        # =================================== TRAINING (PPO clip) =================================== #
        transformer_ddp.train()
        effective_grad_accum_steps = config.train.gradient_accumulation_steps * num_train_timesteps
        current_accumulated_steps = 0
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_filtered_samples = {k: v[perm] for k, v in filtered_samples.items()}

            training_batch_size = total_batch_size_filtered // num_batches
            samples_batched_list = []
            for k_batch in range(num_batches):
                start = k_batch * training_batch_size
                end = (k_batch + 1) * training_batch_size
                samples_batched_list.append({key: val[start:end] for key, val in shuffled_filtered_samples.items()})

            info_accumulated = defaultdict(list)
            for i, train_sample_batch in tqdm(list(enumerate(samples_batched_list)),
                                              desc=f"Epoch {epoch}.{inner_epoch}: training",
                                              position=0, disable=not is_main_process(rank)):
                embeds_list = [epoch_prompt_embeds[int(k)] for k in train_sample_batch["embed_idx"].tolist()]
                traj = train_sample_batch["traj_latents"]              # (b, num_steps+1, C, H, W)
                logp_old_all = train_sample_batch["traj_logprobs"]     # (b, num_steps)
                adv_all = train_sample_batch["advantages"].float()     # (b, num_train_timesteps)

                for j_idx in tqdm(range(num_train_timesteps), desc="Timestep", position=1, leave=False,
                                  disable=not is_main_process(rank)):
                    z_t = traj[:, j_idx].float()
                    z_next = traj[:, j_idx + 1].float()
                    logp_old = logp_old_all[:, j_idx].float()
                    sigma_j = epoch_sigmas[j_idx]
                    adv_j = adv_all[:, j_idx] if adv_all.ndim > 1 else adv_all

                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        transformer_ddp.module.set_adapter("default")
                        v_theta = zimage_v(transformer_ddp, z_t, sigma_j, embeds_list)   # policy velocity (grad)

                    # Recompute log pi_new for the SAME stored transition (action = z_next fixed).
                    _znext, _x0, logp_new = flow_grpo_step(
                        model_output=v_theta.float(), latents=z_t, eta=fg_noise_level,
                        sigmas=epoch_sigmas, index=j_idx, prev_sample=z_next)

                    ratio = torch.exp(logp_new - logp_old)
                    adv_clip = torch.clamp(adv_j, -config.train.adv_clip_max, config.train.adv_clip_max)
                    unclipped = -adv_clip * ratio
                    clipped = -adv_clip * torch.clamp(ratio, 1.0 - fg_clip_range, 1.0 + fg_clip_range)
                    policy_loss = torch.max(unclipped, clipped).mean()
                    loss = policy_loss

                    loss_terms = {
                        "policy_loss": policy_loss.detach(),
                        "ratio": ratio.mean().detach(),
                        "clipfrac": (torch.abs(ratio - 1.0) > fg_clip_range).float().mean().detach(),
                        "approx_kl": (0.5 * (logp_new - logp_old).square().mean()).detach(),
                        "clipfrac_gt_one": (ratio - 1.0 > fg_clip_range).float().mean().detach(),
                        "clipfrac_lt_one": (1.0 - ratio > fg_clip_range).float().mean().detach(),
                        "adv_abs": adv_clip.abs().mean().detach(),
                    }

                    # Optional KL-to-frozen-base regularizer (config.train.beta; 0 = pure PPO clip).
                    if config.train.beta > 0:
                        with torch.no_grad(), torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                            with transformer_ddp.module.disable_adapter():
                                v_base = zimage_v(transformer_ddp, z_t, sigma_j, embeds_list)
                            transformer_ddp.module.set_adapter("default")
                        kl_base = ((v_theta.float() - v_base.float()) ** 2).mean(dim=tuple(range(1, z_t.ndim)))
                        loss = loss + config.train.beta * kl_base.mean()
                        loss_terms["kl_base"] = kl_base.mean().detach()
                    loss_terms["total_loss"] = loss.detach()

                    scaled_loss = loss / effective_grad_accum_steps
                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    profiling.train_bwd_inc()  # §6: diffusion (training) backward count
                    current_accumulated_steps += 1
                    for k_info, v_info in loss_terms.items():
                        info_accumulated[k_info].append(v_info)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
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

                if config.train.ema and ema is not None and (current_accumulated_steps % effective_grad_accum_steps == 0):
                    ema.step(transformer_trainable_parameters, global_step)

        expected_epoch_updates = fg_updates_per_rollout
        if gradient_update_times != expected_epoch_updates:
            raise RuntimeError(
                f"Flow-GRPO made {gradient_update_times} optimizer updates in rollout epoch {epoch}; "
                f"expected {expected_epoch_updates}."
            )

        if world_size > 1:
            dist.barrier(group=POLICY_GROUP, device_ids=[torch.cuda.current_device()])

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
    if _bridge_active:
        # Every policy rank tells the heavy-reward server to exit, then all ranks (policy + server)
        # rendezvous on the gloo barrier before tearing down the process groups.
        from diffusionopsd.zimage_reward_bridge import bridge_client_shutdown
        bridge_client_shutdown()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
