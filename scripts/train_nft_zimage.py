# SPDX-License-Identifier: Apache-2.0
#
# DiffusionNFT baseline on Z-Image-Turbo.
#
# Structurally identical to scripts/train_nft_sd3.py (DDP, LoRA old/new adapters, EMA,
# AdamW, eval loop, metrics.jsonl, group-normalized advantage, adaptive weight_factor,
# KL-to-base). The ONLY differences are the model stack + sign/time convention:
#
#   SD3                                   Z-Image-Turbo
#   ---------------------------------     ----------------------------------------------
#   StableDiffusion3Pipeline              ZImagePipeline (diffusers-from-source)
#   3 text encoders + pooled embeds       Qwen3 single encoder -> LIST of (seq_i,2560) embeds
#   pipeline_with_logprob (dpm2)          zimage_rollout (FlowMatchEuler, 9 steps, gs=0)
#   v = transformer(...)                  v = -transformer(...)   (pipeline negates -> zimage_v)
#   x0 = xt - t*v                         x0 = xt - sigma*v_theta  (SAME formula; v_theta=-v_raw)
#   VAE 1.5305/0.0609 (sd3)               VAE 0.3611/0.1159 (FLUX) via zimage_decode
#   LoRA attn.* targets                   LoRA ["to_q","to_k","to_v","to_out.0","w1","w2","w3"]
#
# Objective (per rollout sample, per forward-noised timestep sigma):
#   x_t = (1 - sigma) x_i + sigma * eps ;  x0_pred = x_t - sigma * v_theta ;  regress x_i.
# MIND THE SIGN: zimage_v returns the diffusers-convention velocity (= -raw transformer out),
# so x0_pred = x_t - sigma*v_theta is correct.

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
from diffusionopsd.internvl_bridge import bridge_enabled as internvl_bridge_enabled  # INTERNVL_BRIDGE 2-GPU 26B reward server (gated; forward-only for nft)
from diffusionopsd.stat_tracking import PerPromptStatTracker, calculate_prompt_group_dispersion
from diffusionopsd.diffusers_patch.zimage_pipeline_with_rollout import (
    zimage_encode_prompt, zimage_rollout, zimage_v,
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
config_flags.DEFINE_config_file("config", "config/zimage.py", "Training configuration.")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# ZIMAGE_HEAVY_BRIDGE: NCCL subgroup over policy ranks only (heavy-reward server excluded). None =>
# default world (bridge off), so passing group=POLICY_GROUP everywhere is byte-identical when off.
POLICY_GROUP = None

# LoRA on the S3-DiT transformer only (attention q/k/v/out + gated-FFN w1/w2/w3).
ZIMAGE_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3"]
QWEN_MAX_SEQ_LEN = 512


# =============================== boilerplate (verbatim from SD3) =============================== #
def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    # Bind THIS rank's GPU BEFORE init and pass device_id so NCCL knows the rank->GPU map up front.
    # Under ZIMAGE_HEAVY_BRIDGE the first default-world NCCL op is make_bridge_groups' eager barrier
    # across all 7 ranks (6 policy + server); on the newer Z-Image ISO-env torch a bare init leaves
    # "devices used by this process are currently unknown" and that barrier HANGS the node (observed:
    # all 7 ranks warn then stall, reaped exitcode 1). device_id binds the default communicator to
    # this rank's GPU at init and eliminates the hang. Guarded so an older torch lacking the kwarg
    # (e.g. the SD3 shared env, which never hit the hang) still works unchanged.
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
        line = self.prompts[idx]
        if "\t" in line:  # internvl_dual: "prompt<TAB>ref_path" -> carry the reference in metadata
            prompt, ref_path = line.split("\t", 1)
            return {"prompt": prompt, "metadata": {"ref_path": ref_path}}
        return {"prompt": line, "metadata": {}}

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




# ===================================== Z-Image helpers ===================================== #
@torch.no_grad()
def encode_prompts_list(pipeline, prompts, device):
    """Qwen3 encode -> list of (seq_i, 2560). Deterministic; cached at the epoch level."""
    return zimage_encode_prompt(pipeline, prompts, device, max_sequence_length=QWEN_MAX_SEQ_LEN)


def eval_fn(pipeline, transformer_ddp, test_dataloader, config, device, rank, world_size,
            global_step, reward_fn, executor, mixed_precision_dtype, ema,
            transformer_trainable_parameters):
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    pipeline.transformer.eval()
    transformer_ddp.module.set_adapter("default")  # eval the trained (default) adapter

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
        dist.barrier(group=POLICY_GROUP, device_ids=[torch.cuda.current_device()])  # None => default world; device_ids pins the barrier GPU (newer-torch safe)


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
    setup_distributed(rank, local_rank, world_size)  # inits the default NCCL world over ALL launched ranks
    device = torch.device(f"cuda:{local_rank}")

    # --- ZIMAGE_HEAVY_BRIDGE (gated): last rank is a single-GPU heavy-reward server; ranks 0..N-2
    # are the policy. torchrun launches N=NPROC ranks. With the bridge, rank N-1 hosts the 7B/8B
    # scorer (hpsv3/deqa) and NEVER touches the policy/optimizer/dataloader/checkpointing; the other
    # N-1 ranks do Z-Image DDP training over a policy-only subgroup. Flag off => this block is skipped
    # entirely and the run is byte-identical to the co-located light-reward path. Reward-gated (like
    # flowgrpo/opsd/refl) so a stale ZIMAGE_HEAVY_BRIDGE=1 on an internvl run does NOT also fire this
    # heavy bridge -> the internvl bridge below is the only one; no double make_bridge_groups. ---
    _heavy_bridge_active = zimage_bridge_enabled() and list(config.reward_fn.keys())[0] in ("hpsv3", "deqa")
    if _heavy_bridge_active:
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

    # --- INTERNVL_BRIDGE (gated): the 26B InternVL rewards (internvl_t2i / internvl_dual) run on a
    # dedicated 2-GPU-sharded reward server for nft too (the 26B co-located on the 6B Z-Image DiT OOMs,
    # unlike on the smaller SD3.5-M where nft co-locates it). nft uses the reward FORWARD-ONLY, so the
    # reward_fn (rewards.internvl_*_score) ships the image to the server via remote_reward_scores_forward
    # under INTERNVL_BRIDGE=1 -- no reward-gradient, no trainer ref-plumbing (internvl_dual's ref is
    # loaded in rewards.py from prompt_metadata['ref_path']). Mutually exclusive with ZIMAGE_HEAVY_BRIDGE. ---
    _internvl_bridge_active = internvl_bridge_enabled() and list(config.reward_fn.keys())[0] in ("internvl_t2i", "internvl_dual")
    if _internvl_bridge_active:
        from diffusionopsd.internvl_bridge import (
            make_bridge_groups as _iv_make_groups, is_server_rank as _iv_is_server,
            RewardServer as _IVRewardServer, bridge_server_devices as _iv_server_devices,
            policy_group as _iv_policy_group,
        )
        _iv_make_groups(world_size)  # ALL ranks call this (new_group is a world collective)
        if _iv_is_server(rank):
            _iv_devs = _iv_server_devices(local_rank)   # [6,7] on a real 8-GPU node (2-GPU shard of the 26B)
            server = _IVRewardServer(_iv_devs[0], _iv_devs, reward_kind=list(config.reward_fn.keys())[0])
            server.serve(n_policy=world_size - 1)        # blocks until every policy rank sends CMD_SHUTDOWN
            cleanup_distributed()
            return
        POLICY_GROUP = _iv_policy_group()               # DDP + all policy collectives use this subgroup
        _iv_server_rank = world_size - 1
        world_size = world_size - 1                     # policy world size drives sampler counts + batch math
        logger.info(f"[internvl-bridge] policy rank={rank} policy_world_size={world_size} "
                    f"(internvl-26B reward server=rank {_iv_server_rank})")

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

    if is_main_process(rank):
        run_meta = {
            "run_name": config.run_name,
            "code_variant": os.environ.get("CODE_VARIANT", "diffusionnft_zimage"),
            "seed_policy": ("no_fixed_seed_random_run" if config.seed is None else f"fixed_seed_{config.seed}"),
            "run_random_nonce": run_random_nonce,
            "sample": {"deterministic": bool(config.sample.deterministic), "solver": config.sample.solver,
                       "num_steps": int(config.sample.num_steps), "eval_num_steps": int(config.sample.eval_num_steps),
                       "guidance_scale": float(config.sample.guidance_scale),
                       "num_image_per_prompt": int(config.sample.num_image_per_prompt)},
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
    # [SMOKE S2] `from diffusers import ZImagePipeline` must work; native sample OK.
    from diffusers import ZImagePipeline
    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32
    # 6B S3-DiT @ 1024^2: load weights in the mixed dtype (fp16) to fit; VAE kept fp32.
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
        # [SMOKE S5] verify the 7 target names resolve against transformer.named_modules().
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

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)  # int(9*0.99)=8

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
        epoch_prompt_embeds = []   # rank-local list of per-sample (seq_i,2560) Qwen3 embeds
        images = None
        prompts = None

        for i in tqdm(range(config.sample.num_batches_per_epoch), desc=f"Epoch {epoch}: sampling",
                      disable=not is_main_process(rank), position=0):
            transformer_ddp.module.set_adapter("default")
            if isinstance(train_sampler, DistributedKRepeatSampler):
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds_list = encode_prompts_list(pipeline, prompts, device)
            # token ids (Qwen2Tokenizer) only for per-prompt stat grouping across ranks.
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
                    out = zimage_rollout(
                        pipeline, prompt_embeds_list, num_inference_steps=config.sample.num_steps,
                        height=config.resolution, width=config.resolution, device=device,
                        guidance_scale=config.sample.guidance_scale, decode=True)
            transformer_ddp.module.set_adapter("default")

            images = out["images"]
            x0 = out["x0"]                                             # (B,C,H,W) clean endpoint
            timesteps = out["timesteps"].repeat(len(prompts), 1).to(device)   # (B, num_steps)

            base_idx = len(epoch_prompt_embeds)
            embed_idx = torch.arange(base_idx, base_idx + len(prompts), device=device)
            epoch_prompt_embeds.extend([e.detach() for e in prompt_embeds_list])

            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)
            samples_data_list.append({
                "prompt_ids": prompt_ids,
                "embed_idx": embed_idx,
                "timesteps": timesteps,
                "next_timesteps": torch.concatenate([timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1),
                "latents_clean": x0,
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

        samples_per_gpu = collated_samples["timesteps"].shape[0]
        if advantages.ndim == 1:
            advantages = advantages[:, None]
        assert advantages.shape[0] == world_size * samples_per_gpu
        collated_samples["advantages"] = torch.from_numpy(
            advantages.reshape(world_size, samples_per_gpu, -1)[rank]).to(device)

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]

        num_batches = config.sample.num_batches_per_epoch * config.sample.train_batch_size // config.train.batch_size
        filtered_samples = collated_samples
        total_batch_size_filtered, num_timesteps_filtered = filtered_samples["timesteps"].shape

        # =================================== TRAINING =================================== #
        transformer_ddp.train()
        effective_grad_accum_steps = config.train.gradient_accumulation_steps * num_train_timesteps
        current_accumulated_steps = 0
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_filtered_samples = {k: v[perm] for k, v in filtered_samples.items()}
            perms_time = torch.stack(
                [torch.randperm(num_timesteps_filtered, device=device) for _ in range(total_batch_size_filtered)])
            for key in ["timesteps", "next_timesteps"]:
                shuffled_filtered_samples[key] = shuffled_filtered_samples[key][
                    torch.arange(total_batch_size_filtered, device=device)[:, None], perms_time]

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

                for j_idx in tqdm(range(num_train_timesteps), desc="Timestep", position=1, leave=False,
                                  disable=not is_main_process(rank)):
                    x0 = train_sample_batch["latents_clean"].float()
                    sigma = train_sample_batch["timesteps"][:, j_idx].float() / 1000.0   # (B,) per-sample sigma
                    sig_e = sigma.view(-1, *([1] * (x0.ndim - 1)))
                    noise = torch.randn_like(x0)
                    xt = (1 - sig_e) * x0 + sig_e * noise                                # forward-noise the endpoint

                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        transformer_ddp.module.set_adapter("old")
                        with torch.no_grad():
                            old_prediction = zimage_v(transformer_ddp, xt, sigma, embeds_list).detach()
                        transformer_ddp.module.set_adapter("default")
                        forward_prediction = zimage_v(transformer_ddp, xt, sigma, embeds_list)   # v_theta (grad)
                        with torch.no_grad():
                            with transformer_ddp.module.disable_adapter():
                                ref_forward_prediction = zimage_v(transformer_ddp, xt, sigma, embeds_list)
                            transformer_ddp.module.set_adapter("default")

                    loss_terms = {}
                    advantages_clip = torch.clamp(train_sample_batch["advantages"][:, j_idx],
                                                  -config.train.adv_clip_max, config.train.adv_clip_max)
                    # adv_mode=all -> no clipping to a single sign (full pos+neg branch).
                    normalized_advantages_clip = (advantages_clip / config.train.adv_clip_max) / 2.0 + 0.5
                    r = torch.clamp(normalized_advantages_clip, 0, 1)

                    loss_terms["x0_norm"] = torch.mean(x0**2).detach()
                    loss_terms["old_deviate"] = torch.mean((forward_prediction - old_prediction) ** 2).detach()
                    positive_prediction = config.beta * forward_prediction + (1 - config.beta) * old_prediction.detach()
                    implicit_negative_prediction = (1.0 + config.beta) * old_prediction.detach() - config.beta * forward_prediction

                    x0_prediction = xt - sig_e * positive_prediction                     # SIGN: v_theta=-v_raw
                    with torch.no_grad():
                        weight_factor = (torch.abs(x0_prediction.double() - x0.double())
                                         .mean(dim=tuple(range(1, x0.ndim)), keepdim=True).clip(min=0.00001))
                    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(dim=tuple(range(1, x0.ndim)))
                    negative_x0_prediction = xt - sig_e * implicit_negative_prediction
                    with torch.no_grad():
                        negative_weight_factor = (torch.abs(negative_x0_prediction.double() - x0.double())
                                                  .mean(dim=tuple(range(1, x0.ndim)), keepdim=True).clip(min=0.00001))
                    negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight_factor).mean(
                        dim=tuple(range(1, x0.ndim)))

                    ori_policy_loss = r * positive_loss / config.beta + (1.0 - r) * negative_loss / config.beta
                    policy_loss = (ori_policy_loss * config.train.adv_clip_max).mean()
                    loss = policy_loss
                    loss_terms["policy_loss"] = policy_loss.detach()

                    kl_div_loss = ((forward_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    loss += config.train.beta * torch.mean(kl_div_loss)
                    loss_terms["kl_div_loss"] = torch.mean(kl_div_loss).detach()
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

        if world_size > 1:
            dist.barrier(group=POLICY_GROUP, device_ids=[torch.cuda.current_device()])  # None => default world; device_ids pins the barrier GPU (newer-torch safe)
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
        dist.barrier(group=POLICY_GROUP, device_ids=[torch.cuda.current_device()])  # None => default world; device_ids pins the barrier GPU (newer-torch safe)

    if is_main_process(rank):
        try:
            with open(os.path.join(config.save_dir, "run_done.json"), "w") as f:
                json.dump({"wall_clock_end": datetime.datetime.now().isoformat(), "global_step": global_step}, f, indent=2)
        except Exception:
            pass
        wandb.finish()
    if _heavy_bridge_active:
        # Every policy rank tells the heavy-reward server to exit, then all ranks (policy + server)
        # rendezvous on the gloo barrier before tearing down the process groups.
        from diffusionopsd.zimage_reward_bridge import bridge_client_shutdown
        bridge_client_shutdown()
    if _internvl_bridge_active:
        # Same handshake for the 26B internvl reward server (only policy ranks reach here).
        from diffusionopsd.internvl_bridge import bridge_client_shutdown as _iv_client_shutdown
        _iv_client_shutdown()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
