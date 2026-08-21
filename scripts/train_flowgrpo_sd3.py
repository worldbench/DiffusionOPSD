# SPDX-License-Identifier: Apache-2.0
#
# Flow-GRPO baseline on SD3.5-Medium — matched local control and efficiency profiler.
#
# This is a MINIMAL, faithful Flow-GRPO compute graph. It was FIRST written for the §6 efficiency
# profiler (which measures peak-VRAM / time / optimizer-step — properties of the graph SHAPE:
# rollout + reward + advantage + PPO-clip backward), and PROFILE=1 still restricts it to a handful
# of steps with checkpointing disabled.
#
# The paper's SD3.5-M main-table FlowGRPO reference row retains its released checkpoint;
# this trainer supplies the locally matched control used for efficiency analysis. Without
# PROFILE, the full path runs: per-epoch save_ckpt at save_freq, checkpoint resume,
# metrics.jsonl, run_done.json — the same scaffolding as train_nft_sd3.py. It is NOT separately
# convergence-tuned: optimizer/LR/LoRA/reward come from the shared frozen main-table config
# and the only Flow-GRPO-specific knobs are the SDE rollout
# (solver='flow', deterministic=False, noise_level=eta) and the PPO clip range.
#
# It ports the model-agnostic Flow-GRPO pieces of ``train_flowgrpo_zimage.py`` (PerPromptStatTracker
# group-relative advantage + PPO-clip loss on per-step SDE log-probs) onto the SD3.5-M pipeline / LoRA
# / DDP / AdamW / reward scaffolding of ``train_nft_sd3.py``, and adds the same ``diffusionopsd.profiling``
# harness that ``train_nft_sd3.py`` uses.
#
#   Algorithm (per rollout epoch, with two optimizer updates as released FlowGRPO):
#     1. STOCHASTIC (SDE) rollout: pipeline_with_logprob(solver='flow', deterministic=False,
#        noise_level=eta>0) runs the 10-step flow schedule and returns the full latent trajectory
#        z_0..z_T PLUS the per-step log-prob log_pi_old of each sampled SDE transition (the 3rd return
#        that DiffusionNFT discards). The behaviour policy is the CURRENT policy (default LoRA
#        adapter) — Flow-GRPO is on-policy, so there is NO lagged "old" adapter (unlike NFT).
#     2. Reward + advantage: score the decoded image, then group-normalize WITHIN each K-repeat prompt
#        group to advantages a = (r - mean_group)/(std_group + eps) via PerPromptStatTracker.
#     3. PPO update: at every stored transition (z_t -> z_{t+1}), recompute the SD3 velocity v_theta
#        (guidance_scale=1.0 => single CFG-free forward, exactly the training-time forward of
#        train_nft_sd3.py), recompute log_pi_new via flow_grpo_step (prev_sample = the FIXED stored
#        z_{t+1}), form ratio = exp(log_pi_new - log_pi_old), and minimize the clipped PPO surrogate
#        -min(ratio*a, clip(ratio, 1-eps, 1+eps)*a). An optional KL-to-frozen-base term (config.train.beta;
#        via disable_adapter()) mirrors the repo's KL convention.
#
# CFG-free assumption: the profiled config sets guidance_scale=1.0, so the rollout does a single
# conditional forward (do_classifier_free_guidance is False) and the recompute mirrors it exactly.
# This keeps log_pi_old and log_pi_new comparable (the on-policy PPO invariant). Enforced below.

from collections import defaultdict
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
import logging
from diffusers import StableDiffusion3Pipeline
import numpy as np
import diffusionopsd.rewards
from diffusionopsd import profiling  # paper efficiency-profiling harness (env PROFILE=1)
from diffusionopsd.stat_tracking import PerPromptStatTracker, calculate_prompt_group_dispersion
from diffusionopsd.metrics import install_wandb_jsonl_tee
from diffusionopsd.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from diffusionopsd.diffusers_patch.solver import flow_grpo_step
from diffusionopsd.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# SD3.5-M attention LoRA targets (verbatim from train_nft_sd3.py).
SD3_LORA_TARGETS = [
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v",
]


# =============================== boilerplate (verbatim from the SD3 trainers) =============================== #
def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(lock_rank)


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
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds




def sd3_velocity(transformer_ddp, latents, sigma, prompt_embeds, pooled_prompt_embeds):
    """CFG-free (guidance_scale=1.0) SD3 velocity, mirroring pipeline_with_logprob.v_pred_fn.

    The rollout used a SINGLE conditional forward at ``timestep = sigma*1000`` (do_classifier_free_
    guidance is False when guidance_scale==1.0), so the on-policy recompute must use the identical
    construction for log pi_new to be comparable with the stored log pi_old. ``transformer_ddp`` is
    called (not ``.module``) so DDP gradient all-reduce fires; the active adapter is set by the caller.
    """
    timesteps = torch.full([latents.shape[0]], sigma * 1000, device=latents.device, dtype=torch.long)
    v = transformer_ddp(
        hidden_states=latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        return_dict=False,
    )[0]
    return v


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
    config = FLAGS.config
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

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
            dist.broadcast(nonce_tensor, src=0)
        run_random_nonce = int(nonce_tensor.item())
        logger.info(f"[seed] NO fixed seed; run_random_nonce={run_random_nonce} (prompt grouping only)")

    # ---- Flow-GRPO config parse (mirrors train_flowgrpo_zimage.py) ----
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
    if str(config.sample.solver) != "flow":
        raise ValueError(f"Flow-GRPO needs the SDE flow solver; got sample.solver={config.sample.solver}.")
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
    if float(config.sample.guidance_scale) != 1.0:
        raise ValueError(
            "This §6 Flow-GRPO profiler assumes CFG-free (guidance_scale=1.0) rollout+recompute; "
            f"got guidance_scale={config.sample.guidance_scale}.")
    if is_main_process(rank):
        logger.info(f"[FlowGRPO] group_size={fg_group_size} clip_range={fg_clip_range} "
                    f"noise_level(eta)={fg_noise_level} kl_beta={config.train.beta} "
                    f"optimizer_updates_per_rollout={fg_updates_per_rollout} "
                    f"grad_accum_batches={expected_grad_accum}")

    if is_main_process(rank):
        run_meta = {
            "run_name": config.run_name,
            "code_variant": os.environ.get("CODE_VARIANT", "flowgrpo_sd3"),
            "seed_policy": ("no_fixed_seed_random_run" if config.seed is None else f"fixed_seed_{config.seed}"),
            "run_random_nonce": run_random_nonce,
            "sample": {"deterministic": bool(config.sample.deterministic), "solver": str(config.sample.solver),
                       "noise_level": fg_noise_level, "num_steps": int(config.sample.num_steps),
                       "eval_num_steps": int(config.sample.eval_num_steps),
                       "guidance_scale": float(config.sample.guidance_scale),
                       "num_image_per_prompt": int(config.sample.num_image_per_prompt)},
            "flowgrpo": {"group_size": fg_group_size, "clip_range": fg_clip_range,
                         "optimizer_updates_per_rollout": fg_updates_per_rollout,
                         "gradient_accumulation_batches": expected_grad_accum,
                         "kl_beta": float(config.train.beta)},
            "reward_fn": {k: float(v) for k, v in dict(config.reward_fn).items()},
            "model": config.pretrained.model, "resolution": int(config.resolution), "world_size": world_size,
            "git_commit": os.environ.get("CODE_COMMIT", "unknown"),
            "wall_clock_start": datetime.datetime.now().isoformat(),
        }
        with open(os.path.join(config.save_dir, "run_config.json"), "w") as f:
            json.dump(run_meta, f, indent=2)

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16
    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=enable_amp)

    # --- Load pipeline and models (verbatim from train_nft_sd3.py) ---
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1, disable=not is_main_process(rank), leave=False, desc="Timestep", dynamic_ncols=True,
    )

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32
    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)

    # Env-gated SD3 transformer gradient-checkpointing (frees GBs for memory-tight rewards).
    if os.environ.get("SD3_TRANSFORMER_GRADCKPT", "0") == "1":
        try:
            pipeline.transformer.enable_gradient_checkpointing()
            if is_main_process(rank):
                logger.info("[mem] SD3 transformer gradient_checkpointing ENABLED (SD3_TRANSFORMER_GRADCKPT=1)")
        except Exception as _e:
            logger.warning(f"pipeline.transformer.enable_gradient_checkpointing() unavailable: {_e}")

    transformer = pipeline.transformer.to(device)

    if config.use_lora:
        transformer_lora_config = LoraConfig(
            r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=SD3_LORA_TARGETS)
        if config.train.lora_path:
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, transformer_lora_config)
        # Flow-GRPO is on-policy: log_pi_old is stored numerically, so no lagged "old" adapter (unlike NFT).
        transformer.set_adapter("default")

    transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank,
                          find_unused_parameters=False)
    transformer_ddp.module.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        transformer_trainable_parameters, lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay, eps=config.train.adam_epsilon)

    # --- Dataset / dataloader (Flow-GRPO K-repeat grouping) ---
    if config.prompt_fn != "general_ocr":
        raise NotImplementedError(f"Flow-GRPO SD3 profiler only supports general_ocr prompts; got {config.prompt_fn}.")
    train_dataset = TextPromptDataset(config.dataset, "train")
    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset, batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt, num_replicas=world_size, rank=rank, seed=run_random_nonce)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0,
                                  collate_fn=train_dataset.collate_fn, pin_memory=True)

    # --- Negative embeddings (unused at guidance_scale=1.0, but pipeline_with_logprob expects them) ---
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device)
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)

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
    # Enable per-optimizer-step timing/counting, wrap the rollout reward scorer so each call counts as
    # one reward forward, and force debug mode so periodic/final save_ckpt is skipped (no checkpoint
    # spam, no 100-epoch training).
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

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)  # SDE steps trained on

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    # --- §6 profiler derives two optimizer steps per rollout epoch from the config. ---
    prof = profiling.Profiler(config, world_size, rank, device) if profiling.is_enabled() else None
    prof_epoch0 = first_epoch

    def _profile_sanity_eval():
        return profiling.run_sanity_eval(
            pipeline=pipeline, reward_fn=eval_reward_fn, compute_text_embeddings=compute_text_embeddings,
            text_encoders=text_encoders, tokenizers=tokenizers, config=config, device=device,
            rank=rank, world_size=world_size,
        )

    for epoch in range(first_epoch, config.num_epochs):
        if prof is not None:
            prof.epoch_begin(epoch - prof_epoch0)
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # =============================== SAMPLING (stochastic SDE rollout, current policy) ============= #
        pipeline.transformer.eval()
        samples_data_list = []
        epoch_sigmas = None
        images = None
        prompts = None

        for i in tqdm(range(config.sample.num_batches_per_epoch), desc=f"Epoch {epoch}: sampling",
                      disable=not is_main_process(rank), position=0):
            transformer_ddp.module.set_adapter("default")
            if isinstance(train_sampler, DistributedKRepeatSampler):
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=128, device=device)
            prompt_ids = tokenizers[0](
                prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
            ).input_ids.to(device)

            if (
                i == 0
                and global_step > 0
                and global_step % config.save_freq == 0
                and is_main_process(rank)
                and not config.debug
            ):
                save_ckpt(config.save_dir, transformer_ddp, global_step, rank, ema,
                          transformer_trainable_parameters, config, optimizer, scaler, epoch_completed=epoch)

            # SDE rollout under the CURRENT policy (default adapter) -> keep all_log_probs (= log_pi_old).
            with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    images, all_latents, all_log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds[: len(prompts)],
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[: len(prompts)],
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,   # 1.0 -> single CFG-free forward
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=fg_noise_level,                    # SDE eta > 0
                        deterministic=config.sample.deterministic,     # False
                        solver=config.sample.solver,                   # "flow"
                        model_type="sd3",
                    )

            traj_latents = torch.stack(all_latents, dim=1)     # (B, num_steps+1, C, H, W)  z_0 .. z_T
            traj_logprobs = torch.stack(all_log_probs, dim=1)  # (B, num_steps)              log_pi_old
            if epoch_sigmas is None:
                # Same sigma schedule run_sampling used (scheduler.sigmas after set_timesteps); shared
                # across all batches of this epoch (retrieve_timesteps is deterministic for fixed num_steps).
                epoch_sigmas = pipeline.scheduler.sigmas.detach().float().to(device)  # (num_steps+1,)
                assert num_train_timesteps <= traj_logprobs.shape[1], (
                    f"num_train_timesteps={num_train_timesteps} exceeds rollout steps="
                    f"{traj_logprobs.shape[1]} (check config.sample.num_steps vs scheduler).")

            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)
            samples_data_list.append({
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
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

        # Image logging (main process); skipped in debug/profile mode to keep the §6 timing window clean.
        if epoch % 10 == 0 and is_main_process(rank) and not config.debug:
            images_to_log = images.cpu()
            rewards_to_log = collated_samples["rewards"]["avg"][-len(images_to_log):].cpu()
            with tempfile.TemporaryDirectory() as tmpdir:
                num_to_log = min(15, len(images_to_log))
                for idx in range(num_to_log):
                    pil = Image.fromarray((images_to_log[idx].numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
                wandb.log({"images": [wandb.Image(os.path.join(tmpdir, f"{idx}.jpg"),
                                                  caption=f"{prompts[idx]:.100} | avg: {rewards_to_log[idx]:.2f}")
                                      for idx in range(num_to_log)]}, step=global_step)

        # ------------------ Group-relative advantage (per-prompt normalization) ------------------ #
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
        # Total number of backward passes before an optimizer step (one per (micro-batch, timestep)).
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
                embeds = train_sample_batch["prompt_embeds"]
                pooled_embeds = train_sample_batch["pooled_prompt_embeds"]
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
                        v_theta = sd3_velocity(transformer_ddp, z_t, sigma_j, embeds, pooled_embeds)  # policy velocity (grad)

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
                                v_base = sd3_velocity(transformer_ddp, z_t, sigma_j, embeds, pooled_embeds)
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
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        gradient_update_times += 1
                        optimizer.zero_grad()
                        log_info = {k: torch.mean(torch.stack(v)).item() for k, v in info_accumulated.items()}
                        info_tensor = torch.tensor([log_info[k] for k in sorted(log_info)], device=device)
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG)
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
            dist.barrier()

        if prof is not None:
            prof.epoch_end(epoch - prof_epoch0, global_step=global_step)
            if prof.done(epoch - prof_epoch0):
                prof.finalize(_profile_sanity_eval)
                break

    if prof is not None and not prof.finalized:  # safety net if num_epochs < warmup+measure
        prof.finalize(_profile_sanity_eval)

    if not config.debug:
        save_ckpt(config.save_dir, transformer_ddp, global_step, rank, ema,
                  transformer_trainable_parameters, config, optimizer, scaler,
                  epoch_completed=config.num_epochs)
    if world_size > 1:
        dist.barrier()

    if is_main_process(rank):
        try:
            with open(os.path.join(config.save_dir, "run_done.json"), "w") as f:
                json.dump({"wall_clock_end": datetime.datetime.now().isoformat(), "global_step": global_step}, f, indent=2)
        except Exception:
            pass
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
