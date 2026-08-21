"""Shared machinery for the OPD-family teacher-distillation trainers.

Design notes
------------
* Self-contained OPD LOGIC (setup, teacher loading, rollout wrappers, distillation-loss helpers,
  benchmark timer) lives here. The heavy, proven diffusion primitives (``pipeline_with_logprob``,
  ``flow_grpo_step``, ``encode_prompt``, ``EMAModuleWrapper``, checkpoint IO) are
  imported from the public release's shared ``diffusionopsd`` package.
* Teachers = frozen DiffusionOPSD ck100 LoRA checkpoints loaded as EXTRA PEFT adapters on the SAME base
  transformer as the student. student adapter = "default" (trainable). teacher adapters =
  "teacher_0/1/2" (frozen). To get a teacher velocity we switch the active adapter under no_grad;
  the student velocity uses the DDP-wrapped module so grad all-reduce fires. One base model in
  memory + 4 lightweight adapters (vs 4 full transformers). Teacher forwards never touch DDP (no_grad),
  so ``find_unused_parameters=False`` stays valid (only the trainable "default" params sync).
"""

import os
import json
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, Sampler
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model

# --- shared DiffusionOPSD primitives ---
from diffusionopsd.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from diffusionopsd.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from diffusionopsd.ema import EMAModuleWrapper
from diffusionopsd.experiment_io import (
    resolve_resume_checkpoint, resume_position, restore_ema_and_rng, save_trainer_state,
)

SD3_LORA_TARGETS = [
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v",
]


# =============================== boilerplate (copied verbatim from the SD3 trainers) =============================== #
def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        with open(os.path.join(dataset, f"{split}.txt")) as f:
            self.prompts = [ln.strip() for ln in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        return [e["prompt"] for e in examples], [e["metadata"] for e in examples]


class DistributedKRepeatSampler(Sampler):
    """K-repeat prompt sampler (verbatim from train_flowgrpo_sd3.py). k=1 => plain per-GPU batch."""
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset; self.batch_size = batch_size; self.k = k
        self.num_replicas = num_replicas; self.rank = rank; self.seed = seed
        self.total_samples = num_replicas * batch_size
        assert self.total_samples % k == 0, f"k({k}) must divide num_replicas*batch_size"
        self.m = self.total_samples // k; self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator(); g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated = [idx for idx in indices for _ in range(self.k)]
            shuf = torch.randperm(len(repeated), generator=g).tolist()
            samples = [repeated[i] for i in shuf]
            per_card = [samples[i * self.batch_size:(i + 1) * self.batch_size] for i in range(self.num_replicas)]
            yield per_card[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        pe, ppe = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        return pe.to(device), ppe.to(device)


def sd3_velocity(transformer, latents, sigma, prompt_embeds, pooled_prompt_embeds):
    """CFG-free SD3 velocity at timestep sigma*1000 (verbatim primitive from train_flowgrpo_sd3.py).
    Pass the DDP wrapper for the STUDENT (grad all-reduce); pass ``.module`` for a TEACHER (no_grad)."""
    timesteps = torch.full([latents.shape[0]], sigma * 1000, device=latents.device, dtype=torch.long)
    return transformer(
        hidden_states=latents, timestep=timesteps,
        encoder_hidden_states=prompt_embeds, pooled_projections=pooled_prompt_embeds,
        return_dict=False,
    )[0]


def flow_transition_mean(latents, velocity, sigmas, index, eta=0.0):
    """One flow step's TRANSITION MEAN — mirrors flow_grpo_step's ``prev_sample_mean`` EXACTLY
    (src/diffusionopsd/diffusers_patch/solver.py L103-106). eta=0 => deterministic Euler mean
    ``latents + velocity*dt``. DiffusionOPD distills THIS (the transition mean), not the raw
    velocity: since mean_s - mean_t = (v_s - v_t)*dt, matching the mean applies the physically
    correct per-step ``dt**2`` weighting (velocity-MSE weights every step uniformly, which is wrong)."""
    sigma = sigmas[index]
    sigma_prev = sigmas[index + 1]
    sigma_max = sigmas[1]
    dt = sigma_prev - sigma
    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * eta
    return (latents * (1 + std_dev_t ** 2 / (2 * sigma) * dt)
            + velocity * (1 + std_dev_t ** 2 * (1 - sigma) / (2 * sigma)) * dt)


def global_stepwise_advantage(reward, world_size, eps=1e-4):
    """Normalize a ``(B, ...)`` per-step reward over the GLOBAL batch (all ``world_size`` GPUs), matching
    the reference trainer's global reward-normalize (train_flowgrpo_sd3.py `gather_tensor_to_all` +
    `(r-mean)/(std+eps)`). all-gathers the reward across ranks, computes mean/std per column over the
    ``world_size*B`` global samples, then normalizes the LOCAL reward with those global stats — much
    lower variance than per-GPU (B=4) normalization. Returns the local ``(B, ...)`` advantage."""
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        gathered = [torch.zeros_like(reward) for _ in range(world_size)]
        dist.all_gather(gathered, reward.contiguous())
        allr = torch.cat(gathered, dim=0)          # (world_size*B, ...) global batch
    else:
        allr = reward
    mean = allr.mean(dim=0, keepdim=True)
    std = allr.std(dim=0, keepdim=True)
    return (reward - mean) / (std + eps)


# =============================== OPD model build (student + teacher adapters) =============================== #
def build_models(config, device, rank, local_rank):
    """Returns (pipeline, transformer_ddp, teacher_names, text_encoders, tokenizers, optimizer, ema,
    scaler, trainable_params, mp_dtype). Student = 'default' adapter (trainable); teachers frozen."""
    mp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "no": None}[config.mixed_precision]
    enable_amp = mp_dtype is not None
    scaler = GradScaler(enabled=enable_amp)

    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    for te in (pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3):
        te.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(disable=True)

    te_dtype = mp_dtype if enable_amp else torch.float32
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=te_dtype)
    pipeline.text_encoder_2.to(device, dtype=te_dtype)
    pipeline.text_encoder_3.to(device, dtype=te_dtype)
    if os.environ.get("SD3_TRANSFORMER_GRADCKPT", "0") == "1":
        try:
            pipeline.transformer.enable_gradient_checkpointing()
        except Exception:
            pass
    transformer = pipeline.transformer.to(device)

    # student LoRA (adapter "default", trainable)
    lora_cfg = LoraConfig(r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=SD3_LORA_TARGETS)
    transformer = get_peft_model(transformer, lora_cfg)
    if config.train.lora_path:  # warm-start student
        transformer.load_adapter(config.train.lora_path, adapter_name="default", is_trainable=True)

    # Teacher adapters are frozen. Enforce exactly N (Open3 = 3). Only the explicit literal
    # "DUMMY" makes a fresh random LoRA (benchmark timing, where teacher IDENTITY is irrelevant). ANY
    # other path MUST be a real LoRA dir (…/checkpoint-N/lora with adapter_config.json) or we HARD-FAIL —
    # never silently substitute a random teacher (a wrong path like …/checkpoint-N without /lora errors).
    teacher_paths = [str(p).rstrip("/") for p in config.opd.teacher_loras]
    n_req = int(os.environ.get("OPD_N_TEACHERS", "3"))
    if len(teacher_paths) != n_req:
        raise ValueError(
            f"OPD needs EXACTLY {n_req} teacher LoRA dirs (got {len(teacher_paths)}: {teacher_paths}). "
            f"Set OPD_TEACHER_LORAS='…/checkpoint-N/lora,…/checkpoint-N/lora,…/checkpoint-N/lora'. "
            f"If temporarily using ONE teacher (e.g. a DiffusionOPSD ckpt), repeat it 3× to keep the "
            f"3-teacher cost/structure. (Override count only via OPD_N_TEACHERS.)")
    teacher_names = []
    for i, tpath in enumerate(teacher_paths):
        name = f"teacher_{i}"
        if tpath == "DUMMY":
            transformer.add_adapter(name, lora_cfg)   # EXPLICIT benchmark-only random dummy
            if is_main_process(rank):
                print(f"[opd] teacher {i} = explicit DUMMY random adapter (BENCHMARK TIMING ONLY — not a real teacher)", flush=True)
        else:
            if not os.path.isfile(os.path.join(tpath, "adapter_config.json")):
                raise FileNotFoundError(
                    f"teacher {i}: no adapter_config.json under '{tpath}'. Pass the LoRA dir itself "
                    f"(…/checkpoint-N/lora), NOT the checkpoint dir. Refusing to silently use a random teacher.")
            transformer.load_adapter(tpath, adapter_name=name, is_trainable=False)
            if is_main_process(rank):
                print(f"[opd] teacher {i} loaded: {tpath}", flush=True)
        teacher_names.append(name)
    # freeze every non-default adapter param explicitly (belt-and-suspenders)
    for n, p in transformer.named_parameters():
        if any(f".{tn}." in n or n.endswith(tn) for tn in teacher_names):
            p.requires_grad_(False)
    transformer.set_adapter("default")

    transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank,
                          find_unused_parameters=False)
    transformer_ddp.module.set_adapter("default")
    trainable = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    optimizer = torch.optim.AdamW(
        trainable, lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay, eps=config.train.adam_epsilon)
    ema = EMAModuleWrapper(trainable, decay=0.9, update_step_interval=1, device=device) if config.train.ema else None

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    if is_main_process(rank):
        print(f"[opd] built student+{len(teacher_names)} teacher adapters; trainable tensors={len(trainable)}", flush=True)
    return (pipeline, transformer_ddp, teacher_names, text_encoders, tokenizers,
            optimizer, ema, scaler, trainable, mp_dtype)


def rollout(config, pipeline, transformer_ddp, prompt_embeds, pooled_prompt_embeds,
            neg_embeds, neg_pooled, device, mp_dtype):
    """One student rollout. Deterministic dpm2 (Dance/Diffusion) or stochastic SDE flow (FlowOPD).
    Returns images (B,C,H,W in [0,1]), traj_latents (B, steps+1, ...), sigmas (steps+1,),
    traj_logprobs (B, steps) — logprobs meaningful only for the SDE solver."""
    enable_amp = mp_dtype is not None
    transformer_ddp.module.set_adapter("default")
    pipeline.transformer.eval()
    with torch.cuda.amp.autocast(enabled=enable_amp, dtype=mp_dtype):
        with torch.no_grad():
            images, all_latents, all_log_probs = pipeline_with_logprob(
                pipeline,
                prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=neg_embeds, negative_pooled_prompt_embeds=neg_pooled,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                output_type="pt", height=config.resolution, width=config.resolution,
                noise_level=float(config.sample.noise_level),
                deterministic=bool(config.sample.deterministic),
                solver=str(config.sample.solver), model_type="sd3",
            )
    traj_latents = torch.stack(all_latents, dim=1)     # (B, steps+1, C, H, W)
    # deterministic (dpm2) rollout has no per-step log-probs -> list of None; only the SDE (flow)
    # rollout returns real log-probs (used by FlowOPD). Guard the stack.
    traj_logprobs = (torch.stack(all_log_probs, dim=1)
                     if (len(all_log_probs) and all_log_probs[0] is not None) else None)
    sigmas = pipeline.scheduler.sigmas.detach().float().to(device)  # (steps+1,)
    return images, traj_latents, sigmas, traj_logprobs


def teacher_velocities(transformer_ddp, teacher_names, latents, sigma, prompt_embeds, pooled):
    """Frozen teacher velocities at (latents, sigma), each via its adapter under no_grad.
    Restores the student 'default' adapter before returning. Returns list[tensor] (detached)."""
    outs = []
    with torch.no_grad():
        for name in teacher_names:
            transformer_ddp.module.set_adapter(name)
            outs.append(sd3_velocity(transformer_ddp.module, latents, sigma, prompt_embeds, pooled).detach())
    transformer_ddp.module.set_adapter("default")
    return outs


def student_velocity(transformer_ddp, latents, sigma, prompt_embeds, pooled):
    """Student velocity WITH grad (default adapter, DDP wrapper -> grad all-reduce)."""
    transformer_ddp.module.set_adapter("default")
    return sd3_velocity(transformer_ddp, latents, sigma, prompt_embeds, pooled)


def save_ckpt(save_dir, transformer_ddp, global_step, rank, ema, trainable, config, optimizer, scaler,
              epoch_completed=None):
    """Save checkpoint-<step>/{lora, optimizer.pt, scaler.pt, trainer_state} (EMA-aware)."""
    if not is_main_process(rank):
        return None
    root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    lora_root = os.path.join(root, "lora"); os.makedirs(lora_root, exist_ok=True)
    if config.train.ema and ema is not None:
        ema.copy_ema_to(trainable, store_temp=True)
    # save ONLY the student "default" adapter (teachers are external refs, not ours to re-emit)
    transformer_ddp.module.save_pretrained(lora_root, selected_adapters=["default"])
    torch.save(optimizer.state_dict(), os.path.join(root, "optimizer.pt"))
    if scaler is not None:
        torch.save(scaler.state_dict(), os.path.join(root, "scaler.pt"))
    save_trainer_state(root, epoch_completed=(global_step if epoch_completed is None else epoch_completed),
                       global_step=global_step, ema=ema)
    if config.train.ema and ema is not None:
        ema.copy_temp_to(trainable)
    return root


def maybe_resume(config, transformer_ddp, optimizer, scaler, ema, device, enable_amp, world_size):
    """Resume student adapter + optimizer + ema from config.resume_from. Returns (first_epoch, global_step)."""
    if not config.get("resume_from", None):
        return 0, 0
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
    resume = resolve_resume_checkpoint(config.resume_from)
    state = load_peft_weights(os.path.join(resume, "lora"), device=str(device))
    set_peft_model_state_dict(transformer_ddp.module, state, adapter_name="default")
    opt = os.path.join(resume, "optimizer.pt")
    if os.path.isfile(opt):
        optimizer.load_state_dict(torch.load(opt, map_location=device))
    sc = os.path.join(resume, "scaler.pt")
    if os.path.isfile(sc) and enable_amp:
        scaler.load_state_dict(torch.load(sc, map_location=device))
    restore_ema_and_rng(resume, ema)
    return resume_position(resume, config, world_size)


class BenchmarkTimer:
    """Pilot instrumentation. Records per-update wall-clock, samples, teacher
    forward count, peak VRAM, and checkpoint-save success. Emits a metrics JSON (rank 0)."""
    def __init__(self, config, world_size, warmup, measured):
        self.config = config; self.world_size = world_size
        self.warmup = warmup; self.measured = measured
        self.times = []; self.teacher_fwd = 0; self.ckpt_ok = False
        self.samples_per_update = int(world_size * config.sample.train_batch_size * config.sample.num_batches_per_epoch)
        self._t0 = None

    def update_begin(self):
        torch.cuda.synchronize(); self._t0 = time.time()

    def update_end(self):
        torch.cuda.synchronize(); self.times.append(time.time() - self._t0)

    def add_teacher_fwd(self, n=1):
        self.teacher_fwd += n

    def measured_times(self):
        return self.times[self.warmup:self.warmup + self.measured]

    def report(self, method, rank, t_ref=175.2, out_path=None):
        if not is_main_process(rank):
            return None
        mt = self.measured_times()
        spu = float(np.mean(mt)) if mt else float("nan")
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        eff_sps = self.samples_per_update / spu if spu > 0 else float("nan")
        gpu_h_300 = (spu * 300 * self.world_size) / 3600.0
        # Sample-count calibration; T_ref must be a real same-setting profile.
        new_samples_per_epoch = int(round(self.samples_per_update * t_ref / spu)) if spu > 0 else None
        rec = {
            "method": method, "world_size": self.world_size,
            "per_gpu_batch": int(self.config.sample.train_batch_size),
            "num_batches_per_epoch": int(self.config.sample.num_batches_per_epoch),
            "seconds_per_update": round(spu, 3),
            "samples_per_update": self.samples_per_update,
            "effective_samples_per_second": round(eff_sps, 3),
            "gpu_hours_per_300_updates": round(gpu_h_300, 2),
            "peak_vram_gb": round(peak_gb, 2),
            "teacher_forwards_per_update": round(self.teacher_fwd / max(1, len(self.times)), 1),
            "checkpoint_save_ok": self.ckpt_ok,
            "warmup_updates": self.warmup, "measured_updates": len(mt),
            "T_ref_seconds": t_ref,
            "calibrated_total_samples_per_update_for_T_ref": new_samples_per_epoch,
            "note": "T_ref default = 175.2s = measured Open3 DiffusionOPSD x300 1-update wall-clock (pass "
                    "T_REF=... to override). Sample counts calibrated to match one DiffusionOPSD-Open3 update.",
        }
        print("[opd-benchmark] " + json.dumps(rec), flush=True)
        if out_path:
            with open(out_path, "w") as f:
                json.dump(rec, f, indent=2)
        return rec
