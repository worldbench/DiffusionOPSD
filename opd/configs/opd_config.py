"""Self-contained configs for the OPD-family teacher-distillation baselines.

Three methods, all in the SD3.5-M / Open3 `Specific=×` setting:
  - danceopd    : student 10-step CFG-free deterministic rollout; distill ONE low-noise query
                  timestep to the 3-teacher velocity ensemble.
  - diffusionopd: student 10-step deterministic rollout; distill ALL 10 transition means to the
                  3-teacher ensemble.
  - flowopd     : FlowGRPO-style stochastic SDE rollout + per-step logprobs; PPO with a STEP-WISE
                  teacher-KL reward against the 3-teacher ensemble.

Teachers = 3 DiffusionOPSD ck100 checkpoints trained on PickScore, ClipScore, and HPSv2.1 as frozen
PEFT LoRA adapters.
Reward models are not loaded in this second stage. #Iter = 300 optimizer updates; normal training
saves every 10 updates. The timing benchmark still saves every measured update so its wall-clock
contract matches the paper. Sample counts below are the documented pilot starting points; the
benchmark re-calibrates them so 1 optimizer update ~= 1 DiffusionOPSD-Open3 epoch.

The exact public reproduction contract is in ``opd/README.md``. This config does not import
``src/config``; field names mirror the shared trainers so the
copied rollout/velocity/save utilities in opd_common.py work unchanged.
"""

import os
from pathlib import Path
import ml_collections

# The defaults are three reward-specific DiffusionOPSD checkpoint-100
# runs for PickScore, ClipScore, and HPSv2.1. Override them through OPD_TEACHER_LORAS when needed.
OPEN3_TEACHERS = ("pickscore", "clipscore", "hpsv2")
PUBLIC_ROOT = Path(__file__).resolve().parents[2]


def _base_config():
    c = ml_collections.ConfigDict()

    # provenance / model
    c.pretrained = ml_collections.ConfigDict()
    c.pretrained.model = os.environ.get("MODEL_PATH", "stabilityai/stable-diffusion-3.5-medium")
    c.pretrained.revision = ""
    c.use_lora = True
    c.mixed_precision = "bf16"
    c.allow_tf32 = True
    c.resolution = 512
    c.seed = None                       # OPD/OPSD policy: no fixed seed (naturally random runs)
    c.debug = False

    # data (Pick-a-Pic prompts, aligned with multi_open3 rows)
    c.dataset = os.environ.get("OPD_DATASET", str(PUBLIC_ROOT / "data" / "pickapic"))
    c.prompt_fn = "general_ocr"         # the framework's TextPromptDataset path
    c.prompt_fn_kwargs = {}

    c.per_prompt_stat_tracking = False  # OPD distills per-sample; no group-relative advantage

    # budget
    c.num_epochs = 300                  # 300 optimizer updates (1 update / epoch, gradient_step_per_epoch=1)
    c.save_freq = int(os.environ.get("OPD_SAVE_FREQ", "10"))
    c.eval_freq = 0                     # post-hoc DrawBench eval, like the main-table methods

    # rollout defaults (overridden per method)
    c.sample = ml_collections.ConfigDict()
    c.sample.num_steps = 10
    c.sample.eval_num_steps = 40
    c.sample.guidance_scale = 1.0       # CFG-free
    c.sample.num_image_per_prompt = 1   # OPD distills each sample independently (K=1)
    c.sample.global_std = True
    c.sample.solver = "dpm2"
    c.sample.deterministic = True
    c.sample.noise_level = 0.0

    # optimizer / train defaults
    c.train = ml_collections.ConfigDict()
    c.train.learning_rate = 3e-4
    c.train.adam_beta1 = 0.9
    c.train.adam_beta2 = 0.999
    c.train.adam_weight_decay = 1e-4
    c.train.adam_epsilon = 1e-8
    c.train.max_grad_norm = 1.0
    c.train.ema = True
    c.train.num_inner_epochs = 1
    c.train.timestep_fraction = 1.0
    c.train.beta = 0.0                  # KL-to-base coeff (teacher-KL uses opd.teacher_kl_coef)
    c.train.lora_path = None
    c.train.beta = 0.0

    # OPD teacher-distillation block
    c.opd = ml_collections.ConfigDict()
    c.opd.teacher_rewards = list(OPEN3_TEACHERS)
    # Public defaults are the three 100-update specialist outputs produced by
    # scripts/train_public.sh. Override with OPD_TEACHER_LORAS if stored elsewhere.
    _OPSD = PUBLIC_ROOT / "outputs"
    c.opd.teacher_loras = [
        str(_OPSD / "sd35_pickscore" / "checkpoints" / "checkpoint-100" / "lora"),
        str(_OPSD / "sd35_clipscore" / "checkpoints" / "checkpoint-100" / "lora"),
        str(_OPSD / "sd35_hpsv2" / "checkpoints" / "checkpoint-100" / "lora"),
    ]
    c.opd.teacher_weights = [1.0, 1.0, 1.0]   # 1:1:1 same-sample ensemble
    c.opd.ensemble = "same_sample"            # not batch rotation

    return c


def get_config(name):
    """name = 'danceopd' | 'diffusionopd' | 'flowopd'."""
    method = name.strip().lower()
    c = _base_config()

    if method == "danceopd":
        # DanceOPD: student 10-step CFG-free rollout; one low-noise velocity-MSE query.
        c.sample.solver = "dpm2"
        c.sample.deterministic = True
        c.sample.noise_level = 0.0
        c.sample.train_batch_size = 6          # per-GPU samples/batch (VRAM 45.9GB @ bench)
        c.sample.num_batches_per_epoch = 56    # T_ref=175.2 calibrated: 42 × 175.2/132.455s -> 2688 samp/upd
        c.opd.loss = "velocity_mse_query"
        c.opd.query_k = 1                      # one query timestep per sample
        c.opd.query_low_t = True               # low-noise (small sigma) query state

    elif method == "diffusionopd":
        # DiffusionOPD: distill teacher transition means over all 10 deterministic steps.
        c.sample.solver = "dpm2"
        c.sample.deterministic = True
        c.sample.noise_level = 0.0
        c.sample.train_batch_size = 3          # VRAM 39.1GB @ bench
        c.sample.num_batches_per_epoch = 22    # T_ref=175.2 calibrated: 21 × 175.2/163.981s -> 528 samp/upd
        c.opd.loss = "transition_mean_mse_all"
        c.opd.query_k = 10                     # all denoising steps

    elif method == "flowopd":
        # FlowOPD: stochastic SDE rollout + logprobs; PPO with a step-wise teacher-KL
        # reward (per-step teacher log-prob of the student transition) + global per-step advantage.
        c.sample.solver = "flow"
        c.sample.deterministic = False
        c.sample.noise_level = 0.7             # SDE eta > 0
        c.train.timestep_fraction = 1.0        # train on ALL SDE transitions
        c.sample.train_batch_size = 4          # VRAM 41.3GB @ bench
        c.sample.num_batches_per_epoch = 17    # T_ref=175.2 calibrated: 18 × 175.2/186.593s -> 544 samp/upd
        c.opd.loss = "ppo_stepwise_teacher_kl"
        c.opd.clip_range = 0.2                 # PPO clip epsilon
        c.opd.teacher_kl_coef = 1.0

    else:
        raise ValueError(f"Unknown OPD method '{name}' (expect danceopd|diffusionopd|flowopd)")

    # gradient accumulation == num_batches_per_epoch (1 optimizer update per epoch)
    c.train.gradient_accumulation_steps = int(c.sample.num_batches_per_epoch)
    c.run_name = f"opd_{method}_open3"
    # bookkeeping: samples/epoch = world_size(8) × per_gpu_batch × num_batches_per_epoch
    c.opd.samples_per_epoch_x8 = 8 * int(c.sample.train_batch_size) * int(c.sample.num_batches_per_epoch)
    return c
