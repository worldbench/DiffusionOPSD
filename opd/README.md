# OPD-family baselines

This directory implements the three teacher-distillation baselines reported in the paper's SD3.5-M Open3 comparison. All three train one shared student from three frozen, reward-specific DiffusionOPSD specialists trained on PickScore, CLIPScore, and HPSv2.1. Reward models do not enter the second-stage loss.

## Reproduction contract

### Shared setup

- Base/student: `stabilityai/stable-diffusion-3.5-medium`, initialized from the base model with LoRA rank 32 and alpha 64.
- Training prompts: the exact 25,415-prompt Pick-a-Pic manifest prepared by `scripts/prepare_pickapic_prompts.py`.
- Resolution and rollout: 512×512, CFG-free.
- Teachers: three frozen DiffusionOPSD checkpoint-100 LoRA adapters trained independently on PickScore, CLIPScore, and HPSv2.1.
- Ensemble: every sample queries all three teachers with 1:1:1 weights; teacher rotation by batch is not used.
- Distillation budget: 300 optimizer updates, checkpoint every 10 updates plus the final update. The timing pilot writes every measured update.
- Evaluation: post-hoc evaluation on 1,000 DrawBench prompts with deterministic 40-step flow/ODE sampling and `guidance_scale=1.0`.

### Objectives and calibrated budget

| Method | Student rollout | Teacher target/update | Samples/update |
|---|---|---|---:|
| DanceOPD | deterministic, CFG-free, 10 steps | equally weighted velocity MSE at one low-noise query | 2,688 |
| DiffusionOPD | deterministic, CFG-free, 10 steps | equally weighted teacher transition-mean MSE at all 10 steps | 528 |
| FlowOPD | stochastic SDE flow | mean teacher transition log-probability, global step-wise advantage, clipped PPO | 544 |

#### DanceOPD

1. Collect a deterministic CFG-free 10-step student trajectory.
2. Select one low-noise query state.
3. Query all three teachers at that state.
4. Minimize the equally weighted mean of the three teacher velocity-MSE losses.

#### DiffusionOPD

1. Collect the same deterministic CFG-free 10-step student trajectory.
2. At all 10 denoising steps, construct each teacher's transition mean under the same solver step.
3. Minimize the equally weighted transition-mean MSE over all teachers and steps, rather than an unweighted raw-velocity loss.

#### FlowOPD

1. Collect a stochastic SDE-flow student trajectory and its transition log probabilities.
2. Evaluate every realized student transition under each teacher transition distribution.
3. Average the teacher log probabilities into a per-step teacher-consistency reward.
4. Normalize advantages globally and independently per step.
5. Apply clipped PPO to all stochastic transitions.

The sample counts are calibrated to a 175.2-second reference update that includes teacher queries and checkpoint writing. At 300 updates, each second-stage run costs approximately 117 GPU-hours, or 14.6 wall-clock hours on eight GPUs. Specialist-training cost is reported separately.

## 1. Train the three specialists

From the public-release root:

```bash
for reward in pickscore clipscore hpsv2; do
  bash scripts/train_public.sh sd35 "$reward"
done
```

The default teacher adapters are:

```text
outputs/sd35_pickscore/checkpoints/checkpoint-100/lora
outputs/sd35_clipscore/checkpoints/checkpoint-100/lora
outputs/sd35_hpsv2/checkpoints/checkpoint-100/lora
```

To use other locations:

```bash
export OPD_TEACHER_LORAS="path/to/pick/lora,path/to/clip/lora,path/to/hpsv2/lora"
```

Missing or invalid adapters fail closed. The explicit value `DUMMY,DUMMY,DUMMY` exists only for a teacher-identity-independent timing check and must not be used for results.

## 2. Run a short full-flow pilot

The pilot covers student rollout, all three teacher queries, backward, optimizer step, and checkpoint writing. Run it before any 300-update job:

```bash
METHODS="danceopd diffusionopd flowopd" \
BENCH_WARMUP=2 BENCH_MEASURED=6 \
bash opd/benchmark_opd.sh
```

## 3. Train

```bash
METHOD=danceopd     bash opd/launch_opd.sh
METHOD=diffusionopd bash opd/launch_opd.sh
METHOD=flowopd      bash opd/launch_opd.sh
```

Each trainer auto-resumes from the latest complete checkpoint. Set `OPD_SAVE_FREQ=1` only when every intermediate curve point is required. The timing pilot always includes checkpoint writing in every measured update, preserving the paper's wall-clock calibration.

## Files

- `opd_common.py`: shared model/adapters, rollout, checkpointing, and benchmark instrumentation.
- `configs/opd_config.py`: the three paper-aligned configurations.
- `train_*_open3.py`: method-specific objectives.
- `benchmark_opd.sh` / `launch_opd.sh`: portable pilot and full-run entry points.
