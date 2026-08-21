"""DanceOPD-Open3Teacher. See ``opd/README.md`` for the reproduction contract.

Student: 10-step CFG-free deterministic (dpm2) rollout. Distillation: pick k=1 LOW-noise query
timestep on the on-policy trajectory; minimize velocity MSE between the student and each of the 3
DiffusionOPSD teachers at that SAME query state, averaged 1:1:1 (same-sample ensemble).
Reward models are absent from the second-stage objective and are used only for post-hoc evaluation.
#Iter = 300 optimizer updates (1 update / epoch), save every 10 updates by default.

Run:
  torchrun --standalone --nproc_per_node=8 opd/train_danceopd_open3.py
Env: SAVE_DIR, LOG_DIR, OPD_TEACHER_LORAS="p0,p1,p2" (3 teacher lora dirs),
     BENCHMARK=1 (+ BENCH_WARMUP, BENCH_MEASURED) for the pilot, MODEL_PATH, SD3_TRANSFORMER_GRADCKPT.
"""
import os
import sys
import json

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opd_common import (  # noqa: E402
    setup_distributed, cleanup_distributed, is_main_process,
    TextPromptDataset, DistributedKRepeatSampler, compute_text_embeddings,
    build_models, rollout, student_velocity, teacher_velocities, save_ckpt,
    maybe_resume, BenchmarkTimer,
)
from configs.opd_config import get_config  # noqa: E402

METHOD = "danceopd"


def pick_query_step(num_steps, low_t, gen):
    """k=1 query step. low_t => a low-noise step (later half of the schedule)."""
    if low_t:
        lo, hi = num_steps // 2, num_steps - 1        # low-noise half, avoid the sigma~0 endpoint
    else:
        lo, hi = 1, num_steps - 1
    return int(torch.randint(lo, hi + 1, (1,), generator=gen).item())


def main():
    rank = int(os.environ["RANK"]); world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    config = get_config(METHOD)
    default_save = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "opd", METHOD))
    config.save_dir = os.environ.get("SAVE_DIR", default_save)
    config.logdir = os.environ.get("LOG_DIR", f"{config.save_dir}/logs")
    if os.environ.get("OPD_TEACHER_LORAS"):
        config.opd.teacher_loras = [p for p in os.environ["OPD_TEACHER_LORAS"].split(",") if p]
    # auto-resume from latest checkpoint unless benchmarking
    bench_on = os.environ.get("BENCHMARK", "0") == "1"
    run_done_path = os.path.join(config.save_dir, "run_done.json")
    if os.path.exists(run_done_path) and not bench_on:
        if is_main_process(rank):
            print(f"[{METHOD}] run_done.json exists — already complete at {config.save_dir}; nothing to do.", flush=True)
        cleanup_distributed(); return
    ck_root = os.path.join(config.save_dir, "checkpoints")
    if not bench_on and os.path.isdir(ck_root):
        cks = sorted([int(d.split("-")[-1]) for d in os.listdir(ck_root) if d.startswith("checkpoint-")])
        if cks:
            config.resume_from = os.path.join(ck_root, f"checkpoint-{cks[-1]}")

    if not config.opd.teacher_loras:
        raise ValueError("OPD_TEACHER_LORAS must give 3 teacher LoRA dirs; see opd/README.md.")
    if is_main_process(rank):
        os.makedirs(config.save_dir, exist_ok=True); os.makedirs(config.logdir, exist_ok=True)
        print(f"[{METHOD}] teachers={config.opd.teacher_loras} save_dir={config.save_dir} bench={bench_on}", flush=True)

    (pipeline, transformer_ddp, teacher_names, text_encoders, tokenizers,
     optimizer, ema, scaler, trainable, mp_dtype) = build_models(config, device, rank, local_rank)
    enable_amp = mp_dtype is not None

    train_dataset = TextPromptDataset(config.dataset, "train")
    sampler = DistributedKRepeatSampler(train_dataset, config.sample.train_batch_size,
                                        config.sample.num_image_per_prompt, world_size, rank, seed=0)
    loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0,
                        collate_fn=train_dataset.collate_fn, pin_memory=True)
    train_iter = iter(loader)

    neg_embed, neg_pooled = compute_text_embeddings([""], text_encoders, tokenizers, 128, device)
    neg_embed = neg_embed.repeat(config.sample.train_batch_size, 1, 1)
    neg_pooled = neg_pooled.repeat(config.sample.train_batch_size, 1)
    first_epoch, global_step = maybe_resume(config, transformer_ddp, optimizer, scaler, ema,
                                            device, enable_amp, world_size)

    warmup = int(os.environ.get("BENCH_WARMUP", "2")); measured = int(os.environ.get("BENCH_MEASURED", "6"))
    bench = BenchmarkTimer(config, world_size, warmup, measured) if bench_on else None
    n_updates = (warmup + measured) if bench_on else config.num_epochs
    grad_accum = config.train.gradient_accumulation_steps
    gen = torch.Generator(); gen.manual_seed(1234 + rank)

    for epoch in range(first_epoch, n_updates):
        sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        if bench:
            bench.update_begin()
        for i in range(config.sample.num_batches_per_epoch):
            sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, _ = next(train_iter)
            embeds, pooled = compute_text_embeddings(prompts, text_encoders, tokenizers, 128, device)

            _, traj_latents, sigmas, _ = rollout(
                config, pipeline, transformer_ddp, embeds, pooled, neg_embed[:len(prompts)],
                neg_pooled[:len(prompts)], device, mp_dtype)

            step_idx = pick_query_step(config.sample.num_steps, config.opd.query_low_t, gen)
            z_t = traj_latents[:, step_idx].detach()
            sigma_t = float(sigmas[step_idx])

            with torch.cuda.amp.autocast(enabled=enable_amp, dtype=mp_dtype):
                v_teachers = teacher_velocities(transformer_ddp, teacher_names, z_t, sigma_t, embeds, pooled)
                if bench:
                    bench.add_teacher_fwd(len(v_teachers))
                v_student = student_velocity(transformer_ddp, z_t, sigma_t, embeds, pooled)
                w = list(config.opd.teacher_weights)
                loss = sum(w[j] * torch.nn.functional.mse_loss(v_student.float(), v_teachers[j].float())
                           for j in range(len(v_teachers))) / (sum(w) * grad_accum)
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, config.train.max_grad_norm)
        scaler.step(optimizer); scaler.update()
        if ema is not None:
            ema.step(trainable, global_step)
        global_step += 1
        # Normal runs save periodically; the timing benchmark saves every measured update so
        # checkpoint-write cost remains inside the paper's full-flow wall-clock contract.
        should_save = (
            bench_on
            or global_step % int(config.save_freq) == 0
            or global_step == config.num_epochs
        )
        root = (save_ckpt(config.save_dir, transformer_ddp, global_step, rank, ema, trainable,
                          config, optimizer, scaler, epoch_completed=epoch + 1)
                if should_save else None)
        if bench and root is not None:
            bench.ckpt_ok = os.path.isdir(os.path.join(root, "lora"))
        if bench:
            bench.update_end()   # AFTER save -> checkpoint-write cost is inside the measured wall-clock
        if is_main_process(rank) and not bench_on:
            print(f"[{METHOD}] update {global_step} done (loss={float(loss)*grad_accum:.5f})", flush=True)

    if bench:
        out = os.path.join(config.save_dir, f"benchmark_{METHOD}.json")
        bench.report(METHOD, rank, t_ref=float(os.environ.get("T_REF", "175.2")), out_path=out)
    elif is_main_process(rank):
        with open(run_done_path, "w") as f:
            json.dump({"method": METHOD, "num_epochs": int(config.num_epochs), "global_step": global_step}, f)
        print(f"[{METHOD}] training complete ({config.num_epochs} updates) -> wrote run_done.json", flush=True)
    cleanup_distributed()


if __name__ == "__main__":
    main()
