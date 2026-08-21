#!/usr/bin/env python3
"""Rollout-budget-matched ReFL trainer for SD3.5-M.

One loss is taken from one uniformly selected late state of every real
current-policy trajectory.  There is no endpoint regression, group advantage,
diffusion pretraining term, reference MSE, or KL term.
"""

from __future__ import annotations

import datetime
import os
import platform
import socket
import time
from typing import Sequence

import torch
import torch.distributed as dist
from absl import app
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusionopsd import profiling
from diffusionopsd.metrics import install_wandb_jsonl_tee
from scripts import train_opsd_ri_sd3 as runtime
from scripts.refl_common import (
    EVAL_SCORE_BATCH_ONE_REWARDS,
    INTERNVL_REWARDS,
    PAIRWISE_REWARDS,
    REFL_IMPLEMENTATION_REVISION,
    REWARDS,
    append_jsonl,
    assert_output_under_allowed_remote_root,
    build_update_records,
    choose_late_index,
    derive_gradient_accumulation_steps,
    distributed_moments,
    distributed_prompt_group_dispersion,
    fixed_eval_manifest_records,
    fixed_manifest_records,
    gather_indexed_scores,
    load_prompt_records,
    load_refl_checkpoint,
    load_ref_images,
    maybe_no_sync,
    provenance_from_env,
    prepare_resume_artifacts,
    read_json,
    require_finite_nonzero,
    require_pairwise_references,
    restore_refl_rng_state,
    shard_update_records,
    save_refl_checkpoint,
    summarize_scores,
    sync_mean,
    write_json,
)


FLAGS = runtime.FLAGS


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _encode(pipeline, prompts: Sequence[str], device: torch.device, dtype: torch.dtype):
    pipeline.text_encoder_3.to(device, dtype=dtype)
    return runtime.compute_text_embeddings(
        list(prompts),
        [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3],
        [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3],
        max_sequence_length=128,
        device=device,
    )


def _park_t5(pipeline) -> None:
    pipeline.text_encoder_3.to("cpu")
    torch.cuda.empty_cache()


@torch.no_grad()
def _rollout(
    pipeline,
    prompt_embeds: torch.Tensor,
    pooled_embeds: torch.Tensor,
    config,
    *,
    generators,
    num_steps: int,
    solver: str,
    stop_before_index: int | None = None,
    decode: bool = True,
):
    images, latents, _ = runtime.pipeline_with_logprob(
        pipeline,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_embeds,
        num_inference_steps=num_steps,
        guidance_scale=1.0,
        output_type="pt",
        height=config.resolution,
        width=config.resolution,
        noise_level=config.sample.noise_level,
        deterministic=True,
        solver=solver,
        model_type="sd3",
        generator=generators,
        stop_before_index=stop_before_index,
        decode=decode,
    )
    return images, latents


def _late_x0(
    model,
    latents: Sequence[torch.Tensor],
    scheduler_sigmas: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_embeds: torch.Tensor,
    late_indices: Sequence[int],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    z_t = torch.cat([latents[index][batch : batch + 1] for batch, index in enumerate(late_indices)]).detach().float()
    sigma = torch.stack([scheduler_sigmas[index].detach().float() for index in late_indices]).to(z_t.device)
    timestep = (sigma * 1000.0).to(dtype=torch.long)
    with torch.autocast("cuda", dtype=dtype):
        velocity = model(
            hidden_states=z_t,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
    sigma_e = sigma.view(-1, *([1] * (z_t.ndim - 1)))
    return z_t - sigma_e * velocity.float(), sigma


def _preencode_records(pipeline, records, rank: int, world_size: int, device, dtype, batch_size: int = 4):
    local = [record for record in records if int(record["index"]) % world_size == rank]
    encoded = []
    for start in range(0, len(local), batch_size):
        chunk = local[start : start + batch_size]
        embeds, pooled = _encode(pipeline, [x["prompt"] for x in chunk], device, dtype)
        for i, record in enumerate(chunk):
            encoded.append((record, embeds[i : i + 1].detach().cpu(), pooled[i : i + 1].detach().cpu()))
    return encoded


def _preencode_eval_batches(pipeline, records, rank: int, world_size: int, device, dtype):
    if not records:
        return []
    canonical_batch = int(records[0]["canonical_generation_batch_size"])
    encoded = []
    for batch_index, start in enumerate(range(0, len(records), canonical_batch)):
        if batch_index % world_size != rank:
            continue
        chunk = list(records[start : start + canonical_batch])
        expected_size = len(chunk)
        expected_seed = int(chunk[0]["base_seed"]) + batch_index
        if any(
            int(record["noise_seed"]) != expected_seed
            or int(record["noise_offset"]) != offset
            or int(record["noise_batch_size"]) != expected_size
            for offset, record in enumerate(chunk)
        ):
            raise RuntimeError(f"canonical SD eval batch metadata mismatch at batch {batch_index}")
        embeds, pooled = _encode(pipeline, [record["prompt"] for record in chunk], device, dtype)
        encoded.append((chunk, embeds.detach().cpu(), pooled.detach().cpu()))
    return encoded


def _encode_update_cache(pipeline, local_micro_batches, device, dtype):
    unique: dict[int, str] = {}
    for micro in local_micro_batches:
        for record in micro:
            unique[int(record["dataset_index"])] = record["prompt"]
    indices = sorted(unique)
    cache = {}
    for start in range(0, len(indices), 4):
        chunk = indices[start : start + 4]
        embeds, pooled = _encode(pipeline, [unique[index] for index in chunk], device, dtype)
        for i, index in enumerate(chunk):
            cache[index] = (embeds[i : i + 1].detach(), pooled[i : i + 1].detach())
    return cache


def _refs(records, root: str, reward: str, device: torch.device):
    if reward not in PAIRWISE_REWARDS:
        return None
    return load_ref_images([record["ref_path"] for record in records], root, device)


def _score_images_eval(scorer, reward: str, images: torch.Tensor, prompts, refs):
    """Paper evaluator's frozen forward scorer path (not the training gradient path)."""
    with torch.no_grad():
        if reward == "internvl_t2i":
            from diffusionopsd.internvl_bridge import remote_reward_scores_forward
            return remote_reward_scores_forward(images, prompts).float()
        if reward == "internvl_dual":
            from diffusionopsd.internvl_bridge import remote_reward_scores_pair_forward
            return remote_reward_scores_pair_forward(images, refs, prompts).float()
        if reward == "pickscore":
            from PIL import Image
            arrays = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            pil_images = [Image.fromarray(array.transpose(1, 2, 0)) for array in arrays]
            return scorer(prompts, pil_images).float()
        if reward == "aesthetic":
            pixels = (images * 255).round().clamp(0, 255).to(torch.uint8)
            return scorer(pixels).float()
        if reward == "imagereward":
            return scorer(prompts, images).float()
        if reward in {"clipscore", "hpsv2", "hpsv3", "deqa", "altclip"}:
            return scorer(images, prompts).float()
        raise ValueError(f"unsupported canonical eval reward: {reward}")


def _forward_reward(pipeline, scorer, reward: str, x0: torch.Tensor, prompts, refs):
    with torch.no_grad():
        images = runtime._decode01(pipeline, x0)
        if reward == "internvl_t2i":
            from diffusionopsd.internvl_bridge import remote_reward_scores_forward
            return remote_reward_scores_forward(images, prompts).float()
        if reward == "internvl_dual":
            from diffusionopsd.internvl_bridge import remote_reward_scores_pair_forward
            return remote_reward_scores_pair_forward(images, refs, prompts).float()
        return runtime._reward_scores_grad(scorer, reward, images, prompts, ref=refs).float()


def _evaluate(
    *,
    pipeline,
    model,
    scorer,
    reward: str,
    encoded_batches,
    manifest_records,
    config,
    device,
    dtype,
    group,
    rank: int,
    num_steps: int,
    solver: str,
    ref_root: str,
    output_path: str,
    update: int | str,
    score_scale: float,
):
    local_indices = []
    local_scores = []
    model.eval()
    for records, embeds_cpu, pooled_cpu in encoded_batches:
        embeds = embeds_cpu.to(device)
        pooled = pooled_cpu.to(device)
        generator = torch.Generator(device=device).manual_seed(int(records[0]["noise_seed"]))
        with torch.autocast("cuda", dtype=dtype):
            images, _latents = _rollout(
                pipeline, embeds, pooled, config, generators=generator,
                num_steps=num_steps, solver=solver,
            )
        images = images.detach().float()
        score_batch = 1 if reward in EVAL_SCORE_BATCH_ONE_REWARDS else len(records)
        for start in range(0, len(records), score_batch):
            chunk = records[start : start + score_batch]
            refs = _refs(chunk, ref_root, reward, device)
            score = _score_images_eval(
                scorer, reward, images[start : start + len(chunk)],
                [record["prompt"] for record in chunk], refs,
            )
            local_indices.extend(int(record["index"]) for record in chunk)
            local_scores.append(score.detach().float() * float(score_scale))
    index_tensor = torch.tensor(local_indices, device=device, dtype=torch.long)
    score_tensor = torch.cat(local_scores) if local_scores else torch.empty(0, device=device)
    merged = gather_indexed_scores(index_tensor, score_tensor, group, rank=rank)
    payload = None
    if rank == 0:
        assert merged is not None
        by_index = {int(record["index"]): record for record in manifest_records}
        values = [score for _, score in merged]
        stats = summarize_scores(values)
        per_image = []
        for index, score in merged:
            source = by_index[index]
            per_image.append({
                "index": index,
                "prompt": source["prompt"],
                "ref_path": source.get("ref_path", ""),
                "seed": int(source["noise_seed"]),
                "noise_offset": int(source["noise_offset"]),
                "noise_batch_size": int(source["noise_batch_size"]),
                "canonical_generation_batch_size": int(source["canonical_generation_batch_size"]),
                "score": score,
            })
        payload = {
            "backbone": "sd35m",
            "reward": reward,
            "update": update,
            "protocol": {
                "sampler": solver,
                "num_steps": num_steps,
                "guidance_scale": 1.0,
                "resolution": int(config.resolution),
                "base_seed": int(manifest_records[0]["base_seed"]),
                "generation_batch_size": int(manifest_records[0]["canonical_generation_batch_size"]),
                "score_batch_size": 1 if reward in EVAL_SCORE_BATCH_ONE_REWARDS
                                    else int(manifest_records[0]["canonical_generation_batch_size"]),
                "reward_path": "canonical_flow_grpo_forward",
                "seed_scheme": str(manifest_records[0]["seed_scheme"]),
                "score_scale": "paper_raw" if score_scale != 1.0 else "native",
            },
            "stats": stats,
            "per_image": per_image,
        }
        write_json(output_path, payload, rank=rank)
    return payload


def _calibrate(
    *, pipeline, model, scorer, reward, encoded_records, config, rr, device, dtype, group, rank
):
    if not bool(rr.standardize_reward):
        payload = {
            "mode": "released_imagereward_raw_scale",
            "reward": reward,
            "raw_mean": None,
            "raw_std": None,
            "samples_global": 0,
            "hinge_margin": float(rr.hinge_margin),
            "used": False,
        }
        write_json(os.path.join(config.save_dir, "calibration.json"), payload, rank=rank)
        return 0.0, 1.0

    local_indices, local_scores = [], []
    model.eval()
    for record, embeds_cpu, pooled_cpu in encoded_records:
        embeds = embeds_cpu.to(device)
        pooled = pooled_cpu.to(device)
        generator = torch.Generator(device=device).manual_seed(int(record["noise_seed"]))
        num_steps = int(config.sample.num_steps)
        late_index = choose_late_index(num_steps, float(rr.late_fraction), int(record["late_seed"]))
        with torch.autocast("cuda", dtype=dtype):
            _images, latents = _rollout(
                pipeline, embeds, pooled, config, generators=generator,
                num_steps=int(config.sample.num_steps), solver=str(config.sample.solver),
                stop_before_index=late_index, decode=False,
            )
        if _images is not None or len(latents) != late_index + 1:
            raise AssertionError("calibration must stop exactly before the sampled denoiser call")
        with torch.no_grad():
            x0, _sigma = _late_x0(
                model, latents, pipeline.scheduler.sigmas, embeds, pooled, [late_index], dtype
            )
        refs = _refs([record], str(rr.pairwise_train_ref_root), reward, device)
        score = _forward_reward(pipeline, scorer, reward, x0, [record["prompt"]], refs)
        local_indices.append(int(record["index"]))
        local_scores.append(score.detach().float())
    merged = gather_indexed_scores(
        torch.tensor(local_indices, device=device, dtype=torch.long),
        torch.cat(local_scores) if local_scores else torch.empty(0, device=device),
        group,
        rank=rank,
    )
    if rank == 0:
        assert merged is not None
        stats = summarize_scores([score for _, score in merged])
        require_finite_nonzero("calibration raw_std", float(stats["std"]), minimum=1e-6)
        payload = {
            "mode": "frozen_base_random_late_one_step_x0",
            "reward": reward,
            "raw_mean": stats["mean"],
            "raw_std": stats["std"],
            "raw_se": stats["se"],
            "samples_global": stats["n"],
            "late_fraction": float(rr.late_fraction),
            "hinge_margin": float(rr.hinge_margin),
            "standardize_reward": True,
            "used": True,
        }
    else:
        payload = None
    obj = [payload]
    dist.broadcast_object_list(obj, src=0, group=group, device=device)
    payload = obj[0]
    write_json(os.path.join(config.save_dir, "calibration.json"), payload, rank=rank)
    return float(payload["raw_mean"]), float(payload["raw_std"])


def _flatten_metrics(prefix: str, stats: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def main(_):
    process_started = time.perf_counter()
    config = FLAGS.config
    rr = config.refl
    reward = str(rr.reward)
    if reward not in REWARDS:
        raise ValueError(f"unsupported ReFL reward: {reward}")
    assert_output_under_allowed_remote_root(config.save_dir)
    assert_output_under_allowed_remote_root(config.logdir)

    rank = int(os.environ["RANK"])
    launch_world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    bridge_active = reward in INTERNVL_REWARDS
    if bridge_active:
        if not runtime.bridge_enabled():
            raise RuntimeError(f"{reward} requires INTERNVL_BRIDGE=1")
        from diffusionopsd.internvl_bridge import server_ranks_for, bridge_server_devices_for
        if rank in server_ranks_for(launch_world):
            local_rank = bridge_server_devices_for(rank, launch_world)[0]
    runtime.setup_distributed(rank, local_rank, launch_world)
    device = torch.device(f"cuda:{local_rank}")

    group = None
    shutdown_bridge = None
    policy_world = launch_world
    if bridge_active:
        from diffusionopsd.internvl_bridge import (
            RewardServer,
            bridge_client_shutdown,
            bridge_server_devices_for,
            is_server_rank,
            make_bridge_groups,
            num_servers,
            policy_count_for_server,
            policy_group,
        )
        make_bridge_groups(launch_world)
        if is_server_rank(rank):
            devices = bridge_server_devices_for(rank, launch_world)
            RewardServer(devices[0], devices, reward_kind=reward).serve(
                n_policy=policy_count_for_server(rank, launch_world)
            )
            runtime.cleanup_distributed()
            return
        group = policy_group()
        runtime.POLICY_GROUP = group
        policy_world = launch_world - num_servers()
        shutdown_bridge = bridge_client_shutdown

    expected_launch = int(rr.expected_launch_world_size)
    expected_policy = int(rr.expected_policy_world_size)
    if launch_world != expected_launch or policy_world != expected_policy:
        raise RuntimeError(
            f"topology mismatch: launch={launch_world}/{expected_launch}, "
            f"policy={policy_world}/{expected_policy}"
        )
    micro_batch = int(rr.micro_batch_size)
    if micro_batch != 1:
        raise RuntimeError("truncated-prefix ReFL currently requires micro_batch_size=1")
    accum = derive_gradient_accumulation_steps(
        int(rr.trajectories_per_update), policy_world, micro_batch
    )
    if accum != int(rr.gradient_accumulation_steps):
        raise RuntimeError(f"resolved accumulation {accum} != config {rr.gradient_accumulation_steps}")
    if profiling.profile_enabled():
        profiling.enable()
    profiler = profiling.Profiler(config, policy_world, rank, device) if profiling.is_enabled() else None
    metrics_path = os.path.join(config.save_dir, "metrics.jsonl")

    seed = int(rr.seed) + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if rank == 0:
        os.makedirs(config.save_dir, exist_ok=False) if not os.path.exists(config.save_dir) else None
        os.makedirs(config.logdir, exist_ok=True)
        runtime.wandb.init(project="diffusionopsd", name=config.run_name, config=config.to_dict(), dir=config.logdir)
        install_wandb_jsonl_tee(runtime.wandb, metrics_path, durable=True, strict=True)
    dist.barrier(group=group, device_ids=[torch.cuda.current_device()])

    dtype = torch.float16 if config.mixed_precision == "fp16" else torch.bfloat16
    scaler = runtime.GradScaler(enabled=(dtype == torch.float16))
    pipeline = runtime.StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.safety_checker = None
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=dtype)
    pipeline.text_encoder_2.to(device, dtype=dtype)
    pipeline.text_encoder_3.to(device, dtype=dtype)
    transformer = pipeline.transformer.to(device)
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    pipeline.vae.enable_gradient_checkpointing()

    targets = [
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
        "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v",
    ]
    lora = LoraConfig(r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=targets)
    if config.train.lora_path:
        transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
    else:
        transformer = get_peft_model(transformer, lora)
    transformer.set_adapter("default")
    model = DDP(
        transformer, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=False, process_group=group,
    )
    pipeline.transformer = model.module
    trainable = [parameter for parameter in model.module.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(rr.learning_rate), betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=float(rr.weight_decay), eps=config.train.adam_epsilon,
    )
    ema = runtime.EMAModuleWrapper(trainable, decay=0.9, update_step_interval=1, device=device)

    provenance = provenance_from_env()
    resume_checkpoint = str(config.resume_from or "")
    global_step = 0
    if resume_checkpoint:
        global_step = load_refl_checkpoint(
            resume_checkpoint,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            expected_provenance=provenance,
            group=group,
        )
    dist.barrier(group=group, device_ids=[torch.cuda.current_device()])

    train_records = load_prompt_records(os.path.join(config.dataset, "train.txt"))
    calibration_records = fixed_manifest_records(
        load_prompt_records(str(rr.calibration_manifest_path)),
        count=int(rr.calibration_num_prompts),
        noise_seed=int(rr.calibration_noise_seed),
        late_seed=int(rr.calibration_late_seed),
    )
    final_records = fixed_eval_manifest_records(
        load_prompt_records(str(rr.final_manifest_path)),
        count=int(rr.final_eval_num_prompts),
        base_seed=int(rr.final_noise_seed),
        generation_batch_size=int(rr.eval_generation_batch_size),
    )
    curve_records = final_records[: int(rr.curve_eval_num_prompts)]
    if reward in PAIRWISE_REWARDS:
        require_pairwise_references(calibration_records, str(rr.pairwise_train_ref_root), label="calibration")
        require_pairwise_references(final_records, str(rr.pairwise_eval_ref_root), label="evaluation")

    write_json(
        os.path.join(config.save_dir, "manifests", "calibration.json"),
        {"source": str(rr.calibration_manifest_path), "records": calibration_records,
         "used": bool(rr.standardize_reward), "object": "random-late one-step x0"}, rank=rank,
    )
    write_json(
        os.path.join(config.save_dir, "manifests", "curve_eval.json"),
        {"source": str(rr.curve_manifest_path), "records": curve_records}, rank=rank,
    )
    write_json(
        os.path.join(config.save_dir, "manifests", "final_eval.json"),
        {"source": str(rr.final_manifest_path), "records": final_records}, rank=rank,
    )
    encoded_calibration = _preencode_records(
        pipeline,
        calibration_records if rr.standardize_reward and not resume_checkpoint else [],
        rank,
        policy_world,
        device,
        dtype,
    )
    encoded_final = _preencode_eval_batches(
        pipeline, final_records, rank, policy_world, device, dtype
    )
    encoded_curve = _preencode_eval_batches(
        pipeline, curve_records, rank, policy_world, device, dtype
    )
    _park_t5(pipeline)

    scorer = None if bridge_active else runtime._load_reward_scorer(reward, device)
    initialization_seconds = time.perf_counter() - process_started
    calibration_started = time.perf_counter()
    if resume_checkpoint:
        calibration_payload = read_json(os.path.join(config.save_dir, "calibration.json")) if rank == 0 else None
        obj = [calibration_payload]
        dist.broadcast_object_list(obj, src=0, group=group, device=device)
        calibration_payload = obj[0]
        if reward == "imagereward":
            if calibration_payload.get("mode") != "released_imagereward_raw_scale":
                raise RuntimeError("resume calibration mode mismatch for ImageReward")
            calibration_mean, calibration_std = 0.0, 1.0
        else:
            if calibration_payload.get("mode") != "frozen_base_random_late_one_step_x0":
                raise RuntimeError("resume calibration mode mismatch")
            calibration_mean = float(calibration_payload["raw_mean"])
            calibration_std = float(calibration_payload["raw_std"])
            require_finite_nonzero("resume calibration raw_std", calibration_std, minimum=1e-6)
    else:
        calibration_mean, calibration_std = _calibrate(
            pipeline=pipeline, model=model.module, scorer=scorer, reward=reward,
            encoded_records=encoded_calibration, config=config, rr=rr, device=device, dtype=dtype,
            group=group, rank=rank,
        )
    calibration_seconds = time.perf_counter() - calibration_started

    score_scale = 26.0 if reward == "pickscore" else 1.0
    eval0_started = time.perf_counter()
    eval0 = None
    if not resume_checkpoint:
        ema.copy_ema_to(trainable, store_temp=True)
        try:
            eval0 = _evaluate(
                pipeline=pipeline, model=model.module, scorer=scorer, reward=reward,
                encoded_batches=encoded_curve, manifest_records=curve_records,
                config=config, device=device, dtype=dtype, group=group, rank=rank,
                num_steps=40, solver="flow", ref_root=str(rr.pairwise_eval_ref_root),
                output_path=os.path.join(config.save_dir, "curve", "update_0000.json"),
                update=0, score_scale=score_scale,
            )
        finally:
            ema.copy_temp_to(trainable)
    elif rank == 0 and not os.path.isfile(os.path.join(config.save_dir, "curve", "update_0000.json")):
        raise RuntimeError("resumable checkpoint exists but update-0 evaluation is missing")
    update0_eval_seconds = time.perf_counter() - eval0_started if not resume_checkpoint else 0.0

    train_loop_started = time.time()
    run_config = {
        "backbone": "SD3.5-M",
        "method": "rollout-budget-matched ReFL",
        "reward": reward,
        "loss": "1e-3 * relu(2 - raw_ImageReward)" if reward == "imagereward"
                else "1e-3 * relu(2 - frozen_base_zscore_reward)",
        "algorithm": {
            "implementation_revision": REFL_IMPLEMENTATION_REVISION,
            "current_policy_fresh_noise_rollout": True,
            "prefix_no_grad": True,
            "sampled_timestep_before_rollout": True,
            "truncated_prefix_rollout": True,
            "unused_suffix_steps_executed": 0,
            "endpoint_decode_during_training": False,
            "late_state_per_trajectory": 1,
            "late_fraction": float(rr.late_fraction),
            "endpoint_regression": False,
            "group_advantage": False,
            "diffusion_pretraining_loss": False,
            "reference_mse_or_kl": False,
            "checkpoint_before_online_eval": True,
            "resume_replays_boundary_online_eval": True,
        },
        "batch": {
            "distinct_prompt_groups": int(rr.distinct_prompt_groups),
            "trajectories_per_prompt": int(rr.trajectories_per_prompt),
            "trajectories_per_update": int(rr.trajectories_per_update),
            "micro_batch_size": micro_batch,
            "policy_world_size": policy_world,
            "launch_world_size": launch_world,
            "actual_compute_gpu_count": int(rr.actual_compute_gpu_count),
            "reserved_gpu_count": int(rr.reserved_gpu_count),
            "gradient_accumulation_steps": accum,
        },
        "num_updates": int(rr.num_updates),
        "checkpoint_policy": (
            f"complete atomic resumable checkpoint every {int(rr.checkpoint_every)} updates "
            "plus final, committed before same-step online eval"
        ),
        "artifact_storage": "config.save_dir",
        "calibration": {"mean": calibration_mean, "std": calibration_std,
                        "one_step_late_state": bool(rr.standardize_reward)},
        "config": config.to_dict(),
        "provenance": provenance,
        "phase_timing_seconds": {
            "initialization": initialization_seconds,
            "calibration": calibration_seconds,
            "update0_eval": update0_eval_seconds,
        },
        "environment": {
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
        },
        "started": _now(),
        "resume_history": [],
    }
    if resume_checkpoint and rank == 0:
        existing_run = read_json(os.path.join(config.save_dir, "run_config.json"))
        if existing_run.get("algorithm", {}).get("implementation_revision") != REFL_IMPLEMENTATION_REVISION:
            raise RuntimeError("resume run_config implementation revision mismatch")
        run_config["started"] = existing_run.get("started", run_config["started"])
        run_config["resume_history"] = list(existing_run.get("resume_history", [])) + [{
            "checkpoint": resume_checkpoint,
            "global_step": global_step,
            "hostname": socket.gethostname(),
            "resumed": _now(),
        }]
    write_json(os.path.join(config.save_dir, "run_config.json"), run_config, rank=rank)
    if rank == 0 and eval0 is not None:
        runtime.wandb.log(
            {"step": 0, "kind": "online_eval", "eval_seconds": update0_eval_seconds, **eval0["stats"]},
            step=0,
        )

    optimizer.zero_grad(set_to_none=True)
    if resume_checkpoint:
        restore_refl_rng_state(resume_checkpoint, device=device, group=group)
        replay_eval_step = (
            global_step
            if global_step % int(rr.eval_every) == 0 or global_step == int(rr.num_updates)
            else None
        )
        retained = prepare_resume_artifacts(
            config.save_dir,
            global_step,
            rank=rank,
            replay_online_eval_step=replay_eval_step,
        )
    else:
        retained = (0.0, update0_eval_seconds, False)
    retained_obj = [retained if rank == 0 else None]
    dist.broadcast_object_list(retained_obj, src=0, group=group, device=device)
    cumulative_train_seconds = float(retained_obj[0][0])
    cumulative_online_eval_seconds = float(retained_obj[0][1])
    replay_boundary_online_eval = bool(retained_obj[0][2])
    cumulative_trajectories = global_step * int(rr.trajectories_per_update)

    if replay_boundary_online_eval:
        online_eval_started = time.perf_counter()
        ema.copy_ema_to(trainable, store_temp=True)
        try:
            result = _evaluate(
                pipeline=pipeline, model=model.module, scorer=scorer, reward=reward,
                encoded_batches=encoded_curve, manifest_records=curve_records,
                config=config, device=device, dtype=dtype,
                group=group, rank=rank, num_steps=40, solver="flow",
                ref_root=str(rr.pairwise_eval_ref_root),
                output_path=os.path.join(
                    config.save_dir, "curve", f"update_{global_step:04d}.json"
                ),
                update=global_step, score_scale=score_scale,
            )
        finally:
            ema.copy_temp_to(trainable)
        if rank == 0 and result is not None:
            eval_seconds = time.perf_counter() - online_eval_started
            cumulative_online_eval_seconds += eval_seconds
            runtime.wandb.log(
                {
                    "step": global_step,
                    "kind": "online_eval",
                    "resume_replay": True,
                    "eval_seconds": eval_seconds,
                    "cumulative_online_eval_seconds": cumulative_online_eval_seconds,
                    **result["stats"],
                },
                step=global_step,
            )

    while global_step < int(rr.num_updates):
        profile_index = global_step
        if profiler is not None:
            profiler.epoch_begin(profile_index)
        step_started = time.perf_counter()
        update_records = build_update_records(
            train_records,
            update_index=global_step,
            distinct_prompt_groups=int(rr.distinct_prompt_groups),
            trajectories_per_prompt=int(rr.trajectories_per_prompt),
            prompt_seed=int(rr.train_prompt_seed),
            shuffle_seed=int(rr.train_shuffle_seed),
            noise_seed=int(rr.train_noise_seed),
            late_seed=int(rr.train_late_seed),
        )
        local_batches = shard_update_records(update_records, rank, policy_world, micro_batch)
        if len(local_batches) != accum:
            raise AssertionError(f"local micro-batches {len(local_batches)} != accumulation {accum}")
        if rank == 0:
            groups = {}
            for record in update_records:
                groups.setdefault(int(record["group_slot"]), {
                    "dataset_index": int(record["dataset_index"]), "prompt": record["prompt"],
                    "ref_path": record["ref_path"],
                })
            append_jsonl(
                os.path.join(config.save_dir, "manifests", "train_updates.jsonl"),
                {
                    "update": global_step + 1,
                    "groups": [groups[i] for i in range(int(rr.distinct_prompt_groups))],
                    "trajectories_per_prompt": int(rr.trajectories_per_prompt),
                    "trajectory_seed_rule": "train_noise_seed + zero_based_update*B + group_slot*K + repeat_index",
                    "late_seed_rule": "train_late_seed + zero_based_update*B + group_slot*K + repeat_index",
                    "shuffle_seed": int(rr.train_shuffle_seed) + global_step,
                },
                rank=rank,
            )
        embed_cache = _encode_update_cache(pipeline, local_batches, device, dtype)
        _park_t5(pipeline)

        raw_values, normalized_values, active_values = [], [], []
        prompt_group_values = []
        sigma_values, late_index_values, loss_values = [], [], []
        local_late_states = 0
        for micro_index, records in enumerate(local_batches):
            prompts = [record["prompt"] for record in records]
            embeds = torch.cat([embed_cache[int(record["dataset_index"])][0] for record in records])
            pooled = torch.cat([embed_cache[int(record["dataset_index"])][1] for record in records])
            generators = [
                torch.Generator(device=device).manual_seed(int(record["noise_seed"])) for record in records
            ]
            if len(generators) == 1:
                generators = generators[0]
            if len(records) != 1:
                raise AssertionError("truncated-prefix rollout requires one trajectory per micro-batch")
            num_steps = int(config.sample.num_steps)
            late_indices = [
                choose_late_index(num_steps, float(rr.late_fraction), int(record["late_seed"]))
                for record in records
            ]
            model.module.eval()
            with torch.autocast("cuda", dtype=dtype):
                _images, latents = _rollout(
                    pipeline, embeds, pooled, config, generators=generators,
                    num_steps=int(config.sample.num_steps), solver=str(config.sample.solver),
                    stop_before_index=late_indices[0], decode=False,
                )
            if _images is not None or len(latents) != late_indices[0] + 1:
                raise AssertionError("training rollout executed beyond the sampled late state")
            model.train()
            should_sync = micro_index == len(local_batches) - 1
            with maybe_no_sync(model, should_sync):
                x0, sigma = _late_x0(model, latents, pipeline.scheduler.sigmas, embeds, pooled, late_indices, dtype)
                refs = _refs(records, str(rr.pairwise_train_ref_root), reward, device)
                raw_reward = runtime._reward_of_latents_grad(
                    pipeline, scorer, reward, x0, prompts, ref=refs
                ).float()
                normalized = (
                    (raw_reward - calibration_mean) / calibration_std
                    if bool(rr.standardize_reward) else raw_reward
                )
                hinge = torch.relu(float(rr.hinge_margin) - normalized)
                loss = float(rr.grad_scale) * hinge.mean()
                scaled_loss = loss / accum
                profiling.reward_bwd_inc()
                profiling.train_bwd_inc()
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
            raw_values.append(raw_reward.detach())
            prompt_group_values.append(torch.tensor(
                [record["group_slot"] for record in records], device=device, dtype=torch.long
            ))
            normalized_values.append(normalized.detach())
            active_values.append((hinge.detach() > 0).float())
            sigma_values.append(sigma.detach())
            late_index_values.append(torch.tensor(late_indices, device=device, dtype=torch.float32))
            loss_values.append(loss.detach().reshape(1))
            local_late_states += len(records)

        if local_late_states != int(rr.trajectories_per_update) // policy_world:
            raise AssertionError("each local trajectory must contribute exactly one late state")
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(trainable, float(rr.max_grad_norm))
        grad_norm = float(grad_norm_tensor.detach())
        if bool(rr.fail_on_nonfinite_or_zero_grad):
            require_finite_nonzero("gradient norm", grad_norm)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        ema.step(trainable, global_step)
        cumulative_trajectories += int(rr.trajectories_per_update)

        step_seconds_tensor = torch.tensor(time.perf_counter() - step_started, device=device)
        dist.all_reduce(step_seconds_tensor, op=dist.ReduceOp.MAX, group=group)
        cumulative_train_seconds += float(step_seconds_tensor)
        raw_stats = distributed_moments(raw_values, group)
        norm_stats = distributed_moments(normalized_values, group)
        active_stats = distributed_moments(active_values, group)
        sigma_stats = distributed_moments(sigma_values, group)
        late_stats = distributed_moments(late_index_values, group)
        loss_stats = distributed_moments(loss_values, group)
        dispersion_stats = distributed_prompt_group_dispersion(
            torch.cat(prompt_group_values),
            torch.cat([value.reshape(-1) for value in raw_values]),
            group,
            rank=rank,
        )
        grad_sync = sync_mean(torch.tensor(grad_norm, device=device), group)
        wall_seconds = time.time() - train_loop_started
        record = {
            "step": global_step,
            "kind": "train",
            **_flatten_metrics("raw_reward", raw_stats),
            **_flatten_metrics("standardized_reward", norm_stats),
            "hinge_active_frac": active_stats["mean"],
            **_flatten_metrics("late_sigma", sigma_stats),
            **_flatten_metrics("late_timestep_index", late_stats),
            "loss": loss_stats["mean"],
            "grad_norm": float(grad_sync),
            "step_seconds": float(step_seconds_tensor),
            "cumulative_train_seconds": cumulative_train_seconds,
            "training_compute_gpu_hours": (
                cumulative_train_seconds * int(rr.actual_compute_gpu_count) / 3600.0
            ),
            "training_reserved_gpu_hours": (
                cumulative_train_seconds * int(rr.reserved_gpu_count) / 3600.0
            ),
            "actual_compute_gpu_count": int(rr.actual_compute_gpu_count),
            "reserved_gpu_count": int(rr.reserved_gpu_count),
            "denoiser_forwards_mean": late_stats["mean"] + 1.0,
            "skipped_suffix_forwards_mean": int(config.sample.num_steps) - late_stats["mean"] - 1.0,
            "late_state_count": int(rr.trajectories_per_update),
            "trajectories_this_update": int(rr.trajectories_per_update),
            "cumulative_trajectories": cumulative_trajectories,
            "optimizer_updates": global_step,
            "train_loop_elapsed_seconds_including_eval": wall_seconds,
            **dispersion_stats,
        }
        if rank == 0:
            runtime.wandb.log(record, step=global_step)
        if profiler is not None:
            profiler.epoch_end(profile_index, global_step=global_step)
            if profiler.done(profile_index):
                profiler.finalize()
                break

        if global_step % int(rr.checkpoint_every) == 0 or global_step == int(rr.num_updates):
            save_refl_checkpoint(
                config.save_dir,
                global_step=global_step,
                rank=rank,
                model=model,
                trainable_parameters=trainable,
                ema=ema,
                config=config,
                optimizer=optimizer,
                scaler=scaler,
                provenance=provenance,
                group=group,
            )
            dist.barrier(group=group, device_ids=[torch.cuda.current_device()])

        if global_step % int(rr.eval_every) == 0 or global_step == int(rr.num_updates):
            online_eval_started = time.perf_counter()
            ema.copy_ema_to(trainable, store_temp=True)
            try:
                result = _evaluate(
                    pipeline=pipeline, model=model.module, scorer=scorer, reward=reward,
                    encoded_batches=encoded_curve, manifest_records=curve_records,
                    config=config, device=device, dtype=dtype,
                    group=group, rank=rank, num_steps=40, solver="flow",
                    ref_root=str(rr.pairwise_eval_ref_root),
                    output_path=os.path.join(config.save_dir, "curve", f"update_{global_step:04d}.json"),
                    update=global_step, score_scale=score_scale,
                )
            finally:
                ema.copy_temp_to(trainable)
            if rank == 0 and result is not None:
                eval_seconds = time.perf_counter() - online_eval_started
                cumulative_online_eval_seconds += eval_seconds
                runtime.wandb.log(
                    {
                        "step": global_step,
                        "kind": "online_eval",
                        "resume_replay": False,
                        "eval_seconds": eval_seconds,
                        "cumulative_online_eval_seconds": cumulative_online_eval_seconds,
                        **result["stats"],
                    },
                    step=global_step,
                )

    expected_updates = global_step if profiler is not None else int(rr.num_updates)
    expected_total = expected_updates * int(rr.trajectories_per_update)
    if cumulative_trajectories != expected_total:
        raise AssertionError(f"trajectory total {cumulative_trajectories} != {expected_total}")
    if profiler is not None:
        if not profiler.finalized:
            profiler.finalize()
        if rank == 0:
            runtime.wandb.finish()
        if shutdown_bridge is not None:
            shutdown_bridge()
        runtime.cleanup_distributed()
        return
    final_eval_started = time.perf_counter()
    ema.copy_ema_to(trainable, store_temp=True)
    try:
        final_result = _evaluate(
            pipeline=pipeline, model=model.module, scorer=scorer, reward=reward,
            encoded_batches=encoded_final, manifest_records=final_records,
            config=config, device=device, dtype=dtype,
            group=group, rank=rank, num_steps=40, solver="flow",
            ref_root=str(rr.pairwise_eval_ref_root),
            output_path=os.path.join(config.save_dir, "final_eval", "result.json"),
            update="final", score_scale=score_scale,
        )
    finally:
        ema.copy_temp_to(trainable)
    if rank == 0 and final_result is not None:
        write_json(
            os.path.join(config.save_dir, "final_eval", "per_image.json"),
            final_result["per_image"], rank=rank,
        )
    final_eval_seconds = time.perf_counter() - final_eval_started
    write_json(
        os.path.join(config.save_dir, "run_done.json"),
        {
            "status": "PASS_PENDING_EXTERNAL_VALIDATOR",
            "global_step": global_step,
            "cumulative_trajectories": cumulative_trajectories,
            "checkpoint": f"checkpoints/checkpoint-{global_step}",
            "final_eval": "final_eval/result.json",
            "timing_seconds": {
                "pure_training": cumulative_train_seconds,
                "online_eval_including_update0": cumulative_online_eval_seconds,
                "final_eval": final_eval_seconds,
                "initialization": initialization_seconds,
                "calibration": calibration_seconds,
            },
            "training_compute_gpu_hours": (
                cumulative_train_seconds * int(rr.actual_compute_gpu_count) / 3600.0
            ),
            "training_reserved_gpu_hours": (
                cumulative_train_seconds * int(rr.reserved_gpu_count) / 3600.0
            ),
            "ended": _now(),
        },
        rank=rank,
    )
    if rank == 0:
        runtime.wandb.finish()
    if shutdown_bridge is not None:
        shutdown_bridge()
    runtime.cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
