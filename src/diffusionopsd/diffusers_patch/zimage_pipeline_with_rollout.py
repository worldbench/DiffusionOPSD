# Z-Image-Turbo rollout pipeline for DiffusionOPSD / DiffusionNFT.
#
# This is the Z-Image analog of `pipeline_with_logprob.py` + `solver.py` for SD3.
# It re-implements the ZImagePipeline 9-step FlowMatchEuler denoising loop so the
# trainers can capture per-step (z_t, sigma, raw transformer output v_raw) and later
# re-run the (LoRA) policy at a stored state with the SAME sign/time convention.
#
# ============================ AUTHORITATIVE CONVENTIONS ============================
# (verified against the upstream Diffusers `pipeline_z_image.py` implementation)
#
#  * Transformer is LIST-based:
#        x        = list of per-sample (C, 1, H, W)   [pipeline does latents.unsqueeze(2).unbind(0)]
#        t        = timestep tensor
#        cap_feats= list of per-sample (seq_i, 2560)  [Qwen3 penultimate hidden state]
#    forward(x, t, cap_feats, return_dict=False)[0] -> list of per-sample (C, 1, H, W).
#
#  * SIGN: the pipeline NEGATES the transformer output before the scheduler step:
#        v_raw     = transformer(...)                  (raw output)
#        noise_pred = -v_raw                           (= diffusers-convention velocity v)
#    Relations:  x0_pred = z_t - sigma * v = z_t - sigma*noise_pred = z_t + sigma*v_raw.
#
#  * TIME: the transformer timestep input is data-progress, NOT the scheduler sigma:
#        t_model = (1000 - t_sched) / 1000 = 1 - sigma   (t_sched = scheduler timestep = sigma*1000)
#
#  * Euler step (deterministic FlowMatchEuler, stochastic_sampling=False):
#        z_next = z_t + (sigma_next - sigma) * noise_pred
#               = z_t + (sigma - sigma_next) * v_raw
#        last sigma_next = 0  ->  z_final = x0_pred.
#
#  * VAE (FLUX AutoencoderKL): scaling_factor=0.3611, shift_factor=0.1159.
#        latents/0.3611 + 0.1159 -> vae.decode -> [-1,1] -> map to [0,1].
#
#  * Sigma schedule (shift=3.0, static): read from scheduler.sigmas; do NOT hardcode.
#        ~[1.000, 0.960, 0.913, 0.857, 0.789, 0.706, 0.600, 0.462, 0.273, 0.000].
#    OPA query state: argmin|sigma - 0.27| -> step 8, sigma~=0.273.
#
#  * guidance_scale = 0.0  ->  ZImagePipeline.do_classifier_free_guidance is False (0 > 0 == False)
#    -> single forward, NO CFG, NO negative prompt. This module implements ONLY that path.
#
# =================================== SMOKE TESTS ===================================
# Before any training:
#  [S1] NFE: print(len(scheduler.timesteps)) and hook-count transformer calls (8 vs 9).
#  [S2] Native ZImagePipeline sample OK at 1024/9-steps/gs=0.
#  [S3] BIT-FOR-BIT: verify `z_t + (sigma-sigma_next)*v_raw` (this module's Euler) reproduces the
#       native pipeline latents. This validates the sign + time reproduction end to end.
#  [S4] HPS grad w.r.t. y0 nonzero (zimage_decode -> HPS -> autograd.grad).
#  [S6] `enable_thinking=True` cached embeds from zimage_encode_prompt match pipe.encode_prompt.
# ==================================================================================

from typing import List, Optional, Union
import torch

# NOTE: `ZImagePipeline` / `ZImageTransformer2DModel` require a Diffusers build that
# includes Z-Image support. The trainers import the pipeline symbol lazily; this
# module operates on an already constructed `pipe`.


# --------------------------------------------------------------------------------- #
#  Text encoding (Qwen3, penultimate hidden, enable_thinking=True, max_len=512)      #
# --------------------------------------------------------------------------------- #
@torch.no_grad()
def zimage_encode_prompt(pipe, prompts: Union[str, List[str]], device,
                         max_sequence_length: int = 512) -> List[torch.Tensor]:
    """Return a LIST of per-prompt caption features (seq_i, 2560).

    Deterministic given the prompt -> cache per prompt. Wraps ZImagePipeline.encode_prompt
    with do_classifier_free_guidance=False (gs=0 -> no negative branch). The chat template
    with enable_thinking=True and the penultimate-layer readout live inside encode_prompt.

    [SMOKE S6] Confirm the returned list matches what ZImagePipeline.__call__ builds
    internally (arg name / order of encode_prompt may differ across the two source PRs;
    if encode_prompt returns a padded tensor + mask instead of a list, adapt here).
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    prompt_embeds, _neg = pipe.encode_prompt(
        prompt=list(prompts),
        device=device,
        do_classifier_free_guidance=False,
        max_sequence_length=max_sequence_length,
    )
    # encode_prompt returns a list (one variable-length (seq_i, 2560) per prompt).
    return [p.to(device) for p in prompt_embeds]


# --------------------------------------------------------------------------------- #
#  Raw transformer forward (reproduces the pipeline's pack / unpack exactly)         #
# --------------------------------------------------------------------------------- #
def _transformer_v_raw(model, latents: torch.Tensor, t_model: torch.Tensor,
                       cap_feats_list: List[torch.Tensor]) -> torch.Tensor:
    """Run the S3-DiT once and return the RAW output v_raw (B, C, H, W), un-negated.

    `model` is either `pipeline.transformer` (no-grad rollout / old / base) or the DDP
    wrapper `transformer_ddp` (trainable policy, so gradients sync). Both share the same
    PEFT-injected LoRA modules; the active adapter is controlled by the caller via
    set_adapter/disable_adapter, exactly as in the SD3 trainer.

    Pack/unpack mirrors diffusers pipeline_z_image.py:
        latent_model_input = latents.unsqueeze(2)              # (B,C,1,H,W)
        latent_model_input_list = list(...unbind(dim=0))       # [ (C,1,H,W) ] * B
        out_list = transformer(x_list, t, cap_feats, return_dict=False)[0]
        noise_pred = stack(out_list).squeeze(2)                # (B,C,H,W)

    [SMOKE] exact forward signature: the diffusers def is
        forward(self, x, t, cap_feats, return_dict=True, ...)
    with x/cap_feats list-based. If the built model exposes keyword names, this positional
    call still works. If `t` must be shape (B,) vs scalar vs per-sample, see zimage_v.
    """
    # ★ Match native ZImagePipeline.__call__ exactly: it casts latents to the TRANSFORMER's dtype
    # (`latents.to(self.transformer.dtype)`), NOT the prompt-embed dtype. Feeding embed-dtype (e.g. fp32)
    # latents to a bf16 transformer perturbs the output at step 0 and COMPOUNDS over the few-step schedule
    # (this was the S3 bit-for-bit bug: max|Δ| grew 0.128→3.58). Use the transformer's parameter dtype.
    try:
        model_dtype = (model.module if hasattr(model, "module") else model).dtype
    except AttributeError:
        model_dtype = next((model.module if hasattr(model, "module") else model).parameters()).dtype
    x = latents.to(model_dtype).unsqueeze(2)              # (B, C, 1, H, W)
    x_list = list(x.unbind(dim=0))                        # list of (C, 1, H, W)
    out = model(x_list, t_model, cap_feats_list, return_dict=False)[0]
    if isinstance(out, (list, tuple)):
        v_raw = torch.stack([o for o in out], dim=0)      # (B, C, 1, H, W)
    else:
        v_raw = out
    if v_raw.dim() == 5 and v_raw.shape[2] == 1:
        v_raw = v_raw.squeeze(2)                          # (B, C, H, W)
    return v_raw.float()


def zimage_v(model, latents: torch.Tensor, sigma, cap_feats_list: List[torch.Tensor]) -> torch.Tensor:
    """Diffusers-convention velocity  v = noise_pred = -v_raw  at state `latents`, noise `sigma`.

    THIS IS THE SIGN FIX. Everywhere the SD3 trainer used the transformer output directly
    (forward_prediction / old_prediction / v_theta / v_old), the Z-Image trainer must use
    `zimage_v(...)` so that x0_pred = latents - sigma * v holds with the correct sign.

    `sigma` may be a python float (shared across the batch, e.g. the OPA query sigma) or a
    (B,) tensor of per-sample sigmas (NFT forward-noising with per-sample shuffled timesteps).
    The transformer receives t_model = 1 - sigma (data-progress).
    """
    B = latents.shape[0]
    dev = latents.device
    if torch.is_tensor(sigma):
        sig = sigma.to(dev).float().reshape(-1)
        if sig.numel() == 1:
            sig = sig.expand(B)
    else:
        sig = torch.full((B,), float(sigma), device=dev, dtype=torch.float32)
    t_model = 1.0 - sig                                    # (B,) data-progress timestep
    v_raw = _transformer_v_raw(model, latents, t_model, cap_feats_list)
    return -v_raw


def recompute_v_theta(model, z_t: torch.Tensor, sigma, cap_feats_list: List[torch.Tensor]) -> torch.Tensor:
    """Training-time re-run of the (LoRA) policy at a stored rollout state z_t.

    Returns the diffusers-convention velocity v_theta = -v_raw, so the caller computes
    y_theta = z_t - sigma * v_theta  (x0-space branch target). Thin alias of zimage_v kept
    as a named entry point to match the task spec (the Z-Image analog of re-running the
    transformer inside pipeline_with_logprob for the NFT/OPA loss).
    """
    return zimage_v(model, z_t, sigma, cap_feats_list)


# --------------------------------------------------------------------------------- #
#  VAE decode (FLUX AutoencoderKL: 0.3611 / 0.1159) -> [0,1] for the reward model    #
# --------------------------------------------------------------------------------- #
def zimage_decode(vae, latents: torch.Tensor) -> torch.Tensor:
    """latents/scaling_factor + shift_factor -> vae.decode -> [-1,1] -> [0,1] (B,3,H,W).

    Uses vae.config.{scaling_factor, shift_factor} (0.3611 / 0.1159 for the Z-Image FLUX VAE).
    Skips PIL: returns a float tensor in [0,1] directly consumable by HPSv2Scorer. Kept
    differentiable (no @no_grad) so the OPA reward ascent can backprop decode->HPS.
    """
    lat = latents / vae.config.scaling_factor + vae.config.shift_factor
    img = vae.decode(lat.to(vae.dtype), return_dict=False)[0]
    return (img / 2 + 0.5).clamp(0, 1).float()


# --------------------------------------------------------------------------------- #
#  Query-state selection: argmin|sigma - target| over the STATE sigmas (exclude 0)   #
# --------------------------------------------------------------------------------- #
def zimage_query_sigma_index(sigmas: torch.Tensor, target: float = 0.27) -> int:
    """Index of the rollout state whose sigma is closest to `target` (default 0.27).

    `sigmas` is pipeline.scheduler.sigmas (len num_steps+1, last ~0). States are z_0..z_{n-1}
    with sigmas sigmas[:-1]; we search those (the final sigma=0 endpoint is x0, never a query
    state). For the shift=3 9-step schedule this returns 8 (sigma~=0.273). DO NOT hardcode.
    """
    sig = sigmas.detach().float()
    return int(torch.argmin((sig[:-1] - float(target)).abs()).item())


# --------------------------------------------------------------------------------- #
#  Rollout: 9-step deterministic FlowMatchEuler, storing per-step z_t / sigma / v_raw #
# --------------------------------------------------------------------------------- #
@torch.no_grad()
def zimage_rollout(pipe, prompt_embeds_list: List[torch.Tensor], num_inference_steps: int,
                   height: int, width: int, device, generator: Optional[torch.Generator] = None,
                   guidance_scale: float = 0.0, decode: bool = True,
                   latents: Optional[torch.Tensor] = None,
                   stop_before_index: Optional[int] = None) -> dict:
    """Run the Turbo denoising loop and return per-step rollout tensors + decoded images.

    Returns a dict:
        latents   : list[(B,C,H,W)] length num_steps+1  -> z_0 (pure noise) .. z_final (x0)
        v_raw     : list[(B,C,H,W)] length num_steps     -> raw transformer output per step
        sigmas    : (num_steps+1,) scheduler sigma schedule (last ~0)
        timesteps : (num_steps,)   scheduler timesteps (= sigma*1000)
        x0        : (B,C,H,W) = latents[-1]  (clean endpoint)
        images    : (B,3,H,W) in [0,1]  (present iff decode=True)

    Uses `pipe.transformer` directly (the active adapter is set by the caller, e.g. "old"
    for the on-policy rollout). Deterministic ODE -> no SDE branch / log-probs (OPA and NFT
    do not need per-step log-probs, unlike DDPO/GRPO).

    [SMOKE S1/S2/S3] The number of iterations equals len(scheduler.timesteps); verify it is
    9 (or 8) and that this loop reproduces the native ZImagePipeline latents bit-for-bit.
    """
    assert float(guidance_scale) == 0.0, (
        "Z-Image-Turbo is native gs=0 (single forward, no CFG / negative prompt). "
        "The spec forbids CFG; this rollout implements only the gs=0 path.")

    transformer = pipe.transformer
    scheduler = pipe.scheduler
    B = len(prompt_embeds_list)

    # in_channels=16 for the Z-Image S3-DiT; prepare_latents makes (B,16,H/8,W/8).
    # `latents` may be passed in (debug / fixed-init reproduction); else sample fresh.
    if latents is None:
        num_channels_latents = transformer.config.in_channels
        latents = pipe.prepare_latents(
            B, num_channels_latents, height, width,
            prompt_embeds_list[0].dtype, device, generator,
        )
        # prepare_latents may return (latents,) or latents; normalize.
        if isinstance(latents, (list, tuple)):
            latents = latents[0]

    # Static shift=3.0 (use_dynamic_shifting=False) -> set_timesteps needs no mu.
    # [SMOKE] If the native pipeline feeds a custom `sigmas`/`mu` into retrieve_timesteps,
    # replicate it here; the invariant to check is scheduler.sigmas == spec schedule.
    scheduler.set_timesteps(num_inference_steps, device=device)
    sigmas = scheduler.sigmas.to(device).float()          # (num_steps+1,), last ~0
    timesteps = scheduler.timesteps.to(device).float()    # (num_steps,)
    num_steps = len(timesteps)
    if stop_before_index is not None and not 0 <= int(stop_before_index) < num_steps:
        raise ValueError(
            f"stop_before_index must select a denoiser call in [0,{num_steps - 1}], "
            f"got {stop_before_index}"
        )
    steps_to_run = num_steps if stop_before_index is None else int(stop_before_index)

    z = latents.float()                                   # upcast for the Euler step
    all_latents = [z]
    all_v_raw = []
    for i in range(steps_to_run):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        t_sched = timesteps[i]
        # data-progress timestep the transformer expects: (1000 - t)/1000 = 1 - sigma
        t_model = torch.full((B,), (1000.0 - float(t_sched)) / 1000.0,
                             device=device, dtype=torch.float32)
        v_raw = _transformer_v_raw(transformer, z, t_model, prompt_embeds_list)
        noise_pred = -v_raw                               # pipeline negation
        z = z + (sigma_next - sigma) * noise_pred         # FlowMatchEuler deterministic step
        all_v_raw.append(v_raw)
        all_latents.append(z)

    out = {
        "latents": all_latents,
        "v_raw": all_v_raw,
        "sigmas": sigmas,
        "timesteps": timesteps,
        "completed_prefix_steps": steps_to_run,
    }
    if steps_to_run == num_steps:
        out["x0"] = all_latents[-1]
    if decode:
        if steps_to_run != num_steps:
            raise ValueError("cannot decode a truncated Z-Image prefix as an endpoint")
        out["images"] = zimage_decode(pipe.vae, all_latents[-1])
    return out
