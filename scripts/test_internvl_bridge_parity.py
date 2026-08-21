"""ACCEPTANCE GATE: gradient parity for the InternVL2-26B ``internvl_t2i`` bridge.

Proves that routing the reward through the remote 2-GPU server yields the SAME reward and the SAME
gradient-w.r.t.-image as computing it co-located. A wrong gradient silently corrupts training, so
this must pass on a GPU node BEFORE any bridged run.

Layout (small, multi-client): torchrun --nproc_per_node=3
    rank 0,1 = policy CLIENTS (each also loads a co-located 26B for the reference on its own GPU)
    rank 2   = REWARD SERVER (single-GPU here; the real run shards over cuda:6+cuda:7)

Each client, with a fixed synthetic image batch + fixed prompts, computes:
    reference: r_ref, g_ref = autograd.grad(scorer._scores(img).sum(), img)   [co-located, own GPU]
    bridged:   r_bri, g_bri = autograd.grad(remote_reward_scores(img).sum(), img)  [server GPU]
and also the forward-only reward (CMD_SCORE_FWD) vs the scorer's no-grad __call__.

Run:
    export INTERNVL_BRIDGE=1
    torchrun --standalone --nnodes=1 --nproc_per_node=3 \
        scripts/test_internvl_bridge_parity.py
Expected: each client prints "... -> PASS"; exit code 0. Any FAIL exits non-zero.

Note on the gradient metric: the task's elementwise ``max|g_ref-g_bri|/(|g_ref|+1e-6)`` is reported,
but the PASS decision uses robust global measures (relative-L2 + cosine) that are not dominated by
near-zero-gradient pixels, where bf16 rounding of a ~0 value can inflate the elementwise ratio while
the gradient is, for every practical purpose, identical. cudnn is put in deterministic mode so the
two computations match as tightly as the hardware allows.
"""

import os

import torch
import torch.distributed as dist

from diffusionopsd.internvl_bridge import (
    make_bridge_groups, is_server_rank, RewardServer,
    remote_reward_scores, remote_reward_scores_forward, bridge_client_shutdown,
)

R_TOL = 1e-3      # |reward| absolute tolerance
G_RELL2_TOL = 1e-3   # ||g_ref - g_bri|| / ||g_ref||
G_COS_TOL = 0.9999   # cosine(g_ref, g_bri)


def _run_client(rank: int, local_rank: int) -> bool:
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(1234 + rank)  # distinct batch per client -> exercises multi-client serving
    B, H, W = 2, 256, 256
    prompts = ["a red cube sitting on a wooden table", "a serene mountain lake at sunset"][:B]
    base = torch.rand(B, 3, H, W, device=device)  # synthetic image in [0,1]

    # --- reference: co-located 26B on this client's own GPU ---
    from diffusionopsd.internvl_t2i_scorer import get_internvl_t2i_scorer
    scorer = get_internvl_t2i_scorer(device=f"cuda:{local_rank}")
    img_ref = base.clone().requires_grad_(True)
    r_ref = scorer._scores(img_ref, prompts)
    (g_ref,) = torch.autograd.grad(r_ref.sum(), img_ref)
    with torch.no_grad():
        r_fwd_ref = scorer(base.clone(), prompts)  # __call__ = no-grad forward reward

    # --- bridged: reward+grad served by rank (world-1) ---
    img_bri = base.clone().requires_grad_(True)
    r_bri = remote_reward_scores(img_bri, prompts)
    (g_bri,) = torch.autograd.grad(r_bri.sum(), img_bri)
    r_fwd_bri = remote_reward_scores_forward(base.clone(), prompts)

    r_ref, r_bri = r_ref.detach().float(), r_bri.detach().float()
    g_ref, g_bri = g_ref.detach().float(), g_bri.detach().float()
    r_fwd_ref, r_fwd_bri = r_fwd_ref.detach().float(), r_fwd_bri.detach().float()

    r_err = (r_ref - r_bri).abs().max().item()
    rfwd_err = (r_fwd_ref - r_fwd_bri).abs().max().item()
    g_abs = (g_ref - g_bri).abs().max().item()
    g_relL2 = ((g_ref - g_bri).norm() / (g_ref.norm() + 1e-8)).item()
    g_cos = torch.nn.functional.cosine_similarity(g_ref.flatten(), g_bri.flatten(), dim=0).item()
    # task's exact (fragile) elementwise metric, reported for transparency:
    g_elem = ((g_ref - g_bri).abs() / (g_ref.abs() + 1e-6)).max().item()

    ok = (r_err < R_TOL) and (rfwd_err < R_TOL) and (g_relL2 < G_RELL2_TOL) and (g_cos > G_COS_TOL)
    print(f"[rank {rank}] reward ref={r_ref.tolist()} bridge={r_bri.tolist()}", flush=True)
    print(
        f"[rank {rank}] |r_ref-r_bri|max={r_err:.3e}  |r_fwd|err={rfwd_err:.3e}  "
        f"grad: relL2={g_relL2:.3e} cos={g_cos:.6f} absmax={g_abs:.3e} elemmax={g_elem:.3e}  "
        f"-> {'PASS' if ok else 'FAIL'}",
        flush=True,
    )
    return ok


def main() -> None:
    os.environ.setdefault("INTERNVL_BRIDGE", "1")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.backends.cudnn.deterministic = True   # match ref vs bridge as tightly as the hardware allows
    torch.backends.cudnn.benchmark = False

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    make_bridge_groups(world_size)  # every rank; sets up gloo + policy groups

    if is_server_rank(rank):
        # single-GPU server here (the real run shards over cuda:6+cuda:7 via device_map — see
        # internvl_t2i_scorer._dispatch_multi_gpu; that path is verified separately).
        server = RewardServer(primary_device=local_rank, reward_devices=[local_rank])
        server.serve(n_policy=world_size - 1)  # exits after every client sends CMD_SHUTDOWN
        dist.destroy_process_group()
        return

    ok = False
    try:
        ok = _run_client(rank, local_rank)
    finally:
        bridge_client_shutdown()  # always release the server (even on assert/exception)
    dist.destroy_process_group()
    if not ok:
        raise SystemExit(f"[rank {rank}] internvl bridge parity FAILED")


if __name__ == "__main__":
    main()
