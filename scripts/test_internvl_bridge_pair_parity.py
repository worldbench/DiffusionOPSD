"""ACCEPTANCE GATE: gradient parity for the PAIRWISE ``internvl_dual`` bridge.

Proves that routing the pairwise reward P(gen>ref) through the remote 2-GPU server yields the SAME
reward and the SAME gradient-w.r.t.-GEN as co-located. Mirror of ``test_internvl_bridge_parity.py``
(pointwise) for the dual scorer. The reference is a fixed no-grad conditioning image — only the
GENERATED image carries gradient, so we compare grad wrt gen only. A wrong gradient silently
corrupts training, so this must pass on a GPU node BEFORE any bridged internvl_dual run.

Layout (single-server, K=1, torchrun --nproc_per_node=3):
    rank 0,1 = policy CLIENTS (each also loads a co-located dual 26B for the reference on its own GPU)
    rank 2   = REWARD SERVER (single-GPU here; the real K=1 run shards over cuda:6+cuda:7)

Multi-server (K = INTERNVL_BRIDGE_SERVERS > 1) mirrors the real training layout on an 8-GPU node:
    export INTERNVL_BRIDGE=1 INTERNVL_BRIDGE_SERVERS=2
    torchrun --standalone --nnodes=1 --nproc_per_node=6 scripts/test_internvl_bridge_pair_parity.py
    => ranks 0..3 = policy CLIENTS (cuda:0..3), ranks 4,5 = REWARD SERVERS sharded over [4,5] and [6,7].
    Each client round-robins to one server (2 clients/server), so this exercises the multi-client-per-
    server serve loop and the _MY_SERVER routing that a routing/shutdown-count bug would DEADLOCK on.
    Reward+grad stay bit-identical (each server runs the same 26B) => PASS also re-verifies correctness.

Run (single-server):
    export INTERNVL_BRIDGE=1
    torchrun --standalone --nnodes=1 --nproc_per_node=3 \
        scripts/test_internvl_bridge_pair_parity.py
Expected: each client prints "... -> PASS"; exit code 0. Any FAIL exits non-zero.
"""

import os

import torch
import torch.distributed as dist

from diffusionopsd.internvl_bridge import (
    make_bridge_groups, is_server_rank, RewardServer,
    remote_reward_scores_pair, remote_reward_scores_pair_forward, bridge_client_shutdown,
    num_servers, server_ranks_for, bridge_server_devices_for, policy_count_for_server,
)

R_TOL = 1e-3         # |reward| absolute tolerance
G_RELL2_TOL = 1e-3   # ||g_ref - g_bri|| / ||g_ref||
G_COS_TOL = 0.9999   # cosine(g_ref, g_bri)


def _run_client(rank: int, local_rank: int) -> bool:
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(1234 + rank)  # distinct batch per client -> exercises multi-client serving
    B, H, W = 2, 256, 256
    prompts = ["a red cube sitting on a wooden table", "a serene mountain lake at sunset"][:B]
    gen_base = torch.rand(B, 3, H, W, device=device)   # synthetic generated image in [0,1]
    ref_base = torch.rand(B, 3, H, W, device=device)   # fixed reference image (no grad)

    # --- reference: co-located dual 26B on this client's own GPU ---
    from diffusionopsd.internvl_dual_scorer import get_internvl_dual_scorer
    scorer = get_internvl_dual_scorer(device=f"cuda:{local_rank}")
    gen_ref = gen_base.clone().requires_grad_(True)
    r_ref = scorer._scores(gen_ref, ref_base, prompts)
    (g_ref,) = torch.autograd.grad(r_ref.sum(), gen_ref)   # grad wrt gen only (ref detached in scorer)
    with torch.no_grad():
        r_fwd_ref = scorer._scores(gen_base.clone(), ref_base, prompts)

    # --- bridged: reward+grad(gen) served by rank (world-1) ---
    gen_bri = gen_base.clone().requires_grad_(True)
    r_bri = remote_reward_scores_pair(gen_bri, ref_base, prompts)
    (g_bri,) = torch.autograd.grad(r_bri.sum(), gen_bri)
    r_fwd_bri = remote_reward_scores_pair_forward(gen_base.clone(), ref_base, prompts)

    r_ref, r_bri = r_ref.detach().float(), r_bri.detach().float()
    g_ref, g_bri = g_ref.detach().float(), g_bri.detach().float()
    r_fwd_ref, r_fwd_bri = r_fwd_ref.detach().float(), r_fwd_bri.detach().float()

    r_err = (r_ref - r_bri).abs().max().item()
    rfwd_err = (r_fwd_ref - r_fwd_bri).abs().max().item()
    g_abs = (g_ref - g_bri).abs().max().item()
    g_relL2 = ((g_ref - g_bri).norm() / (g_ref.norm() + 1e-8)).item()
    g_cos = torch.nn.functional.cosine_similarity(g_ref.flatten(), g_bri.flatten(), dim=0).item()

    ok = (r_err < R_TOL) and (rfwd_err < R_TOL) and (g_relL2 < G_RELL2_TOL) and (g_cos > G_COS_TOL)
    print(f"[rank {rank}] PAIR reward ref={r_ref.tolist()} bridge={r_bri.tolist()}", flush=True)
    print(
        f"[rank {rank}] |r_ref-r_bri|max={r_err:.3e}  |r_fwd|err={rfwd_err:.3e}  "
        f"grad(gen): relL2={g_relL2:.3e} cos={g_cos:.6f} absmax={g_abs:.3e}  "
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

    # Multi-server (K = INTERNVL_BRIDGE_SERVERS > 1): a server rank binds to its assigned GPU block's
    # base BEFORE init, mirroring train_opsd_ri_sd3.py. K=1 is a no-op (keeps the original nproc=3
    # single-GPU-server layout: server on cuda:LOCAL_RANK, serving all world-1 clients).
    K = num_servers()
    if K > 1 and rank in server_ranks_for(world_size):
        local_rank = bridge_server_devices_for(rank, world_size)[0]

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    make_bridge_groups(world_size)  # every rank; sets up gloo + policy groups

    if is_server_rank(rank):
        if K > 1:
            sdev = bridge_server_devices_for(rank, world_size)          # this server's own 2-GPU block
            n_pol = policy_count_for_server(rank, world_size)           # only its round-robin clients
        else:
            sdev = [local_rank]                                        # original single-GPU server
            n_pol = world_size - 1
        server = RewardServer(primary_device=sdev[0], reward_devices=sdev,
                              reward_kind="internvl_dual")
        server.serve(n_policy=n_pol)  # exits after its assigned clients each send CMD_SHUTDOWN
        dist.destroy_process_group()
        return

    ok = False
    try:
        ok = _run_client(rank, local_rank)
    finally:
        bridge_client_shutdown()  # always release the server (even on assert/exception)
    dist.destroy_process_group()
    if not ok:
        raise SystemExit(f"[rank {rank}] internvl_dual bridge parity FAILED")


if __name__ == "__main__":
    main()
