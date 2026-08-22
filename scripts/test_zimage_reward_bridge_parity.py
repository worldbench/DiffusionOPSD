"""ACCEPTANCE GATE + DIAGNOSTIC: reward parity for the Z-Image HEAVY forward-only reward bridge.

Proves that routing a heavy scorer (hpsv3 7B / deqa 8B) through the dedicated reward SERVER is
CORRECT — i.e. the server scores exactly the pixels the policy rank shipped, and the reward comes
back intact. nft uses the reward forward-only, so there is no gradient to check.

Why a bit-exact reward gate is the WRONG test here
--------------------------------------------------
hpsv3's scorer runs a 7B Qwen2-VL in train() mode with gradient checkpointing; its attention uses
flash-attn / SDPA bf16 kernels whose reductions are NON-DETERMINISTIC (and NOT covered by
cudnn.deterministic, nor seed-controllable). So even the CO-LOCATED scorer returns slightly
different rewards for the SAME image on two calls. A naive |r_ref - r_bri| < 1e-3 gate then FAILS
for a perfectly correct bridge, purely because r_ref (client GPU) and r_bri (server GPU) are two
independent noisy draws.

So this test separates TRANSPORT correctness from SCORER noise, and gates on transport:
  A. SELF-NOISE  — score the SAME image twice with ONE co-located scorer on ONE GPU. This measures
     the scorer's intrinsic run-to-run jitter (sigma_self). If it is ~0.1-0.3 on the ~[-10,10] mu
     scale, non-determinism is proven and it fully explains the coloc-vs-bridge diffs.
  B. TRANSPORT   — CMD_ECHO ships the image and gets it back; torch.equal(sent, echoed) proves the
     NCCL P2P round-trip is BIT-EXACT. Since the server scores exactly that img_buf, this is an
     airtight proof the bridge delivers the right pixels (rules out dtype / range [0,1]vs[-1,1] /
     layout / double-preprocess corruption — any of which would make echo != sent).
  C. REWARD      — coloc r_ref vs bridge r_bri, reported and sanity-checked against sigma_self.

PASS = transport bit-exact AND |r_ref - r_bri| within the scorer's own self-noise envelope
(5*sigma_self, floored at R_ATOL). This does NOT hide a real bug: a transport bug fails (B); a
server-side scorer mismatch (different model/config) biases r_bri by >> sigma_self and fails (C).

Layout (torchrun --nproc_per_node=3):
    rank 0,1 = policy CLIENTS (each loads a co-located scorer for the reference on its own GPU)
    rank 2   = REWARD SERVER (single-GPU; the real run puts it on rank 6 / cuda:6)

Run (pick the heavy reward to host on the server):
    export ZIMAGE_HEAVY_BRIDGE=1
    export ZIMAGE_BRIDGE_PARITY_REWARD=hpsv3   # or deqa
    torchrun --standalone --nnodes=1 --nproc_per_node=3 \
        scripts/test_zimage_reward_bridge_parity.py
Expected: each client prints "-> PASS"; exit 0. Any FAIL exits non-zero.
"""

import os

import torch
import torch.distributed as dist

from diffusionopsd.zimage_reward_bridge import (
    make_bridge_groups, is_server_rank, HeavyRewardServer,
    bridge_reward, bridge_echo, bridge_client_shutdown,
)

R_ATOL = 2e-3    # absolute floor for the reward-parity envelope (guards the ~0 self-noise edge case)
NOISE_K = 5.0    # coloc-vs-bridge reward diff must be within NOISE_K * sigma_self (two noisy draws)


def _load_colocated_scorer(kind: str, local_rank: int):
    dev = f"cuda:{local_rank}"
    if kind == "hpsv3":
        from diffusionopsd.hpsv3_scorer import get_hpsv3_scorer
        return get_hpsv3_scorer(device=dev)
    if kind == "deqa":
        from diffusionopsd.deqa_scorer import get_deqa_scorer
        return get_deqa_scorer(device=dev)
    raise ValueError(f"ZIMAGE_BRIDGE_PARITY_REWARD must be 'hpsv3' or 'deqa', got '{kind}'")


def _run_client(rank: int, local_rank: int, kind: str) -> bool:
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(1234 + rank)  # distinct batch per client -> exercises multi-client serving
    B, H, W = 2, 512, 512
    prompts = ["a red cube sitting on a wooden table", "a serene mountain lake at sunset"][:B]
    base = torch.rand(B, 3, H, W, device=device).float()  # synthetic image in [0,1], fp32
    scorer = _load_colocated_scorer(kind, local_rank)

    # --- A. SELF-NOISE: same image scored TWICE by ONE co-located scorer (same GPU, no bridge) ---
    with torch.no_grad():
        r_a = scorer(base.clone(), prompts).detach().float()
        r_b = scorer(base.clone(), prompts).detach().float()
    self_noise = (r_a - r_b).abs()
    sigma_self = self_noise.max().item()

    # --- B. TRANSPORT: bit-exact image round-trip through the bridge (CMD_ECHO) ---
    sent = base.detach().contiguous().float()
    echoed = bridge_echo(base.clone())
    transport_ok = bool(torch.equal(sent, echoed))
    max_pix_diff = (sent - echoed).abs().max().item()

    # --- C. REWARD: co-located reference vs bridge (independent noisy draws of the same scorer) ---
    r_ref = r_a  # reuse one co-located draw as the reference
    r_bri = bridge_reward(base.clone(), prompts).detach().float()
    reward_diff = (r_ref - r_bri).abs()

    tol = max(NOISE_K * sigma_self, R_ATOL)
    reward_ok = reward_diff.max().item() <= tol
    ok = transport_ok and reward_ok

    nd = ("CONFIRMED (>>1e-3): a bit-exact reward gate is invalid for this scorer"
          if sigma_self > 1e-3 else "not observed (scorer looks deterministic here)")
    tr = ("CLEAN: server scores the exact shipped pixels"
          if transport_ok else "CORRUPT: bridge altered the image in transit (REAL BUG)")
    print(f"[rank {rank}] kind={kind}", flush=True)
    print(f"[rank {rank}] A self-noise (same img x2, co-located): {self_noise.tolist()} "
          f"sigma_self={sigma_self:.3e}  -> non-determinism {nd}", flush=True)
    print(f"[rank {rank}] B transport echo bit-exact={transport_ok}  max|sent-echo|={max_pix_diff:.3e}"
          f"  -> {tr}", flush=True)
    print(f"[rank {rank}] C reward ref={r_ref.tolist()} bridge={r_bri.tolist()} "
          f"diff={reward_diff.tolist()} max={reward_diff.max().item():.3e} tol(5*sigma)={tol:.3e}",
          flush=True)
    print(f"[rank {rank}] VERDICT transport_ok={transport_ok} reward_within_noise={reward_ok} "
          f"-> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main() -> None:
    os.environ.setdefault("ZIMAGE_HEAVY_BRIDGE", "1")
    kind = os.environ.get("ZIMAGE_BRIDGE_PARITY_REWARD", "hpsv3")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.backends.cudnn.deterministic = True   # tightens (does NOT eliminate) flash-attn/bf16 jitter
    torch.backends.cudnn.benchmark = False

    # Mirror the trainer's robust init: bind the GPU BEFORE init + pass device_id so the all-ranks
    # eager barrier in make_bridge_groups can't hit the "devices unknown" hang (guarded for old torch).
    torch.cuda.set_device(local_rank)
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size,
                                device_id=torch.device(f"cuda:{local_rank}"))
    except TypeError:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    make_bridge_groups(world_size)  # every rank; sets up gloo + policy groups

    if is_server_rank(rank):
        server = HeavyRewardServer(primary_device=local_rank, reward_kind=kind)
        server.serve(n_policy=world_size - 1)  # exits after every client sends CMD_SHUTDOWN
        dist.destroy_process_group()
        return

    ok = False
    try:
        ok = _run_client(rank, local_rank, kind)
    finally:
        bridge_client_shutdown()  # always release the server (even on assert/exception)
    dist.destroy_process_group()
    if not ok:
        raise SystemExit(f"[rank {rank}] zimage heavy reward bridge parity FAILED")


if __name__ == "__main__":
    main()
