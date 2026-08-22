"""In-node FORWARD-ONLY reward bridge for HEAVY Z-Image scorers (hpsv3 7B / deqa 8B).

Purpose
-------
The Z-Image-Turbo ``nft`` trainer co-locates the reward scorer on every policy GPU. That fits the
light public scorers next to the 6 B Z-Image DiT, but the heavy
VLM rewards OOM: ``hpsv3`` (Qwen2-VL-7B + ranknet, ~14 GB bf16) and ``deqa`` (mPLUG-Owl2-8B, ~16 GB
bf16) cannot be colocated reliably with the 6 B DiT. This module hosts the heavy
scorer on a dedicated SERVER rank (its own GPU) and lets the policy ranks obtain scalar rewards over
the wire:

    ranks 0..N-2  = POLICY  (Z-Image DDP training; NO heavy scorer resident)
    rank  N-1     = REWARD SERVER (owns ONE GPU holding the 7 B / 8 B scorer)

Everything here is INERT unless ``ZIMAGE_HEAVY_BRIDGE=1``; the trainer only wires it in behind that
flag, so co-located (light-reward) runs are byte-for-byte unchanged.

``nft`` uses the reward ONLY as a forward, no-grad scalar (group-relative advantage); it never
backprops through the reward. So this bridge ships images -> scalar rewards and NEVER computes or
returns a gradient. No ``autograd.Function``, no grad tensor on the wire, one scoring command.

GPU layout (the 7 B/8 B public scorers fit on one server GPU)
------------------------------------------------------------
    torchrun --nproc_per_node=7  ->  ranks 0..5 = policy (cuda:0..5), rank 6 = server (cuda:6)
The policy world size is therefore 6; the trainer + config batch math use n_policy_gpus = 6 (which
keeps the K=12 group recipe's batch search valid — 7 policy GPUs would not, since gcd(7,12)=1).
Overridable: ``ZIMAGE_HEAVY_BRIDGE_SERVER_RANK`` (default last rank),
``ZIMAGE_HEAVY_BRIDGE_SERVER_DEVICES`` (default ``[local_rank]``, single GPU).

Two process groups
------------------
* ``_GLOO_GROUP``   — gloo over ALL ranks: the small control header (recv-from-ANY on the server) +
  the prompt string list + the shutdown barrier. NCCL cannot send Python objects nor recv-from-any.
* ``_POLICY_GROUP`` — NCCL over policy ranks only: ALL policy DDP/all_gather/broadcast/all_reduce/
  barrier collectives use this group, so the server (never a member) is excluded and can always make
  progress serving the next request.
The heavy image/reward TENSORS go over the default NCCL world via point-to-point send/recv, disjoint
from the policy collective communicator so a P2P and a collective can never contend.

Deadlock-avoidance
------------------
1. The server never joins a policy collective, so ``recv(src=None)`` always progresses.
2. A policy rank blocks on the server only INSIDE a reward call, never while inside a policy
   collective: the trainer scores all rollout batches (P2P) and drains every ``rewards_future``
   BEFORE the reward all_gather / DDP backward, so no rank is ever simultaneously "in a collective
   waiting for peer X" while "peer X waits for the server".
3. Each client request is one strict, fully-matched sequence (header -> prompts -> image -> reward);
   a process-wide ``_LOCK`` makes it atomic so concurrent executor worker-thread reward calls cannot
   interleave on the wire.
"""

import os
import threading
from typing import List, Optional, Sequence

import torch
import torch.distributed as dist

# --- control-protocol constants (header = int64[5] = [cmd, B, C, H, W]) ------------------------
CMD_SHUTDOWN = 0    # policy rank is done; server exits after all policy ranks send this
CMD_SCORE_FWD = 1   # forward-only: reward [B] (no autograd) — the scoring command used in training
CMD_ECHO = 2        # debug/parity: server echoes the received image back bit-for-bit (no scoring)
_HEADER_LEN = 5

# --- module state (populated by make_bridge_groups; None => bridge inactive) --------------------
_GLOO_GROUP = None            # gloo PG over all ranks (control + prompts + shutdown barrier)
_POLICY_GROUP = None          # NCCL PG over policy ranks only (DDP + policy collectives)
_SERVER_RANK: Optional[int] = None
_POLICY_RANKS: List[int] = []
_LOCK = threading.Lock()      # serialises a client request so it is atomic on the wire


def zimage_bridge_enabled() -> bool:
    """True iff ZIMAGE_HEAVY_BRIDGE=1. Guards every bridge branch; unset => classic co-located path."""
    return os.environ.get("ZIMAGE_HEAVY_BRIDGE", "0") == "1"


def _server_rank_for(world_size: int) -> int:
    return int(os.environ.get("ZIMAGE_HEAVY_BRIDGE_SERVER_RANK", str(world_size - 1)))


def is_server_rank(rank: int) -> bool:
    """The reward-server rank (last rank by default). Overridable via ZIMAGE_HEAVY_BRIDGE_SERVER_RANK."""
    return rank == _SERVER_RANK


def policy_group():
    """NCCL subgroup for policy collectives (None when bridge inactive => default world)."""
    return _POLICY_GROUP


def make_bridge_groups(world_size: int):
    """Create the gloo (all-rank) and policy (NCCL, server-excluded) subgroups.

    MUST be called by EVERY rank (new_group is itself a collective over the default world), in the
    same order, right after ``dist.init_process_group``. Returns (gloo_group, policy_group).
    """
    global _GLOO_GROUP, _POLICY_GROUP, _SERVER_RANK, _POLICY_RANKS
    _SERVER_RANK = _server_rank_for(world_size)
    _POLICY_RANKS = [r for r in range(world_size) if r != _SERVER_RANK]
    # gloo over ALL ranks: control headers (recv-from-any), prompt objects, shutdown barrier.
    _GLOO_GROUP = dist.new_group(ranks=list(range(world_size)), backend="gloo", timeout=__import__("datetime").timedelta(hours=6))  # >30min default: server recv must survive the ~33min/epoch optimize phase between reward calls
    # NCCL over policy ranks only: DDP + all_gather + broadcast + all_reduce + barrier (server excluded).
    _POLICY_GROUP = dist.new_group(ranks=_POLICY_RANKS)
    # Eagerly initialise the default WORLD NCCL communicator across ALL ranks now (every rank calls
    # this fn). Later the WORLD group carries ONLY policy<->server point-to-point tensor transfers;
    # forcing its init here means a P2P never triggers a lazy all-ranks communicator handshake in the
    # middle of training (where only 2 ranks are active) — a classic hang.
    # device_ids pins this all-ranks barrier to THIS rank's GPU. Without it, newer torch reports
    # "devices used by this process are currently unknown" and this 7-rank eager barrier can HANG the
    # node (the exact startup hang observed on the Z-Image ISO env). current_device() == the rank's
    # local GPU (setup_distributed's set_device already ran before make_bridge_groups).
    dist.barrier(device_ids=[torch.cuda.current_device()])
    return _GLOO_GROUP, _POLICY_GROUP


def bridge_server_devices(local_rank: int) -> List[int]:
    """GPU index(es) the server hosts the scorer on. Default: single ``[local_rank]`` — a 7 B/8 B
    scorer in bf16 (~14-16 GB) fits comfortably on one 80 GB card. Overridable via
    ZIMAGE_HEAVY_BRIDGE_SERVER_DEVICES="6,7"."""
    env = os.environ.get("ZIMAGE_HEAVY_BRIDGE_SERVER_DEVICES", "")
    if env.strip():
        return [int(x) for x in env.split(",") if x.strip() != ""]
    return [local_rank]


# ============================ client (policy rank) ============================================
@torch.no_grad()
def _client_request(images01: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
    """Ship one image batch to the server, receive the scalar reward [B]. Atomic under ``_LOCK``.

    Strict request->response so it cannot deadlock or interleave (see module docstring):
      client: send header (gloo) -> send prompts (gloo) -> send image (nccl) -> recv reward (nccl)
      server: recv header (gloo,any) -> recv prompts (gloo) -> recv image (nccl) -> send reward (nccl)
    """
    assert _GLOO_GROUP is not None and _SERVER_RANK is not None, "bridge groups not initialised"
    if isinstance(prompts, str):
        prompts = [prompts]
    prompts = list(prompts)
    if images01.dim() == 3:
        images01 = images01.unsqueeze(0)
    dev = images01.device
    assert images01.dim() == 4, f"bridge expects image [B,3,H,W], got {tuple(images01.shape)}"
    B, C, H, W = (int(s) for s in images01.shape)
    assert len(prompts) == B, f"bridge: {B} images vs {len(prompts)} prompts"
    header = torch.tensor([CMD_SCORE_FWD, B, C, H, W], dtype=torch.int64)  # CPU tensor for gloo
    img = images01.detach().contiguous().float()          # [B,C,H,W] float32 on cuda:local (bit-exact P2P)
    reward = torch.empty(B, dtype=torch.float32, device=dev)
    # _LOCK makes the whole exchange atomic (reward calls run in executor worker THREADS; without it
    # two threads would interleave on the wire and corrupt the header/prompt framing). torch.cuda.device
    # pins this thread's CUDA context to the tensor's GPU for the NCCL P2P.
    with _LOCK, torch.cuda.device(dev):
        dist.send(header, dst=_SERVER_RANK, group=_GLOO_GROUP)
        dist.send_object_list([prompts], dst=_SERVER_RANK, group=_GLOO_GROUP,
                              device=torch.device("cpu"))
        dist.send(img, dst=_SERVER_RANK)                  # default world (NCCL P2P)
        dist.recv(reward, src=_SERVER_RANK)
    return reward


@torch.no_grad()
def bridge_reward(images01: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
    """Forward-only reward [B] for local decoded ``images01`` [B,3,H,W] in [0,1], served remotely.
    Drop-in for the co-located ``scorer(images01, prompts)`` no-grad ``__call__`` at the nft rollout/
    eval reward-scoring call sites. Returns a float tensor [B] on ``images01.device``."""
    return _client_request(images01, prompts)


@torch.no_grad()
def bridge_echo(images01: torch.Tensor) -> torch.Tensor:
    """DEBUG/PARITY transport-integrity probe: ship ``images01`` and receive it back UNMODIFIED from
    the server (no scoring). ``torch.equal(images01.float().contiguous(), bridge_echo(images01))``
    proves the NCCL P2P round-trip is bit-exact — and since the server would score exactly this
    ``img_buf``, it isolates transport correctness from the (non-deterministic) scorer forward. Not
    used in training; the parity test calls it to distinguish 'bridge corrupts the image' from
    'the scorer is stochastic'."""
    assert _GLOO_GROUP is not None and _SERVER_RANK is not None, "bridge groups not initialised"
    if images01.dim() == 3:
        images01 = images01.unsqueeze(0)
    dev = images01.device
    B, C, H, W = (int(s) for s in images01.shape)
    header = torch.tensor([CMD_ECHO, B, C, H, W], dtype=torch.int64)
    img = images01.detach().contiguous().float()
    echoed = torch.empty(B, C, H, W, dtype=torch.float32, device=dev)
    with _LOCK, torch.cuda.device(dev):
        dist.send(header, dst=_SERVER_RANK, group=_GLOO_GROUP)
        dist.send_object_list([[]], dst=_SERVER_RANK, group=_GLOO_GROUP,   # empty prompts: keep the
                              device=torch.device("cpu"))                  # header->prompts->image sequence uniform
        dist.send(img, dst=_SERVER_RANK)
        dist.recv(echoed, src=_SERVER_RANK)
    return echoed


def bridge_client_shutdown() -> None:
    """Tell the server this policy rank is done, then rendezvous with all ranks on the gloo barrier.
    Called by every policy rank at the very end of training (after the final policy barrier)."""
    header = torch.tensor([CMD_SHUTDOWN, 0, 0, 0, 0], dtype=torch.int64)
    with _LOCK:
        dist.send(header, dst=_SERVER_RANK, group=_GLOO_GROUP)
    dist.barrier(group=_GLOO_GROUP)  # matched by the server after its serve loop breaks


# ============================ server (last rank) =============================================
class HeavyRewardServer:
    """Owns the frozen heavy scorer (hpsv3 / deqa) and serves scalar rewards to the policy ranks.

    Single-threaded serve loop => no server-side locking. Loads the scorer on a single GPU via its
    process-level ``get_*_scorer`` singleton, exactly the model the co-located path would build.
    """

    def __init__(self, primary_device: int, reward_kind: str):
        self.reward_dev = torch.device(f"cuda:{primary_device}")
        torch.cuda.set_device(primary_device)
        self.reward_kind = reward_kind
        dev = f"cuda:{primary_device}"
        if reward_kind == "hpsv3":
            from diffusionopsd.hpsv3_scorer import get_hpsv3_scorer
            self.scorer = get_hpsv3_scorer(device=dev)
        elif reward_kind == "deqa":
            from diffusionopsd.deqa_scorer import get_deqa_scorer
            self.scorer = get_deqa_scorer(device=dev)
        else:
            raise ValueError(
                f"[zimage_reward_bridge] server got unsupported heavy reward_kind='{reward_kind}' "
                f"(expected 'hpsv3' or 'deqa')")

    @torch.no_grad()
    def serve(self, n_policy: int) -> None:
        """Serve until all ``n_policy`` policy ranks have sent CMD_SHUTDOWN, then gloo-barrier."""
        assert _GLOO_GROUP is not None
        header = torch.empty(_HEADER_LEN, dtype=torch.int64)  # CPU tensor for gloo
        shutdowns = 0
        while True:
            src = dist.recv(header, src=None, group=_GLOO_GROUP)  # recv-from-ANY policy rank
            cmd = int(header[0].item())
            if cmd == CMD_SHUTDOWN:
                shutdowns += 1
                if shutdowns >= n_policy:
                    break
                continue
            if cmd not in (CMD_SCORE_FWD, CMD_ECHO):
                raise ValueError(f"[zimage_reward_bridge] server got unknown cmd={cmd}")
            B, C, H, W = (int(header[i].item()) for i in range(1, 5))
            objs: List = [None]
            dist.recv_object_list(objs, src=src, group=_GLOO_GROUP, device=torch.device("cpu"))
            prompts = objs[0]
            img_buf = torch.empty(B, C, H, W, dtype=torch.float32, device=self.reward_dev)
            dist.recv(img_buf, src=src)  # default world (NCCL P2P); lands on the reward GPU
            if cmd == CMD_ECHO:
                # transport-integrity probe: send the received image straight back, bit-for-bit. The
                # client's torch.equal(sent, echoed) then proves the P2P round-trip is lossless.
                dist.send(img_buf.contiguous(), dst=src)
                continue
            # Identical to the co-located reward path: the scorer's no-grad __call__ (hpsv3 chunks
            # internally; deqa is no-reference and ignores prompts). Same model + same pixels => the
            # SAME reward the policy rank would have computed co-located.
            r = self.scorer(img_buf, prompts)
            dist.send(r.detach().float().contiguous().to(self.reward_dev), dst=src)
        dist.barrier(group=_GLOO_GROUP)  # rendezvous with policy ranks' bridge_client_shutdown()
