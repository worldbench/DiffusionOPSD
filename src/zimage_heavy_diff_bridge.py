"""In-node DIFFERENTIABLE reward bridge for HEAVY Z-Image scorers (hpsv3 7B / deqa 8B).

Purpose
-------
The Z-Image OPSD/ReFL trainers need the reward as a *differentiable* function of the locally
decoded image: they backprop the reward gradient into the policy (OPSD reward-gradient target
ascent / ReFL direct reward-gradient step). Co-located on every policy GPU this fits the light
scorers next to the 6 B Z-Image DiT, but the heavy public VLM
rewards OOM on the BACKWARD: hpsv3 (Qwen2-VL-7B + ranknet, ~14 GB bf16) and deqa (mPLUG-Owl2-8B,
~16 GB bf16) blow up when autograd retains the 7 B/8 B activations for the gradient-to-image next
to the 6 B DiT. This module hosts the heavy scorer on a dedicated SERVER rank (its own single GPU)
and lets the policy ranks obtain reward AND its gradient over the wire:

    ranks 0..N-2  = POLICY  (Z-Image DDP training; NO heavy scorer resident)
    rank  N-1     = REWARD SERVER (owns ONE GPU holding the 7 B / 8 B scorer)

The policy ranks call the reward as an ordinary differentiable function of their locally decoded
image; under the hood the image is shipped to the server, which computes reward+gradient and ships
them back. Everything here is INERT unless ZIMAGE_HEAVY_DIFF_BRIDGE=1 — the trainer only wires it
in behind that flag, so co-located (light-reward) runs are byte-for-byte unchanged.

This is the DIFFERENTIABLE analogue of zimage_reward_bridge (forward-only, for the nft/flowgrpo
baselines that never backprop the reward). It reuses the single-GPU heavy layout of
zimage_reward_bridge with a differentiable transport protocol.

Two process groups
-------------------
* _GLOO_GROUP   — gloo over ALL ranks: control header (recv-from-ANY on the server) + prompt list +
  shutdown barrier. NCCL is tensor-only; gloo carries Python objects and recv-from-any.
* _POLICY_GROUP — NCCL over policy ranks only: ALL policy DDP/collectives use this, excluding the
  server so it can always progress serving the next request.
The heavy image/reward/gradient tensors go over the default NCCL world via P2P send/recv, disjoint
from the policy collective communicator, so a P2P and a policy collective never contend.

Deadlock-avoidance
------------------
1. The server never joins a policy collective, so its gloo recv(src=None) always progresses.
2. A policy rank blocks on the server only inside a reward call, never inside a policy collective:
   the trainer drains every reward BEFORE any reward all_gather / DDP backward.
3. Each request is one strict, fully-matched sequence (header -> prompts -> image -> reward
   [-> grad]); a process-wide _LOCK makes it atomic so executor-thread forward calls can't
   interleave on the wire.
"""

import os
import threading
from datetime import timedelta
from typing import List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

# --- control-protocol constants (header = int64[5] = [cmd, B, C, H, W]) -------------------------
CMD_SHUTDOWN = 0    # policy rank is done; server exits after all policy ranks send this
CMD_SCORE_GRAD = 1  # reward [B] AND d(reward_b)/d(image[b]) [B,C,H,W]  (OPSD/ReFL grad)
CMD_SCORE_FWD = 2   # reward [B] only (no autograd) — cheap rollout/eval scoring
_HEADER_LEN = 5

# --- module state (populated by make_bridge_groups; None => bridge inactive) --------------------
_GLOO_GROUP = None            # gloo PG over all ranks (control + prompts + shutdown barrier)
_POLICY_GROUP = None          # NCCL PG over policy ranks only (DDP + policy collectives)
_SERVER_RANK: Optional[int] = None
_POLICY_RANKS: List[int] = []
_LOCK = threading.Lock()      # serialises a client request so it is atomic on the wire


def zimage_heavy_diff_bridge_enabled() -> bool:
    """True iff ZIMAGE_HEAVY_DIFF_BRIDGE=1."""
    return os.environ.get("ZIMAGE_HEAVY_DIFF_BRIDGE", "0") == "1"


def _server_rank_for(world_size: int) -> int:
    return int(os.environ.get("ZIMAGE_HEAVY_DIFF_BRIDGE_SERVER_RANK", str(world_size - 1)))


def is_server_rank(rank: int) -> bool:
    """The reward-server rank (last rank by default). Overridable via ZIMAGE_HEAVY_DIFF_BRIDGE_SERVER_RANK."""
    return rank == _SERVER_RANK


def policy_group():
    """NCCL subgroup for policy collectives (None when bridge inactive => default world)."""
    return _POLICY_GROUP


def make_bridge_groups(world_size: int):
    """Create the gloo (all-rank) and policy (NCCL, server-excluded) subgroups. MUST be called by
    EVERY rank right after dist.init_process_group, in the same order (new_group is itself a world
    collective). Returns (gloo_group, policy_group)."""
    global _GLOO_GROUP, _POLICY_GROUP, _SERVER_RANK, _POLICY_RANKS
    _SERVER_RANK = _server_rank_for(world_size)
    _POLICY_RANKS = [r for r in range(world_size) if r != _SERVER_RANK]
    # gloo over ALL ranks (control headers recv-from-any, prompt objects, shutdown barrier). 6 h
    # timeout (>> the 30 min gloo default): between two reward calls the server's recv must survive
    # the long Z-Image optimize phase, exactly as in zimage_reward_bridge.
    _GLOO_GROUP = dist.new_group(
        ranks=list(range(world_size)), backend="gloo", timeout=timedelta(hours=6))
    # NCCL over policy ranks only: DDP + all_gather + broadcast + all_reduce + barrier (server excluded).
    _POLICY_GROUP = dist.new_group(ranks=_POLICY_RANKS)
    # Eagerly init the default WORLD NCCL communicator now (every rank calls this). Later the WORLD
    # group carries ONLY policy<->server P2P transfers; forcing init here avoids a lazy all-ranks
    # handshake mid-training (a classic hang). device_ids pins this all-ranks barrier to THIS rank's
    # GPU; without it newer torch reports "devices used by this process are currently unknown" and
    # this eager barrier can HANG the node (the Z-Image ISO-env startup hang). set_device already ran.
    dist.barrier(device_ids=[torch.cuda.current_device()])
    return _GLOO_GROUP, _POLICY_GROUP


def bridge_server_devices(local_rank: int) -> List[int]:
    """GPU index(es) the server hosts the scorer on. Default single [local_rank] — a 7 B/8 B scorer
    in bf16 (~14-16 GB) fits on one 80 GB card. Overridable via
    ZIMAGE_HEAVY_DIFF_BRIDGE_SERVER_DEVICES="6,7"."""
    env = os.environ.get("ZIMAGE_HEAVY_DIFF_BRIDGE_SERVER_DEVICES", "")
    if env.strip():
        return [int(x) for x in env.split(",") if x.strip() != ""]
    return [local_rank]


# ============================ client (policy rank) ============================================
def _client_request(images01: torch.Tensor, prompts: Sequence[str], want_grad: bool
                    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Ship one image batch to the server, receive reward (+grad). Atomic under _LOCK.
      client: send header(gloo) -> send prompts(gloo) -> send image(nccl) -> recv reward(nccl)
              [-> recv grad(nccl)]
      server: recv header(gloo,any) -> recv prompts(gloo) -> recv image(nccl) -> send reward(nccl)
              [-> send grad(nccl)]
    Requires a 4-D [B,3,H,W] batch (the autograd backward returns a grad of the exact input rank; no
    implicit unsqueeze)."""
    assert _GLOO_GROUP is not None and _SERVER_RANK is not None, "bridge groups not initialised"
    if isinstance(prompts, str):
        prompts = [prompts]
    prompts = list(prompts)
    dev = images01.device
    assert images01.dim() == 4, f"bridge expects image [B,3,H,W], got {tuple(images01.shape)}"
    B, C, H, W = (int(s) for s in images01.shape)
    assert len(prompts) == B, f"bridge: {B} images vs {len(prompts)} prompts"
    cmd = CMD_SCORE_GRAD if want_grad else CMD_SCORE_FWD
    header = torch.tensor([cmd, B, C, H, W], dtype=torch.int64)   # CPU tensor for gloo
    img = images01.detach().contiguous().float()                 # [B,C,H,W] float32 on cuda:local
    reward = torch.empty(B, dtype=torch.float32, device=dev)
    grad = torch.empty(B, C, H, W, dtype=torch.float32, device=dev) if want_grad else None
    # _LOCK makes the whole exchange atomic (forward-reward calls run in executor worker THREADS;
    # without it two threads would interleave on the wire and corrupt the header/prompt framing).
    # torch.cuda.device pins this thread's CUDA context to the tensor's GPU for the NCCL P2P.
    with _LOCK, torch.cuda.device(dev):
        dist.send(header, dst=_SERVER_RANK, group=_GLOO_GROUP)
        dist.send_object_list([prompts], dst=_SERVER_RANK, group=_GLOO_GROUP,
                              device=torch.device("cpu"))
        dist.send(img, dst=_SERVER_RANK)                         # default world (NCCL P2P)
        dist.recv(reward, src=_SERVER_RANK)
        if want_grad:
            dist.recv(grad, src=_SERVER_RANK)
    return reward, grad


class RemoteHeavyDiffReward(torch.autograd.Function):
    """Differentiable drop-in for scorer._scores(images01, prompts) served by the remote heavy VLM.

    Reward r_b depends ONLY on image[b] (per-sample independence), so the reward Jacobian
    d(reward)/d(image) is block-diagonal and the server returns grad_img[b] = d r_b / d image[b].
    For upstream loss L with grad_output[b] = dL/d r_b, the VJP is
        dL/d image[b] = grad_output[b] * grad_img[b],
    which backward returns as grad_output.view(-1,1,1,1) * grad_img — bit-for-bit the gradient a
    co-located autograd.grad(reward, image) would produce."""

    @staticmethod
    def forward(ctx, images01, prompts):
        reward, grad_img = _client_request(images01, prompts, want_grad=True)
        ctx.save_for_backward(grad_img)
        return reward  # [B] on images01.device, wired into autograd via this Function

    @staticmethod
    def backward(ctx, grad_output):
        (grad_img,) = ctx.saved_tensors
        grad_in = grad_output.view(-1, 1, 1, 1) * grad_img
        return grad_in, None  # (grad wrt images01, grad wrt prompts=None)


def remote_heavy_reward_scores(images01: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
    """Differentiable reward [B] for local decoded images01 [B,3,H,W] in [0,1], served remotely.
    Drop-in for the heavy scorer's _scores at the OPSD/ReFL reward-gradient call sites."""
    return RemoteHeavyDiffReward.apply(images01, prompts)


@torch.no_grad()
def remote_heavy_reward_forward(images01: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
    """Forward-only reward [B] (rollout/eval scoring). No autograd graph, no server backward."""
    reward, _ = _client_request(images01, prompts, want_grad=False)
    return reward


def bridge_client_shutdown() -> None:
    """Tell the server this policy rank is done, then rendezvous with all ranks on the gloo barrier.
    Called by every policy rank at the very end of training (after the final policy barrier)."""
    header = torch.tensor([CMD_SHUTDOWN, 0, 0, 0, 0], dtype=torch.int64)
    with _LOCK:
        dist.send(header, dst=_SERVER_RANK, group=_GLOO_GROUP)
    dist.barrier(group=_GLOO_GROUP)  # matched by the server after its serve loop breaks


# ============================ server (last rank) =============================================
class HeavyDiffRewardServer:
    """Owns the frozen heavy scorer (hpsv3 / deqa) on ONE GPU and serves reward+gradient to the
    policy ranks. Single-threaded serve loop => no server-side lock. serve() is NOT wrapped in
    torch.no_grad (unlike the forward-only zimage_reward_bridge server) because CMD_SCORE_GRAD must
    run the scorer with grad enabled; the CMD_SCORE_FWD branch takes its own torch.no_grad."""

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
                f"[zimage_heavy_diff_bridge] server got unsupported heavy reward_kind='{reward_kind}' "
                f"(expected 'hpsv3' or 'deqa')")

    def serve(self, n_policy: int) -> None:
        """Serve until all n_policy policy ranks have sent CMD_SHUTDOWN, then gloo-barrier."""
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
            if cmd not in (CMD_SCORE_GRAD, CMD_SCORE_FWD):
                raise ValueError(f"[zimage_heavy_diff_bridge] server got unknown cmd={cmd}")
            B, C, H, W = (int(header[i].item()) for i in range(1, 5))
            objs: List = [None]
            dist.recv_object_list(objs, src=src, group=_GLOO_GROUP, device=torch.device("cpu"))
            prompts = objs[0]
            img_buf = torch.empty(B, C, H, W, dtype=torch.float32, device=self.reward_dev)
            dist.recv(img_buf, src=src)  # default world (NCCL P2P); lands on the reward GPU
            if cmd == CMD_SCORE_GRAD:
                # grad-enabled: same model + same pixels the policy rank would use co-located, so the
                # returned reward and per-sample gradient equal the co-located autograd's. requires_grad
                # on a detached leaf => autograd.grad gives d(sum r)/d img (block-diagonal per sample).
                image = img_buf.detach().requires_grad_(True)
                r = self.scorer._scores(image, prompts)        # [B], grad graph intact (bf16 model)
                (g,) = torch.autograd.grad(r.sum(), image)     # d(sum r)/d image = per-sample block
                dist.send(r.detach().float().contiguous().to(self.reward_dev), dst=src)
                dist.send(g.detach().float().contiguous().to(self.reward_dev), dst=src)
            else:  # CMD_SCORE_FWD
                with torch.no_grad():
                    r = self.scorer._scores(img_buf, prompts)
                dist.send(r.detach().float().contiguous().to(self.reward_dev), dst=src)
        dist.barrier(group=_GLOO_GROUP)  # rendezvous with policy ranks' bridge_client_shutdown()
