"""In-node gradient bridge for the InternVL2-26B ``internvl_t2i`` reward.

Purpose
-------
The ``internvl_t2i`` reward is a frozen InternVL2-26B (~52 GB bf16). Co-located with the
SD3.5-M policy it *barely* fits on 8x80 GB and the OPA/ReFL **backward** (autograd through the
26 B to the decoded image) is the memory killer. This module lets a single node run as

    ranks 0..N-2  = POLICY  (SD3 DDP training; NO 26 B resident)
    rank  N-1     = REWARD SERVER (owns the 26 B, optionally sharded over 2 GPUs)

The policy ranks call the reward as an ordinary differentiable function of their locally decoded
image; under the hood the image is shipped to the server, which computes reward+gradient and
ships them back. Everything here is INERT unless ``INTERNVL_BRIDGE=1`` — the trainer only wires
it in behind that flag, so co-located runs are byte-for-byte unchanged.

Two process groups
------------------
``torchrun`` initialises the default **NCCL** world (all N ranks). On top of that we create:

* ``_GLOO_GROUP``   — gloo over ALL ranks. NCCL is tensor-only and cannot send Python objects or
  do "receive-from-ANY-source"; gloo can. Used for the small control header (recv-from-any on the
  server) + the prompt string list + the final shutdown barrier.
* ``_POLICY_GROUP`` — NCCL over ranks 0..N-2 only. ALL policy DDP/all_gather/broadcast/barrier
  collectives use this group, so the server (which never joins it) is excluded.

The heavy image/reward/gradient TENSORS go over the default NCCL world with point-to-point
send/recv (GPUDirect). The default world is used ONLY for P2P; policy collectives live on the
disjoint ``_POLICY_GROUP`` communicator, so a P2P and a collective can never contend on the same
NCCL communicator.

Deadlock-avoidance (see ``RewardServer.serve`` and ``_client_request``)
-----------------------------------------------------------------------
1. The server never participates in any policy collective, so it can always make progress serving
   whichever request arrives next (``recv(src=None)`` on gloo).
2. A policy rank only ever blocks on the server *inside* a reward call, at which point it is NOT
   inside a policy collective. Reward calls (P2P) and policy collectives (DDP all-reduce,
   all_gather, barrier) are separated into distinct phases of the epoch (rollout scoring →
   all_gather → OPA target build → DDP backward), and each phase is drained before the next by the
   trainer's ``rewards_future.result()`` / natural ordering. Hence no rank is ever simultaneously
   "in a collective waiting for peer X" and "peer X is waiting for the server".
3. Every client request is one strict, fully-matched sequence
   (header→prompts→image→reward[→grad]); the server executes the mirror sequence for exactly that
   source rank before recv-ing the next header. A process-wide lock makes each client request
   atomic so concurrent worker-thread forward-reward calls cannot interleave on the wire.
"""

import os
import threading
from datetime import timedelta
from typing import List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

# --- control-protocol constants (header = int64[8] = [cmd, B, C, H, W, rC, rH, rW]) ------------
# rC,rH,rW = reference-image dims for the PAIRWISE commands (0 for pointwise/shutdown).
CMD_SHUTDOWN = 0        # policy rank is done; server exits after all policy ranks send this
CMD_SCORE_GRAD = 1      # pointwise: reward [B] AND d(reward_b)/d(image[b]) [B,C,H,W]
CMD_SCORE_FWD = 2       # pointwise: reward [B] only (no autograd) — cheap rollout/eval scoring
CMD_SCORE_GRAD_PAIR = 3 # pairwise: reward [B]=P(gen>ref) AND d(reward_b)/d(gen[b]); ref is no-grad
CMD_SCORE_FWD_PAIR = 4  # pairwise: reward [B] only
_HEADER_LEN = 8

# --- module state (populated by make_bridge_groups; None => bridge inactive) --------------------
_GLOO_GROUP = None            # gloo PG over all ranks (control + prompts + shutdown barrier)
_POLICY_GROUP = None          # NCCL PG over policy ranks only (DDP + policy collectives)
_SERVER_RANKS: List[int] = []       # the last K ranks (K = INTERNVL_BRIDGE_SERVERS) = reward servers
_MY_SERVER: Optional[int] = None    # (policy rank only) the server rank THIS policy rank talks to
_POLICY_RANKS: List[int] = []
_LOCK = threading.Lock()      # serialises a client request so it is atomic on the wire


def bridge_enabled() -> bool:
    """True iff INTERNVL_BRIDGE=1. Guards every bridge branch; unset => classic co-located path."""
    return os.environ.get("INTERNVL_BRIDGE", "0") == "1"


def num_servers() -> int:
    """K = number of reward-server ranks (INTERNVL_BRIDGE_SERVERS, default 1 => original single
    server, byte-identical). K>1 runs K independently-sharded 26B servers so the per-sample reward
    forward+backward parallelises K-way across policy ranks (each policy rank talks to one server)."""
    return max(1, int(os.environ.get("INTERNVL_BRIDGE_SERVERS", "1")))


def server_ranks_for(world_size: int) -> List[int]:
    """The K server ranks = the LAST K ranks. Pure (no process group) so setup_distributed can call
    it before the bridge groups exist (to pick a server rank's primary GPU)."""
    K = num_servers()
    return list(range(world_size - K, world_size))


def server_for(policy_rank: int, world_size: int) -> int:
    """Which server rank a policy rank talks to — round-robin over the K servers for load balance."""
    srv = server_ranks_for(world_size)
    return srv[policy_rank % len(srv)]


def policy_count_for_server(server_rank: int, world_size: int) -> int:
    """Number of policy ranks assigned to this server (== the CMD_SHUTDOWN count its serve() waits for)."""
    srv = server_ranks_for(world_size)
    P = world_size - len(srv)
    i = srv.index(server_rank)
    return sum(1 for p in range(P) if p % len(srv) == i)


def is_server_rank(rank: int) -> bool:
    """True for the K reward-server ranks (the last K). Valid after make_bridge_groups; before that
    (e.g. the setup_distributed GPU pick) callers use ``rank in server_ranks_for(world_size)``."""
    return rank in _SERVER_RANKS


def policy_group():
    """NCCL subgroup for policy collectives (None when bridge inactive => default world)."""
    return _POLICY_GROUP


def make_bridge_groups(world_size: int):
    """Create the gloo (all-rank) and policy (NCCL, server-excluded) subgroups.

    MUST be called by EVERY rank (new_group is itself a collective over the default world), in the
    same order, right after ``dist.init_process_group``. Ranks not in ``_POLICY_GROUP`` (the server)
    receive a non-member sentinel handle they never use. Returns (gloo_group, policy_group).
    """
    global _GLOO_GROUP, _POLICY_GROUP, _SERVER_RANKS, _MY_SERVER, _POLICY_RANKS
    _SERVER_RANKS = server_ranks_for(world_size)
    _POLICY_RANKS = [r for r in range(world_size) if r not in _SERVER_RANKS]
    _my_rank = dist.get_rank()
    if _my_rank in _POLICY_RANKS:
        _MY_SERVER = server_for(_my_rank, world_size)   # this policy rank's assigned reward server
    # gloo over ALL ranks: control headers (recv-from-any), prompt objects, shutdown barrier.
    # Explicit 6h timeout: gloo new_group does NOT inherit init_process_group's timeout and would
    # otherwise default to 30 min -> the server's recv-from-any header blocks up to 30 min, and a
    # slow rollout/scoring phase (or a peer momentarily stuck in a policy collective) trips
    # "gloo ... Timed out waiting 1800000ms for recv" and kills the run (observed on nft/internvl_*).
    # 6h matches the NCCL groups (train_*_zimage.py init_process_group timeout) so all groups agree.
    _GLOO_GROUP = dist.new_group(ranks=list(range(world_size)), backend="gloo", timeout=timedelta(hours=6))
    # NCCL over policy ranks only: DDP + all_gather + broadcast + barrier (server excluded).
    _POLICY_GROUP = dist.new_group(ranks=_POLICY_RANKS)
    # Eagerly initialise the default WORLD NCCL communicator across ALL ranks now (every rank calls
    # this fn). Later the WORLD group carries ONLY policy<->server point-to-point tensor transfers;
    # forcing its init here means that P2P never triggers a lazy all-ranks communicator handshake in
    # the middle of training (where only 2 ranks are active) — a classic hang.
    dist.barrier()
    return _GLOO_GROUP, _POLICY_GROUP


def bridge_server_devices(local_rank: int) -> List[int]:
    """GPU indices the server shards the 26 B over. Default: [local_rank, local_rank+1] when a
    spare GPU exists (real 8-GPU node: server=rank6 -> [6,7]); else single [local_rank] (parity
    test / 1-GPU server). Overridable via INTERNVL_BRIDGE_SERVER_DEVICES="6,7"."""
    env = os.environ.get("INTERNVL_BRIDGE_SERVER_DEVICES", "")
    if env.strip():
        return [int(x) for x in env.split(",") if x.strip() != ""]
    n = torch.cuda.device_count()
    if local_rank + 1 < n:
        return [local_rank, local_rank + 1]
    return [local_rank]


def bridge_server_devices_for(server_rank: int, world_size: int) -> List[int]:
    """GPU block a given server shards its 26B over, for K>=1 servers. Policy ranks own cuda:0..P-1
    (P = world_size - K); the K servers split the remaining GPUs evenly, server i owning the g GPUs
    starting at P + i*g (g = (n_gpus - P)//K). K=1 => the single last rank owns [P .. n_gpus-1] ==
    the original 2-GPU [6,7] shard on an 8-GPU node, byte-identical. INTERNVL_BRIDGE_SERVER_DEVICES
    (single-server manual layout / parity test) still overrides when set."""
    env = os.environ.get("INTERNVL_BRIDGE_SERVER_DEVICES", "")
    if env.strip():
        return [int(x) for x in env.split(",") if x.strip() != ""]
    K = num_servers()
    P = world_size - K
    n = torch.cuda.device_count()
    g = max(1, (n - P) // K)
    i = server_ranks_for(world_size).index(server_rank)
    base = P + i * g
    return list(range(base, min(base + g, n)))


# ============================ client (policy rank) ============================================
def _client_request(images01: torch.Tensor, prompts: Sequence[str], want_grad: bool,
                    ref01: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Ship one image batch to the server, receive reward (+grad). Atomic under ``_LOCK``.

    Pointwise (ref01 None) and pairwise (ref01 given => a per-sample reference image, sent after
    the generated image; no grad flows to it). Ordering is a strict request→response so it cannot
    deadlock or interleave (see module docstring):
      client: send header (gloo) -> send prompts (gloo) -> send gen (nccl) [-> send ref (nccl)]
              -> recv reward (nccl) [-> recv grad (nccl)]
      server: recv header (gloo,any) -> recv prompts (gloo) -> recv gen (nccl) [-> recv ref (nccl)]
              -> send reward (nccl) [-> send grad (nccl)]
    """
    assert _GLOO_GROUP is not None and _MY_SERVER is not None, "bridge groups not initialised (policy rank)"
    if isinstance(prompts, str):
        prompts = [prompts]
    prompts = list(prompts)
    dev = images01.device
    assert images01.dim() == 4, f"bridge expects image [B,3,H,W], got {tuple(images01.shape)}"
    B, C, H, W = (int(s) for s in images01.shape)
    assert len(prompts) == B, f"bridge: {B} images vs {len(prompts)} prompts"
    is_pair = ref01 is not None
    if is_pair:
        ref = ref01.detach().contiguous().float().to(dev)       # reference image, no grad
        assert ref.dim() == 4 and int(ref.shape[0]) == B, \
            f"bridge pair: ref {tuple(ref.shape)} vs B={B}"
        rC, rH, rW = (int(s) for s in ref.shape[1:])
        cmd = CMD_SCORE_GRAD_PAIR if want_grad else CMD_SCORE_FWD_PAIR
    else:
        ref = None
        rC = rH = rW = 0
        cmd = CMD_SCORE_GRAD if want_grad else CMD_SCORE_FWD
    header = torch.tensor([cmd, B, C, H, W, rC, rH, rW], dtype=torch.int64)  # CPU tensor for gloo
    img = images01.detach().contiguous().float()                # [B,C,H,W] float32 on cuda:local
    reward = torch.empty(B, dtype=torch.float32, device=dev)
    grad = torch.empty(B, C, H, W, dtype=torch.float32, device=dev) if want_grad else None
    # _LOCK makes the whole exchange atomic (forward-reward calls run in executor worker THREADS;
    # without it two threads would interleave on the wire and corrupt the header/prompt framing).
    # torch.cuda.device pins this thread's CUDA context to the tensor's GPU for the NCCL P2P.
    with _LOCK, torch.cuda.device(dev):
        dist.send(header, dst=_MY_SERVER, group=_GLOO_GROUP)
        dist.send_object_list([prompts], dst=_MY_SERVER, group=_GLOO_GROUP,
                              device=torch.device("cpu"))
        dist.send(img, dst=_MY_SERVER)                           # default world (NCCL P2P)
        if is_pair:
            dist.send(ref, dst=_MY_SERVER)                       # reference image (no grad)
        dist.recv(reward, src=_MY_SERVER)
        if want_grad:
            dist.recv(grad, src=_MY_SERVER)
    return reward, grad


class RemoteInternVLReward(torch.autograd.Function):
    """Differentiable drop-in for ``scorer._scores(images01, prompts)`` served by the remote 26 B.

    Gradient math — why the bridged gradient equals the co-located one:
      The server returns ``grad_img = d(sum_b r_b)/d image``. Because reward r_b depends ONLY on
      image[b] (per-sample independence), the reward Jacobian d(reward)/d(image) is block-diagonal,
      so ``grad_img[b] = d r_b / d image[b]`` — exactly one diagonal block per sample. For an
      upstream loss L, the incoming ``grad_output[b] = dL/d r_b`` and the vector-Jacobian product is
          dL/d image[b] = (dL/d r_b) * (d r_b / d image[b]) = grad_output[b] * grad_img[b].
      ``backward`` therefore returns ``grad_output.view(-1,1,1,1) * grad_img`` (broadcasting the
      per-sample scalar over C,H,W), which is bit-for-bit the VJP a co-located
      ``autograd.grad(reward, image)`` would produce.
    """

    @staticmethod
    def forward(ctx, images01: torch.Tensor, prompts):
        reward, grad_img = _client_request(images01, prompts, want_grad=True)
        ctx.save_for_backward(grad_img)
        return reward  # [B] on images01.device, wired into autograd via this Function

    @staticmethod
    def backward(ctx, grad_output):
        (grad_img,) = ctx.saved_tensors
        grad_in = grad_output.view(-1, 1, 1, 1) * grad_img
        return grad_in, None  # (grad wrt images01, grad wrt prompts=None)


def remote_reward_scores(images01: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
    """Differentiable reward [B] for local decoded ``images01`` [B,3,H,W] in [0,1], served remotely.
    Drop-in for ``internvl_t2i`` ``scorer._scores`` at the OPA/ReFL reward-gradient call sites."""
    return RemoteInternVLReward.apply(images01, prompts)


@torch.no_grad()
def remote_reward_scores_forward(images01: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
    """Forward-only reward [B] (rollout/eval scoring). No autograd graph, no server backward."""
    reward, _ = _client_request(images01, prompts, want_grad=False)
    return reward


class RemoteInternVLPairReward(torch.autograd.Function):
    """Differentiable drop-in for the PAIRWISE ``dual_scorer._scores(gen01, ref01, prompts)`` served
    by the remote 26 B. Only the GENERATED image carries gradient; the reference is a fixed no-grad
    conditioning input, so backward returns grad wrt gen only. Same block-diagonal VJP as
    RemoteInternVLReward (reward r_b depends only on gen[b] and its fixed ref) => grad_gen[b] =
    d r_b / d gen[b], and dL/d gen[b] = grad_output[b] * grad_gen[b]."""

    @staticmethod
    def forward(ctx, gen01: torch.Tensor, ref01: torch.Tensor, prompts):
        reward, grad_gen = _client_request(gen01, prompts, want_grad=True, ref01=ref01)
        ctx.save_for_backward(grad_gen)
        return reward

    @staticmethod
    def backward(ctx, grad_output):
        (grad_gen,) = ctx.saved_tensors
        # (grad wrt gen01, grad wrt ref01=None [no-grad reference], grad wrt prompts=None)
        return grad_output.view(-1, 1, 1, 1) * grad_gen, None, None


def remote_reward_scores_pair(gen01: torch.Tensor, ref01: torch.Tensor,
                              prompts: Sequence[str]) -> torch.Tensor:
    """Differentiable pairwise reward [B]=P(gen>ref), served remotely. Drop-in for the dual
    ``scorer._scores(gen01, ref01, prompts)`` at the OPA/ReFL reward-gradient call sites."""
    return RemoteInternVLPairReward.apply(gen01, ref01, prompts)


@torch.no_grad()
def remote_reward_scores_pair_forward(gen01: torch.Tensor, ref01: torch.Tensor,
                                      prompts: Sequence[str]) -> torch.Tensor:
    """Forward-only pairwise reward [B] (rollout/eval scoring). No autograd graph, no server backward."""
    reward, _ = _client_request(gen01, prompts, want_grad=False, ref01=ref01)
    return reward


def bridge_client_shutdown() -> None:
    """Tell the server this policy rank is done, then rendezvous with all ranks on the gloo barrier.
    Called by every policy rank at the very end of training (after the final policy barrier)."""
    header = torch.tensor([CMD_SHUTDOWN, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int64)
    with _LOCK:
        dist.send(header, dst=_MY_SERVER, group=_GLOO_GROUP)
    dist.barrier(group=_GLOO_GROUP)  # all ranks (K servers + policy) rendezvous after serve loops break


# ============================ server (last rank) =============================================
class RewardServer:
    """Owns the frozen InternVL2-26B and serves reward+gradient to the policy ranks.

    Single-threaded serve loop => no server-side locking. Loads the scorer sharded over
    ``reward_devices`` when >1 (device_map path in internvl_t2i_scorer), else single-device.
    """

    def __init__(self, primary_device: int, reward_devices: List[int],
                 reward_kind: str = "internvl_t2i"):
        self.reward_dev = torch.device(f"cuda:{primary_device}")
        torch.cuda.set_device(primary_device)
        self.reward_kind = reward_kind
        # multi-GPU: shard the 26 B across the reward GPUs (backward-activation headroom); the dual
        # (pairwise, 2 images/call) reward is heavier and effectively REQUIRES the shard.
        multi = reward_devices if len(reward_devices) > 1 else None
        if reward_kind == "internvl_dual":
            from diffusionopsd.internvl_dual_scorer import get_internvl_dual_scorer
            self.scorer = get_internvl_dual_scorer(
                device=f"cuda:{primary_device}", device_map_devices=multi)
        else:
            from diffusionopsd.internvl_t2i_scorer import get_internvl_t2i_scorer
            if multi is not None:
                self.scorer = get_internvl_t2i_scorer(
                    device=f"cuda:{primary_device}", device_map_devices=multi)
            else:
                self.scorer = get_internvl_t2i_scorer(device=f"cuda:{primary_device}")

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
            B, C, H, W = (int(header[i].item()) for i in range(1, 5))
            rC, rH, rW = (int(header[i].item()) for i in range(5, 8))
            objs: List = [None]
            dist.recv_object_list(objs, src=src, group=_GLOO_GROUP, device=torch.device("cpu"))
            prompts = objs[0]
            img_buf = torch.empty(B, C, H, W, dtype=torch.float32, device=self.reward_dev)
            dist.recv(img_buf, src=src)  # default world (NCCL P2P); lands on the reward GPU
            ref_buf = None
            if cmd in (CMD_SCORE_GRAD_PAIR, CMD_SCORE_FWD_PAIR):
                ref_buf = torch.empty(B, rC, rH, rW, dtype=torch.float32, device=self.reward_dev)
                dist.recv(ref_buf, src=src)  # reference image (kept no-grad => no grad flows to ref)
            if cmd == CMD_SCORE_GRAD:
                image = img_buf.detach().requires_grad_(True)
                r = self.scorer._scores(image, prompts)            # [B], grad graph intact
                (g,) = torch.autograd.grad(r.sum(), image)         # d(sum r)/d image = per-sample block
                dist.send(r.detach().float().contiguous(), dst=src)
                dist.send(g.detach().float().contiguous(), dst=src)
            elif cmd == CMD_SCORE_FWD:
                with torch.no_grad():
                    r = self.scorer._scores(img_buf, prompts)
                dist.send(r.detach().float().contiguous(), dst=src)
            elif cmd == CMD_SCORE_GRAD_PAIR:
                gen = img_buf.detach().requires_grad_(True)
                r = self.scorer._scores(gen, ref_buf, prompts)     # pairwise; ref_buf no-grad
                (g,) = torch.autograd.grad(r.sum(), gen)           # grad wrt gen only
                dist.send(r.detach().float().contiguous(), dst=src)
                dist.send(g.detach().float().contiguous(), dst=src)
            elif cmd == CMD_SCORE_FWD_PAIR:
                with torch.no_grad():
                    r = self.scorer._scores(img_buf, ref_buf, prompts)
                dist.send(r.detach().float().contiguous(), dst=src)
            else:
                raise ValueError(f"[internvl_bridge] server got unknown cmd={cmd}")
        dist.barrier(group=_GLOO_GROUP)  # rendezvous with policy ranks' bridge_client_shutdown()
