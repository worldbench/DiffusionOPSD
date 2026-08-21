#!/usr/bin/env python3
"""Cheap release/config validation; does not load model weights or require a GPU."""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
CORE_ROOT = SRC_ROOT
sys.path.insert(0, str(ROOT))

PUBLIC_REWARDS = ("hpsv2", "clipscore", "pickscore", "aesthetic", "imagereward", "hpsv3", "deqa")
ALL_CONFIGS = (
    *(f"sd35_{reward}" for reward in PUBLIC_REWARDS),
    "sd35_open3",
    *(f"zimage_{reward}" for reward in PUBLIC_REWARDS),
)


@contextmanager
def public_topology(name: str):
    """Resolve configs exactly as their launcher would, without leaking env state."""

    keys = (
        "PUBLIC_N_GPUS", "PUBLIC_POLICY_WORLD_SIZE", "PUBLIC_LAUNCH_WORLD_SIZE",
        "ZIMAGE_HEAVY_BRIDGE", "ZIMAGE_HEAVY_DIFF_BRIDGE", "PUBLIC_NUM_UPDATES",
        "PUBLIC_SAVE_FREQ",
    )
    old = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.pop("PUBLIC_NUM_UPDATES", None)
        os.environ.pop("PUBLIC_SAVE_FREQ", None)
        if name.startswith("zimage_") and name.rsplit("_", 1)[-1] in {"hpsv3", "deqa"}:
            os.environ.update({
                "ZIMAGE_HEAVY_BRIDGE": "0",
                "ZIMAGE_HEAVY_DIFF_BRIDGE": "1",
                "PUBLIC_POLICY_WORLD_SIZE": "6",
                "PUBLIC_LAUNCH_WORLD_SIZE": "7",
            })
        else:
            os.environ["ZIMAGE_HEAVY_BRIDGE"] = "0"
            os.environ["ZIMAGE_HEAVY_DIFF_BRIDGE"] = "0"
            os.environ["PUBLIC_POLICY_WORLD_SIZE"] = "8"
            os.environ["PUBLIC_LAUNCH_WORLD_SIZE"] = "8"
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_public_config(name: str):
    path = ROOT / "config" / "public.py"
    spec = importlib.util.spec_from_file_location("_diffusionopsd_public_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    with public_topology(name):
        return module.get_config(name)


def assert_paper_config(name: str, cfg) -> None:
    """Assert the protocol values stated in the paper appendix."""

    assert Path(cfg.dataset) == ROOT / "data" / "pickapic"
    assert cfg.train.beta == 0.0
    assert cfg.opsd.opa == 1
    assert cfg.opsd.train_state == "rollout"
    assert cfg.opsd.opa_rho == 0.10
    assert cfg.opsd.opa_n_ascent == 2
    assert cfg.opsd.opa_eta == 1.0
    assert cfg.opsd.opa_dual_neg == 1
    assert cfg.opsd.opa_cert == 0
    assert cfg.opsd.opa_dir_mode == "grad"
    assert cfg.train.learning_rate == 3e-4
    assert cfg.train.adam_beta1 == 0.9
    assert cfg.train.adam_beta2 == 0.999
    assert cfg.train.adam_weight_decay == 1e-4
    assert cfg.train.adam_epsilon == 1e-8
    assert cfg.train.adv_clip_max == 5
    assert cfg.train.num_inner_epochs == 1
    assert cfg.train.ema is True
    assert cfg.sample.global_std is True
    assert cfg.decay_type == 1
    assert cfg.save_freq == 10
    assert cfg.eval_freq == 0

    if name.startswith("sd35_"):
        reward = name.removeprefix("sd35_")
        assert cfg.pretrained.model == "stabilityai/stable-diffusion-3.5-medium"
        assert cfg.resolution == 512
        assert cfg.mixed_precision == "fp16"
        assert cfg.sample.num_steps == 10
        assert cfg.sample.eval_num_steps == 40
        assert cfg.sample.guidance_scale == 1.0
        assert cfg.sample.num_image_per_prompt == 24
        assert cfg.sample.train_batch_size == 9
        assert cfg.sample.num_batches_per_epoch == 16
        assert cfg.train.gradient_accumulation_steps == 16
        assert cfg.sample.deterministic is True
        assert cfg.sample.solver == "dpm2"
        assert cfg.opsd.opa_query_sigma == 0.278
        assert cfg.beta == (0.1 if reward == "open3" else 1.0)
        assert cfg.num_epochs == (300 if reward == "open3" else 100)
        assert cfg.opsd.opa_mb == (1 if reward in {"imagereward", "deqa"} else 6)
        if reward == "open3":
            assert dict(cfg.reward_fn) == {
                "pickscore": 1.0,
                "clipscore": 1.0,
                "hpsv2": 1.0,
            }
            assert cfg.opsd.opa_reward_kind == "open3"
        else:
            assert dict(cfg.reward_fn) == {reward: 1.0}
    else:
        reward = name.removeprefix("zimage_")
        assert cfg.pretrained.model == "Tongyi-MAI/Z-Image-Turbo"
        assert cfg.resolution == 1024
        assert cfg.mixed_precision == "bf16"
        assert cfg.sample.num_steps == 9
        assert cfg.sample.eval_num_steps == 9
        assert cfg.sample.guidance_scale == 0.0
        assert cfg.sample.num_image_per_prompt == 12
        assert cfg.sample.train_batch_size == 6
        expected_batches = 16 if reward in {"hpsv3", "deqa"} else 12
        assert cfg.sample.num_batches_per_epoch == expected_batches
        assert cfg.train.gradient_accumulation_steps == expected_batches
        assert cfg.sample.deterministic is True
        assert cfg.sample.solver == "flow_euler"
        assert cfg.opsd.opa_query_sigma == 0.273
        assert cfg.opsd.opa_mb == 2
        assert cfg.beta == 1.0
        assert cfg.num_epochs == 100
        assert dict(cfg.reward_fn) == {reward: 1.0}


def assert_shared_metrics_tee() -> None:
    from diffusionopsd.metrics import install_wandb_jsonl_tee

    class FakeWandb:
        def __init__(self):
            self.calls = []

        def log(self, data, step=None, **kwargs):
            self.calls.append((data, step, kwargs))
            return "delegated"

    with tempfile.TemporaryDirectory() as tmpdir:
        fake = FakeWandb()
        path = Path(tmpdir, "metrics.jsonl")
        install_wandb_jsonl_tee(fake, path)
        image = object()
        result = fake.log({"loss": 1.25, "epoch": 2, "image": image}, step=7, commit=False)
        assert result == "delegated"
        assert fake.calls == [({"loss": 1.25, "epoch": 2, "image": image}, 7, {"commit": False})]
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record == {"loss": 1.25, "epoch": 2, "_step": 7}

    trainer_paths = sorted((ROOT / "scripts").glob("train_*.py"))
    consumers = 0
    for path in trainer_paths:
        source = path.read_text(encoding="utf-8")
        assert "def _tee_log" not in source
        if "metrics.jsonl" in source:
            assert source.count("install_wandb_jsonl_tee(") == 1
            consumers += 1
    assert consumers == 8

    # OPD uses one shared benchmark/instrumentation implementation too.
    opd_common = (ROOT / "opd" / "opd_common.py").read_text(encoding="utf-8")
    assert opd_common.count("class BenchmarkTimer") == 1
    assert "load_monitor_reward" not in opd_common
    assert "diffusionopsd.rewards" not in opd_common
    for path in (ROOT / "opd").glob("train_*.py"):
        source = path.read_text(encoding="utf-8")
        assert "class BenchmarkTimer" not in source
        assert "BenchmarkTimer" in source
        assert "load_monitor_reward" not in source


def assert_prompt_recipe() -> None:
    recipe = json.loads((ROOT / "data" / "pickapic_recipe.json").read_text(encoding="utf-8"))
    assert recipe["dataset_id"] == "sayakpaul/pick-a-pic-v2-unique-prompts"
    assert recipe["dataset_revision"] == "ec6ade5fc615ec90152a78140492cec60da9680c"
    assert recipe["dataset_column"] == "prompt"
    assert recipe["dataset_rows"] == 58960
    assert recipe["paper_train_lines"] == 25415
    assert recipe["paper_train_sha256"] == "39d94f994a249777207878d793f65d1dbaa7227e7e2409039cca8518aaa7100c"
    assert len(recipe["selectors"]) == recipe["paper_train_lines"]
    assert sum(isinstance(x, int) for x in recipe["selectors"]) == recipe["hf_selector_count"] == 24902
    assert sum(isinstance(x, str) for x in recipe["selectors"]) == recipe["legacy_selector_count"] == 513
    assert len((ROOT / "data" / "drawbench" / "test.txt").read_text(encoding="utf-8").splitlines()) == 1000


def load_refl_config(name: str):
    path = ROOT / "config" / "refl.py"
    spec = importlib.util.spec_from_file_location("_diffusionopsd_refl_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    topology_name = name.replace("sd35m_", "sd35_")
    with public_topology(topology_name):
        return module.get_config(name)


def assert_refl_configs() -> None:
    for backbone in ("sd35m", "zimage"):
        for reward in PUBLIC_REWARDS:
            cfg = load_refl_config(f"{backbone}_{reward}")
            rr = cfg.refl
            assert Path(cfg.dataset) == ROOT / "data" / "pickapic"
            assert rr.num_updates == 100
            assert rr.distinct_prompt_groups == 48
            assert rr.trajectories_per_prompt == (24 if backbone == "sd35m" else 12)
            assert rr.trajectories_per_update == 48 * rr.trajectories_per_prompt
            assert rr.micro_batch_size == 1
            assert rr.late_fraction == 0.25
            assert rr.hinge_margin == 2.0
            assert rr.checkpoint_every == 10
            assert rr.standardize_reward is (reward != "imagereward")
            assert cfg.train.beta == 0.0
            assert cfg.sample.deterministic is True
            if backbone == "zimage" and reward in {"hpsv3", "deqa"}:
                assert rr.expected_policy_world_size == 6
                assert rr.expected_launch_world_size == 7
            else:
                assert rr.expected_policy_world_size == 8
                assert rr.expected_launch_world_size == 8


@contextmanager
def baseline_topology(backbone: str, reward: str):
    """Resolve baseline configs with the launcher topology and restore the environment."""

    keys = (
        "PUBLIC_N_GPUS",
        "PUBLIC_POLICY_WORLD_SIZE",
        "PUBLIC_LAUNCH_WORLD_SIZE",
        "ZIMAGE_HEAVY_BRIDGE",
        "ZIMAGE_HEAVY_DIFF_BRIDGE",
    )
    old = {key: os.environ.get(key) for key in keys}
    heavy = backbone == "zimage" and reward in {"hpsv3", "deqa"}
    try:
        os.environ.update(
            {
                "PUBLIC_N_GPUS": "7" if heavy else "8",
                "PUBLIC_POLICY_WORLD_SIZE": "6" if heavy else "8",
                "PUBLIC_LAUNCH_WORLD_SIZE": "7" if heavy else "8",
                "ZIMAGE_HEAVY_BRIDGE": "1" if heavy else "0",
                "ZIMAGE_HEAVY_DIFF_BRIDGE": "0",
            }
        )
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_named_config(filename: str, name: str):
    path = ROOT / "config" / filename
    spec = importlib.util.spec_from_file_location(f"_diffusionopsd_{path.stem}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_config(name)


def assert_baseline_configs() -> None:
    """Check every public DiffusionNFT/FlowGRPO preset against the appendix protocol."""

    for reward in PUBLIC_REWARDS:
        with baseline_topology("sd35", reward):
            cfg = load_named_config("nft.py", f"sd3_{reward}")
        assert Path(cfg.dataset) == ROOT / "data" / "pickapic"
        assert cfg.pretrained.model == "stabilityai/stable-diffusion-3.5-medium"
        assert dict(cfg.reward_fn) == {reward: 1.0}
        assert cfg.resolution == 512 and cfg.mixed_precision == "fp16"
        assert cfg.sample.num_steps == 10 and cfg.sample.eval_num_steps == 40
        assert cfg.sample.guidance_scale == 1.0
        assert cfg.sample.solver == "dpm2" and cfg.sample.deterministic is True
        assert cfg.sample.num_image_per_prompt == 24
        assert cfg.sample.train_batch_size == 9 and cfg.sample.num_batches_per_epoch == 16
        assert cfg.train.beta == 1e-4 and cfg.train.gradient_accumulation_steps == 16
        assert cfg.num_epochs == 100
        assert cfg.save_freq == 10

    with baseline_topology("sd35", "clipscore"):
        cfg = load_named_config("flowgrpo.py", "sd35_clipscore")
    assert Path(cfg.dataset) == ROOT / "data" / "pickapic"
    assert cfg.pretrained.model == "stabilityai/stable-diffusion-3.5-medium"
    assert dict(cfg.reward_fn) == {"clipscore": 1.0}
    assert cfg.resolution == 512 and cfg.mixed_precision == "fp16"
    assert cfg.sample.num_steps == 10 and cfg.sample.eval_num_steps == 40
    assert cfg.sample.guidance_scale == 1.0
    assert cfg.sample.solver == "flow" and cfg.sample.deterministic is False
    assert cfg.sample.noise_level == 0.7 and cfg.sample.num_image_per_prompt == 24
    assert cfg.sample.train_batch_size == 9 and cfg.sample.num_batches_per_epoch == 16
    assert cfg.train.beta == 0.0 and cfg.train.timestep_fraction == 1.0
    assert cfg.train.gradient_accumulation_steps == 8
    assert cfg.flowgrpo.group_size == 24 and cfg.flowgrpo.clip_range == 1e-4
    assert cfg.flowgrpo.optimizer_updates_per_rollout == 2 and cfg.num_epochs == 50
    assert cfg.save_freq == 10

    for method in ("nft", "flowgrpo"):
        for reward in PUBLIC_REWARDS:
            with baseline_topology("zimage", reward):
                cfg = load_named_config("zimage.py", f"zimg_{method}_{reward}")
            heavy = reward in {"hpsv3", "deqa"}
            batches = 16 if heavy else 12
            assert Path(cfg.dataset) == ROOT / "data" / "pickapic"
            assert cfg.pretrained.model == "Tongyi-MAI/Z-Image-Turbo"
            assert dict(cfg.reward_fn) == {reward: 1.0}
            assert cfg.resolution == 1024 and cfg.mixed_precision == "bf16"
            assert cfg.sample.num_steps == 9 and cfg.sample.eval_num_steps == 9
            assert cfg.sample.guidance_scale == 0.0
            assert cfg.sample.num_image_per_prompt == 12
            assert cfg.sample.train_batch_size == 6
            assert cfg.sample.num_batches_per_epoch == batches
            if method == "nft":
                assert cfg.sample.solver == "flow_euler" and cfg.sample.deterministic is True
                assert cfg.train.beta == 1e-4
                assert cfg.train.gradient_accumulation_steps == batches
                assert cfg.num_epochs == 100
                assert cfg.save_freq == 10
            else:
                assert cfg.sample.solver == "flow_sde" and cfg.sample.deterministic is False
                assert cfg.sample.noise_level == 0.7
                assert cfg.train.beta == 0.0 and cfg.train.timestep_fraction == 1.0
                assert cfg.train.gradient_accumulation_steps == batches // 2
                assert cfg.flowgrpo.group_size == 12 and cfg.flowgrpo.clip_range == 1e-4
                assert cfg.flowgrpo.optimizer_updates_per_rollout == 2
                assert cfg.num_epochs == 50
                assert cfg.save_freq == 10

    launcher = (ROOT / "scripts" / "train_baseline.sh").read_text(encoding="utf-8")
    for fragment in (
        "config/flowgrpo.py:sd35_clipscore",
        "config/zimage.py:zimg_nft_${REWARD}",
        "config/zimage.py:zimg_flowgrpo_${REWARD}",
        "export ZIMAGE_HEAVY_BRIDGE=1",
        "NUM_EPOCHS=$((UPDATES / 2))",
        "--config.debug=true",
    ):
        assert fragment in launcher, f"baseline launcher lost protocol fragment: {fragment}"


def assert_mixed_reward_configs() -> None:
    """Check the paper Open3 objective and a non-paper arbitrary weighted sum."""

    keys = (
        "PUBLIC_MIXED_REWARDS",
        "PUBLIC_POLICY_WORLD_SIZE",
        "PUBLIC_N_GPUS",
        "PUBLIC_NUM_UPDATES",
        "PUBLIC_SAVE_FREQ",
    )
    old = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["PUBLIC_POLICY_WORLD_SIZE"] = "8"
        os.environ.pop("PUBLIC_N_GPUS", None)
        os.environ.pop("PUBLIC_NUM_UPDATES", None)
        os.environ.pop("PUBLIC_SAVE_FREQ", None)

        os.environ["PUBLIC_MIXED_REWARDS"] = "pickscore=1,clipscore=1,hpsv2=1"
        for method in ("opsd", "nft"):
            cfg = load_named_config("mixed.py", f"sd35_{method}")
            assert dict(cfg.reward_fn) == {
                "pickscore": 1.0,
                "clipscore": 1.0,
                "hpsv2": 1.0,
            }
            assert cfg.pretrained.model == "stabilityai/stable-diffusion-3.5-medium"
            assert cfg.beta == 0.1 and cfg.num_epochs == 300 and cfg.save_freq == 10
            assert cfg.sample.num_steps == 10 and cfg.sample.eval_num_steps == 40
            if method == "opsd":
                assert cfg.train.beta == 0.0
                assert cfg.opsd.opa_reward_kind == "mixed"
            else:
                assert cfg.train.beta == 1e-4

        custom = {
            "clipscore": 0.5,
            "hpsv2": 2.0,
            "aesthetic": 0.25,
            "imagereward": 0.125,
        }
        os.environ["PUBLIC_MIXED_REWARDS"] = ",".join(
            f"{name}={weight}" for name, weight in custom.items()
        )
        cfg = load_named_config("mixed.py", "sd35_opsd")
        assert dict(cfg.reward_fn) == custom
        assert cfg.opsd.opa_reward_kind == "mixed" and cfg.opsd.opa_mb == 1

        module_path = ROOT / "config" / "mixed.py"
        spec = importlib.util.spec_from_file_location("_diffusionopsd_mixed_parser", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        for invalid in ("clipscore", "clipscore=1,clipscore=2", "clipscore=0,hpsv2=1", "unknown=1,hpsv2=1"):
            try:
                module.parse_reward_spec(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid mixed reward spec was accepted: {invalid}")
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    launcher = (ROOT / "scripts" / "train_mixed_reward.sh").read_text(encoding="utf-8")
    for fragment in (
        "pickscore=1,clipscore=1,hpsv2=1",
        "config/mixed.py:sd35_${METHOD}",
        "scripts/check_reward_setup.py --reward \"$reward\" --backbone sd35",
        "--config.debug=true",
    ):
        assert fragment in launcher, f"mixed-reward launcher lost required fragment: {fragment}"


def assert_opd_configs() -> None:
    readme = ROOT / "opd" / "README.md"
    assert readme.is_file()
    assert not (ROOT / "opd" / "OPD.md").exists()
    readme_text = readme.read_text(encoding="utf-8")
    for heading in (
        "## Reproduction contract",
        "#### DanceOPD",
        "#### DiffusionOPD",
        "#### FlowOPD",
    ):
        assert heading in readme_text, f"OPD README lost merged section: {heading}"
    for source_path in (ROOT / "opd").rglob("*"):
        if source_path.is_file() and source_path.suffix in {".md", ".py", ".sh"}:
            assert "OPD.md" not in source_path.read_text(encoding="utf-8"), (
                f"stale OPD.md reference in {source_path.relative_to(ROOT)}"
            )

    path = ROOT / "opd" / "configs" / "opd_config.py"
    spec = importlib.util.spec_from_file_location("_diffusionopsd_opd_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    expected = {
        "danceopd": (2688, "velocity_mse_query", 1),
        "diffusionopd": (528, "transition_mean_mse_all", 10),
        "flowopd": (544, "ppo_stepwise_teacher_kl", None),
    }
    # Local launch environments commonly override these two paths. The release
    # checker validates canonical paper defaults, so isolate it from those overrides.
    override_keys = ("MODEL_PATH", "OPD_DATASET", "OPD_SAVE_FREQ")
    old_overrides = {key: os.environ.pop(key, None) for key in override_keys}
    try:
        for method, (samples, loss, query_k) in expected.items():
            cfg = module.get_config(method)
            assert cfg.pretrained.model == "stabilityai/stable-diffusion-3.5-medium"
            assert Path(cfg.dataset) == ROOT / "data" / "pickapic"
            assert cfg.resolution == 512
            assert cfg.num_epochs == 300
            assert cfg.save_freq == 10
            assert cfg.sample.num_steps == 10
            assert cfg.sample.eval_num_steps == 40
            assert cfg.sample.guidance_scale == 1.0
            assert cfg.opd.teacher_rewards == ["pickscore", "clipscore", "hpsv2"]
            assert cfg.opd.teacher_weights == [1.0, 1.0, 1.0]
            assert cfg.opd.ensemble == "same_sample"
            assert "reward_fn" not in cfg
            assert cfg.opd.samples_per_epoch_x8 == samples
            assert cfg.opd.loss == loss
            if query_k is not None:
                assert cfg.opd.query_k == query_k
        flow = module.get_config("flowopd")
        assert flow.sample.deterministic is False
        assert flow.sample.noise_level > 0
        assert flow.opd.clip_range == 0.2
    finally:
        for key, value in old_overrides.items():
            if value is not None:
                os.environ[key] = value


def assert_public_reward_sources() -> None:
    registry = (CORE_ROOT / "rewards.py").read_text(encoding="utf-8")
    for reward in PUBLIC_REWARDS:
        function = "clip_score" if reward == "clipscore" else f"{reward}_score"
        assert f"def {function}(" in registry, f"missing reward registry entry: {reward}"
    tree = ast.parse(registry)
    assignments = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in {"PUBLIC_REWARDS", "INTERNAL_REWARDS"}
    }
    assert assignments["PUBLIC_REWARDS"] == set(PUBLIC_REWARDS)
    assert assignments["INTERNAL_REWARDS"] == {"altclip", "internvl_t2i", "internvl_dual"}
    for legacy in ("blipitm", "internvl2b", "unifiedreward", "qalign", "geneval"):
        assert legacy not in registry, f"non-paper reward leaked into registry: {legacy}"
    public_cfg = (ROOT / "config" / "public.py").read_text(encoding="utf-8")
    assert '"hpsv3", "deqa"' in public_cfg
    z_trainer = (ROOT / "scripts" / "train_opsd_zimage.py").read_text(encoding="utf-8")
    assert '(("hpsv3", "deqa") if _diff_bridge_active else ())' in z_trainer
    bridge = (CORE_ROOT / "zimage_heavy_diff_bridge.py").read_text(encoding="utf-8")
    assert "torch.autograd.grad(r.sum(), image)" in bridge
    hpsv3 = (CORE_ROOT / "hpsv3_scorer.py").read_text(encoding="utf-8")
    assert 'os.environ.get("HPSV3_CHECKPOINT") or None' in hpsv3
    deqa = (CORE_ROOT / "deqa_scorer.py").read_text(encoding="utf-8")
    assert 'attn_implementation="eager"' in deqa

    smoke = (ROOT / "scripts" / "smoke_reward_gradient.py").read_text(encoding="utf-8")
    smoke_tree = ast.parse(smoke)
    smoke_rewards = next(
        ast.literal_eval(node.value)
        for node in smoke_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "PUBLIC_REWARDS"
            for target in node.targets
        )
    )
    assert tuple(smoke_rewards) == PUBLIC_REWARDS
    assert "_load_reward_scorer" in smoke and "_reward_scores_grad" in smoke


def assert_baseline_source_invariants() -> None:
    for filename in ("train_nft_sd3.py", "train_nft_zimage.py"):
        source = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
        for fragment in (
            "positive_loss = ((x0_prediction - x0) ** 2 / weight_factor)",
            "negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight_factor)",
            "loss += config.train.beta * torch.mean(kl_div_loss)",
        ):
            assert fragment in source, f"{filename} lost DiffusionNFT invariant: {fragment}"

    for filename in ("train_flowgrpo_sd3.py", "train_flowgrpo_zimage.py"):
        source = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
        for fragment in (
            "ratio = torch.exp(logp_new - logp_old)",
            "torch.clamp(ratio, 1.0 - fg_clip_range, 1.0 + fg_clip_range)",
            "policy_loss = torch.max(unclipped, clipped).mean()",
        ):
            assert fragment in source, f"{filename} lost FlowGRPO invariant: {fragment}"

    for filename in (
        "train_opsd_ri_sd3.py",
        "train_nft_sd3.py",
        "train_flowgrpo_sd3.py",
        "train_opsd_zimage.py",
        "train_nft_zimage.py",
        "train_flowgrpo_zimage.py",
    ):
        source = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
        assert "and global_step > 0" in source
        assert "and global_step % config.save_freq == 0" in source, (
            f"{filename} checkpoint cadence is not expressed in optimizer updates"
        )

    common = (ROOT / "scripts" / "refl_common.py").read_text(encoding="utf-8")
    for fragment in (
        "late_count = max(1, math.ceil(num_steps * late_fraction))",
        "late_start = num_steps - late_count",
        "trajectories_per_update must divide exactly",
    ):
        assert fragment in common
    for filename in ("train_refl_sd35m.py", "train_refl_zimage.py"):
        source = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
        for fragment in (
            "choose_late_index(",
            "stop_before_index=late_index",
            "torch.relu(float(rr.hinge_margin) - normalized)",
            "loss = float(rr.grad_scale) * hinge.mean()",
            "ema.step(trainable, global_step)",
        ):
            assert fragment in source, f"{filename} lost ReFL invariant: {fragment}"

    dance = (ROOT / "opd" / "train_danceopd_open3.py").read_text(encoding="utf-8")
    assert "lo, hi = num_steps // 2, num_steps - 1" in dance
    assert "teacher_velocities(" in dance and "mse_loss(v_student.float()" in dance
    diffusion = (ROOT / "opd" / "train_diffusionopd_open3.py").read_text(encoding="utf-8")
    assert "for step_idx in range(num_steps):" in diffusion
    assert "flow_transition_mean(" in diffusion
    flow = (ROOT / "opd" / "train_flowopd_open3.py").read_text(encoding="utf-8")
    for fragment in (
        "teacher_stepwise_logprob_reward(",
        "global_stepwise_advantage(reward, world_size)",
        "torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)",
    ):
        assert fragment in flow
    for source in (dance, diffusion, flow):
        assert "global_step % int(config.save_freq) == 0" in source
        assert "should_save = (" in source and "bench_on" in source


def assert_opsd_source_invariants() -> None:
    """Guard the paper-critical equations against accidental source drift."""

    common_fragments = (
        "budget = (rho * x0.flatten(1).norm(dim=1)).view(-1, 1, 1, 1)",
        "step_len = eta * budget / max(n_ascent, 1)",
        "v_pos = config.beta * v_theta_q.float() + (1 - config.beta) * v_old_q",
        "v_neg = (1.0 + config.beta) * v_old_q - config.beta * v_theta_q.float()",
        "r1 = torch.clamp((adv_clip / config.train.adv_clip_max) / 2.0 + 0.5, 0, 1)",
        ".mean(dim=rd, keepdim=True).clip(min=1e-5)",
        "loss = opa_policy",
        "decay = return_decay(global_step, config.decay_type)",
        "r=32, lora_alpha=64",
    )
    for filename in ("train_opsd_ri_sd3.py", "train_opsd_zimage.py"):
        source = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
        for fragment in common_fragments:
            assert fragment in source, f"{filename} lost paper invariant: {fragment}"

    pickscore_source = (CORE_ROOT / "pickscore_scorer.py").read_text(encoding="utf-8")
    assert "scores = scores / 26" in pickscore_source

    sd3_source = (ROOT / "scripts" / "train_opsd_ri_sd3.py").read_text(encoding="utf-8")
    for fragment in (
        'if kind in ("open3", "multi_open3", "mixed"):',
        'weighted = float(weight) * sub_scores',
        'reward_weights=config.reward_fn if opa_kind == "mixed" else None',
    ):
        assert fragment in sd3_source, f"SD3 trainer lost mixed-reward gradient invariant: {fragment}"

    ema_source = (CORE_ROOT / "ema.py").read_text(encoding="utf-8")
    assert "(1 + optimization_step) / (10 + optimization_step)" in ema_source


def assert_readme_results() -> None:
    """Keep the published tables synchronized with the paper source."""

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    expected_rows = (
        "| **DiffusionOPSD** | **24.94** | **0.340** | **0.390** | 12.08 | **1.76** | **13.34** | **4.94** | **0.450** | **0.214** | **0.465** |",
        "| **DiffusionOPSD** | **25.15** | **0.320** | **0.390** | **10.74** | **1.79** | **14.44** | **4.78** | **0.451** | **0.243** | **0.551** |",
        "| SD3.5-M | DiffusionNFT | 212.4 | 47.8 GB | 47.2 | 1.00× |",
        "| SD3.5-M | **DiffusionOPSD** | **126.9** | 50.0 GB | **28.2** | **0.60×** |",
        "| Z-Image-Turbo | DiffusionNFT | 1826.2 | 49.9 GB | 405.8 | 1.00× |",
        "| Z-Image-Turbo | **DiffusionOPSD** | **674.0** | 61.5 GB | **149.8** | **0.37×** |",
    )
    for row in expected_rows:
        assert row in readme, f"README result row drifted from paper: {row}"


def assert_release_hygiene() -> None:
    """Reject generated artifacts, private paths, credentials, and local storage names."""

    assert CORE_ROOT.is_dir(), "missing flat src package"
    assert not (CORE_ROOT / "diffusionopsd").exists(), "nested src/diffusionopsd package remains"
    assert not (ROOT / "flow_grpo").exists(), "legacy flow_grpo package remains in release"

    def release_paths():
        # A standalone clone contains Git's binary object database; it is not
        # release content and may contain arbitrary compressed byte sequences.
        for path in ROOT.rglob("*"):
            if ".git" not in path.relative_to(ROOT).parts:
                yield path

    forbidden_dirs = {"__pycache__", ".ruff_cache", ".pytest_cache", "build", "dist"}
    for path in release_paths():
        if path.is_dir():
            assert path.name not in forbidden_dirs, f"generated directory in release: {path.relative_to(ROOT)}"
            # ``pip install -e .`` (the documented setup) materializes a
            # gitignored ``*.egg-info`` directory in the checkout.  Accept it
            # here so the post-install configuration check remains runnable;
            # source-control/packaging checks still keep it out of the release.
            if path.name.endswith(".egg-info"):
                continue
        elif path.suffix in {".pyc", ".pyo"}:
            raise AssertionError(f"generated bytecode in release: {path.relative_to(ROOT)}")

    private_prefixes = tuple(
        b"/" + suffix
        for suffix in (
            b"Users/", b"mnt/", b"home/", b"opt/", b"tmp/", b"private/", b"workspace/", b"root/"
        )
    )
    private_terms = (
        b"hdfs" + b"://",
        b"search-" + b"auto-eval",
        b"seed-" + b"aigc",
        b"zhou" + b"wei",
        b"sys-" + b"proxy",
        b".byte" + b"d.org",
        b"BEGIN " + b"PRIVATE KEY",
    )
    hf_token = re.compile(rb"hf_[A-Za-z0-9]{20,}")
    for path in release_paths():
        if not path.is_file():
            continue
        data = path.read_bytes()
        for marker in private_prefixes + private_terms:
            assert marker not in data, f"private path or identifier in {path.relative_to(ROOT)}"
        assert hf_token.search(data) is None, f"Hugging Face token in {path.relative_to(ROOT)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configs",
        nargs="*",
        default=list(ALL_CONFIGS),
    )
    args = parser.parse_args()
    for name in args.configs:
        cfg = load_public_config(name)
        assert_paper_config(name, cfg)
        print(
            f"[ok] {name}: model={cfg.pretrained.model} rewards={dict(cfg.reward_fn)} "
            f"steps={cfg.sample.num_steps} resolution={cfg.resolution}"
        )
    assert_shared_metrics_tee()
    print("[ok] all eight WandB trainers share diffusionopsd.metrics.install_wandb_jsonl_tee")
    assert_prompt_recipe()
    print("[ok] pinned Pick-a-Pic recipe and 1,000-prompt DrawBench manifest")
    assert_refl_configs()
    print("[ok] 14 public ReFL configurations match the paper protocol")
    assert_baseline_configs()
    print(
        "[ok] 7 SD3.5-M DiffusionNFT, the matched SD3.5-M FlowGRPO control, "
        "and 14 Z-Image baseline presets match the paper protocol"
    )
    assert_mixed_reward_configs()
    print("[ok] arbitrary weighted SD3.5-M DiffusionOPSD/DiffusionNFT mixed-reward presets")
    assert_opd_configs()
    print("[ok] DanceOPD, DiffusionOPD, and FlowOPD configs match the paper protocol")
    assert_public_reward_sources()
    print("[ok] all seven public reward adapters and the Z-Image gradient bridge are wired")
    assert_baseline_source_invariants()
    print("[ok] DiffusionNFT, FlowGRPO, ReFL, and OPD source objectives retain their paper-critical operations")
    assert_opsd_source_invariants()
    print("[ok] paper-critical trainer equations and EMA schedules are present")
    assert_readme_results()
    print("[ok] README result and efficiency tables match the paper")
    assert_release_hygiene()
    print("[ok] no generated artifacts, private paths, private identifiers, or credentials")


if __name__ == "__main__":
    main()
