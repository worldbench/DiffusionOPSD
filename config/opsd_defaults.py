"""Canonical paper hyperparameters for SD3.5-M and Z-Image-Turbo."""

from __future__ import annotations

from copy import deepcopy


_COMMON_OPSD_DEFAULT = {
    "train_state": "rollout",
    "refine": 0,
    "aux_mode": 0,
    "opa": 1,
    "opa_margin": 0.005,
    "opa_n_ascent": 2,
    "opa_eta": 1.0,
    "opa_mb": 6,
    "opa_dual_neg": 1,
    "opa_cert": 0,
    "opa_dir_mode": "grad",
    # Mechanism ablations described in the paper appendix. Defaults below reproduce the
    # canonical OPSD method byte-for-byte; only the ablation launcher flips them. Each gates a
    # REAL alternate code path in the OPA loss block of train_opsd_ri_sd3.py (not a config relabel).
    "opa_state_mode": "rollout",   # rollout(method, on-policy z_q) | forward (offline forward-noised z_q)
    "opa_target_mode": "replace",  # replace(method, refined target IS the endpoint) | aux (NFT-endpoint + λ·reward-grad)
    "opa_aux_lambda": 1.0,         # auxiliary-loss weight; consumed only when opa_target_mode="aux"
    # Inactive compatibility fields read by the shared trainer; ``opa=1`` uses
    # the paper values above and does not execute the legacy RI branch.
    "rho": 0.03,
    "n_ascent": 2,
    "eta": 1.0,
    "margin": 0.005,
    "ri_mb": 6,
}


SD35M_OPSD_DEFAULT = {
    **_COMMON_OPSD_DEFAULT,
    "opa_rho": 0.10,
    "opa_query_sigma": 0.278,
}


ZIMAGE_TURBO_OPSD_DEFAULT = {
    **_COMMON_OPSD_DEFAULT,
    "opa_rho": 0.10,
    "opa_query_sigma": 0.273,
}


def opsd_default_params(backbone: str = "sd35m", **overrides):
    """Return a mutable copy of the canonical OPSD defaults."""

    key = backbone.lower().replace("-", "").replace("_", "")
    if key in {"sd35m", "sd3", "sd35medium", "sd3.5m", "sd3.5medium"}:
        params = deepcopy(SD35M_OPSD_DEFAULT)
    elif key in {"zimage", "zimg", "zimageturbo"}:
        params = deepcopy(ZIMAGE_TURBO_OPSD_DEFAULT)
    else:
        raise ValueError(f"Unknown OPSD default backbone: {backbone}")
    params.update(overrides)
    return params


def sd35m_opsd_default_params(**overrides):
    return opsd_default_params("sd35m", **overrides)


def zimage_turbo_opsd_default_params(**overrides):
    return opsd_default_params("zimage_turbo", **overrides)
