"""
reward.py
=========
Observation construction and reward computation for STNGPeEnv, driven by
config.OBS_NORM and config.REWARD.

Observation, each scaled to ~[0, 1] via OBS_NORM, in OBS_METRICS order, with the
agent's previous (normalized) action appended (so the policy stays Markov):
    [synchrony R, beta_power (dB), entropy H,  <last action in [0,1]^k>]

Reward (per decision, over the trailing metric window): a monotonic objective
matching the project goal:
    reward = w_entropy * H - w_sync * R - lambda_charge * |charge|
where R is synchrony, H is spectral entropy, and |charge| is absolute injected
charge for the current pulse/window. beta_power is logged/observed but not used
unless config.REWARD["w_beta"] is set nonzero. An optional bad-state penalty is
applied when synchrony is high and entropy is low at the same time.
No time scaling.
"""

import numpy as np
from rl_stn_gpe import config


def _scale(value, lo, hi):
    return float(np.clip((value - lo) / (hi - lo + 1e-12), 0.0, 1.0))


def make_observation(R, beta, H, last_action=None):
    """Return the observation vector.

    [metrics in config.OBS_METRICS order] (+ last_action appended if given).
    `last_action` is the previous action normalized to [0, 1]^k (one entry per
    enabled action param); pass None to omit it.

    Scaling depends on config.USE_VECNORMALIZE:
      - False -> metrics are pre-scaled to ~[0, 1] via config.OBS_NORM (legacy).
      - True  -> RAW metric values are returned; SB3 VecNormalize whitens them
                 (running mean/std). The env declares an unbounded obs space in
                 this mode so raw values are always in-space.
    """
    vals = {"synchrony": R, "beta_power": beta, "entropy": H}
    if config.USE_VECNORMALIZE:
        obs = [float(vals[k]) for k in config.OBS_METRICS]
    else:
        obs = [_scale(vals[k], *config.OBS_NORM[k]) for k in config.OBS_METRICS]
    if last_action is not None:
        obs = obs + list(np.asarray(last_action, dtype=float).ravel())
    return np.asarray(obs, dtype=np.float32)


def _bump(value, target, scale, shape="gauss"):
    """Smooth exponential reward bump for one metric, peaking at `target`.

        dist = |value - target|
        shape "gauss"   ->  exp(-(dist / scale)**2)
        shape "laplace" ->  exp(-dist / scale)

    Returns a value in (0, 1] (== 1 exactly at target). `scale` sets the width
    (larger => broader credit around the target).
    """
    dist = abs(value - target)
    s = max(float(scale), 1e-9)
    if shape == "laplace":
        return float(np.exp(-dist / s))
    return float(np.exp(-(dist / s) ** 2))


# metric name -> the short breakdown key used in the env `info` dict and the
# training logger (kept stable for backward-compat).
_TERM_KEY = {"synchrony": "r_sync", "entropy": "r_entropy", "beta_power": "r_beta"}


def compute_reward(R, H, beta, charge=0.0, cfg=None):
    """Compute the scalar reward and a per-term breakdown (for logging).

    Parameters
    ----------
    R : float      - synchrony over the window
    H : float      - spectral entropy
    beta : float   - beta-band power in dB
    charge : float - |injected charge| this decision (energy cost; >= 0)

    Monotonic objective: lower synchrony is better, higher entropy is better,
    and lower absolute stimulation charge is better. If configured, an extra
    penalty is applied when R is above the synchrony threshold and H is below
    the entropy threshold. No time scaling.
    """
    cfg = cfg or config.REWARD

    # Previous target-bump reward, kept for reference.
    # shape = cfg.get("shape", "gauss")
    # vals = {"synchrony": R, "entropy": H, "beta_power": beta}
    # breakdown = {"r_sync": 0.0, "r_entropy": 0.0, "r_beta": 0.0}
    # total = 0.0
    # for name, spec in cfg["terms"].items():
    #     if spec.get("enabled", True) and spec.get("w", 0.0) != 0.0:
    #         r = spec["w"] * _bump(vals[name], spec["target"], spec["scale"], shape)
    #     else:
    #         r = 0.0
    #     breakdown[_TERM_KEY[name]] = float(r)
    #     total += r
    # r_charge = -cfg.get("w_charge", 1.0) * cfg.get("lambda_charge", 0.0) * float(charge)
    # breakdown["r_charge"] = float(r_charge)
    # total = float(total + r_charge)
    # return total, breakdown

    r_sync = -cfg.get("w_sync", 0.0) * float(R)
    r_entropy = cfg.get("w_entropy", 0.0) * float(H)
    r_beta = -cfg.get("w_beta", 0.0) * float(beta)
    r_charge = -cfg.get("lambda_charge", 0.0) * float(charge)
    bad = cfg.get("bad_state", {})
    if (
        bad.get("enabled", False)
        and float(R) > bad.get("sync_threshold", float("inf"))
        and float(H) < bad.get("entropy_threshold", float("-inf"))
    ):
        r_bad_state = -float(bad.get("penalty", 0.0))
    else:
        r_bad_state = 0.0

    breakdown = {
        "r_sync": float(r_sync),
        "r_entropy": float(r_entropy),
        "r_beta": float(r_beta),
        "r_charge": float(r_charge),
        "r_bad_state": float(r_bad_state),
    }
    total = float(r_sync + r_entropy + r_beta + r_charge + r_bad_state)
    return total, breakdown
