"""
reward.py
=========
Observation construction and reward computation for STNGPeEnv, driven by
config.OBS_NORM and config.REWARD.

Observation, each scaled to ~[0, 1] via OBS_NORM, in OBS_METRICS order, with the
agent's previous (normalized) action appended (so the policy stays Markov):
    [synchrony R, beta_power (dB), entropy H,  <last action in [0,1]^k>]

Reward (per decision, over the trailing metric window): a simple banded distance
per metric. For dist = |value - target|:
    dist < near_tol  ->  +r_near              (bullseye)
    dist < far_tol   ->  +r_far               (close-ish)
    else             ->  -neg_scale * dist    (miss: penalty grows with distance)
Each term has an on/off weight; a per-pulse charge penalty is subtracted:
    reward = w_sync*t_sync + w_entropy*t_entropy + w_beta*t_beta
             - w_charge * lambda_charge * |charge|
No time scaling.
"""

import numpy as np
from rl_stn_gpe import config


def _scale(value, lo, hi):
    return float(np.clip((value - lo) / (hi - lo + 1e-12), 0.0, 1.0))


def make_observation(R, beta, H, last_action=None):
    """Return the normalized observation vector.

    [scaled metrics in config.OBS_METRICS order] (+ last_action appended if given).
    `last_action` is the previous action already normalized to [0, 1]^k (one entry
    per enabled action param); pass None to omit it.
    """
    vals = {"synchrony": R, "beta_power": beta, "entropy": H}
    obs = [_scale(vals[k], *config.OBS_NORM[k]) for k in config.OBS_METRICS]
    if last_action is not None:
        obs = obs + list(np.asarray(last_action, dtype=float).ravel())
    return np.asarray(obs, dtype=np.float32)


def _banded(value, target, near_tol, far_tol, r_near, r_far, neg_scale):
    """Banded distance reward for one metric (symmetric distance to target).

        dist = |value - target|
        dist < near_tol  ->  +r_near              (bullseye)
        dist < far_tol   ->  +r_far               (close-ish)
        else             ->  -neg_scale * dist    (miss: scaled penalty)
    """
    dist = abs(value - target)
    if dist < near_tol:
        return float(r_near)
    if dist < far_tol:
        return float(r_far)
    return float(-neg_scale * dist)


def compute_reward(R, H, beta, charge=0.0, cfg=None):
    """Compute the scalar reward and a per-term breakdown (for logging).

    Parameters
    ----------
    R : float      - synchrony over the window           (target = target_sync)
    H : float      - spectral entropy                    (target = target_entropy)
    beta : float   - beta-band power in dB               (target = target_beta)
    charge : float - |injected charge| this decision (energy cost; >= 0)

    Each metric term is a banded distance to its target (see _banded), gated by an
    on/off weight. A per-pulse charge penalty is subtracted. No time scaling.
    """
    cfg = cfg or config.REWARD
    near, far = cfg["near_tol"], cfg["far_tol"]
    r_near, r_far, neg = cfg["r_near"], cfg["r_far"], cfg["neg_scale"]

    r_sync = cfg["w_sync"] * _banded(R, cfg["target_sync"], near, far, r_near, r_far, neg)
    r_entropy = cfg["w_entropy"] * _banded(H, cfg["target_entropy"], near, far, r_near, r_far, neg)
    r_beta = cfg["w_beta"] * _banded(
        beta, cfg["target_beta"], cfg["near_tol_beta"], cfg["far_tol_beta"],
        r_near, r_far, neg)

    r_charge = -cfg.get("w_charge", 1.0) * cfg.get("lambda_charge", 0.0) * float(charge)
    total = float(r_sync + r_entropy + r_beta + r_charge)
    return total, {"r_sync": float(r_sync), "r_entropy": float(r_entropy),
                   "r_beta": float(r_beta), "r_charge": float(r_charge)}
