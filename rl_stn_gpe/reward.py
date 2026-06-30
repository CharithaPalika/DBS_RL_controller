"""
reward.py
=========
Observation construction and reward computation for STNGPeEnv, driven by
config.OBS_NORM and config.REWARD.

Observation (STN only), each scaled to ~[0, 1] via OBS_NORM, in OBS_METRICS order:
    [synchrony R, beta_power (dB), entropy H]

Reward (per decision, over the trailing metric window): a threshold-band "tent"
per metric (synchrony, entropy, beta power), minus a per-pulse penalty.
    term = w * max(-1, 1 - err/tol)   # err = distance on the BAD side of target
    reward = term_sync + term_entropy + term_beta - lambda * pulse
-> positive inside the tolerance band, 0 at the edge, negative beyond; each term
in [-w, +w] so the total reward is bounded.
"""

import numpy as np
from rl_stn_gpe import config


def _scale(value, lo, hi):
    return float(np.clip((value - lo) / (hi - lo + 1e-12), 0.0, 1.0))


def make_observation(R, beta, H):
    """Return the normalized observation vector in config.OBS_METRICS order."""
    vals = {"synchrony": R, "beta_power": beta, "entropy": H}
    obs = [_scale(vals[k], *config.OBS_NORM[k]) for k in config.OBS_METRICS]
    return np.asarray(obs, dtype=np.float32)


def _tent(err, tol, w, neg_clip=None):
    """One-sided threshold band.

    +w at/past target, 0 at the tolerance edge, then NEGATIVE and growing with
    distance beyond the band (so 'far from optimal' is punished more). The
    positive side is naturally capped at +w (err>=0). `neg_clip` optionally
    floors the negative side for stability; None = unbounded (errs are bounded
    in practice, so the reward stays in a reasonable range).
    """
    val = w * (1.0 - err / tol)
    if neg_clip is not None:
        val = max(neg_clip, val)
    return float(val)


def compute_reward(R, H, beta, emit, cfg=None):
    """Compute the scalar reward and a per-term breakdown (for logging).

    Parameters
    ----------
    R : float    - synchrony over the window   (lower is better)
    H : float    - spectral entropy            (higher is better)
    beta : float - beta-band power in dB        (lower is better)
    emit : bool  - whether a pulse was actually emitted this decision
    """
    cfg = cfg or config.REWARD
    nc = cfg.get("neg_clip")
    r_sync = _tent(max(0.0, R - cfg["target_sync"]), cfg["tol_sync"], cfg["w_sync"], nc)
    r_entropy = _tent(max(0.0, cfg["target_entropy"] - H), cfg["tol_entropy"], cfg["w_entropy"], nc)
    r_beta = _tent(max(0.0, beta - cfg["target_beta"]), cfg["tol_beta"], cfg["w_beta"], nc)
    r_stim = -cfg["lambda"] * (1.0 if emit else 0.0)
    total = float(r_sync + r_entropy + r_beta + r_stim)
    return total, {"r_sync": r_sync, "r_entropy": r_entropy,
                   "r_beta": r_beta, "r_stim": r_stim}
