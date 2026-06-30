"""
reward.py
=========
Observation construction and reward computation for STNGPeEnv, driven by
config.OBS_NORM and config.REWARD.

Observation, each scaled to ~[0, 1] via OBS_NORM, in OBS_METRICS order, with the
agent's previous (normalized) action appended (so the policy stays Markov):
    [synchrony R, beta_power (dB), entropy H,  <last action in [0,1]^k>]

Reward (per decision, over the trailing metric window): a threshold-band "tent"
per metric (synchrony, entropy, beta power), TIME-SCALED by the decision's
elapsed time, minus a per-pulse CHARGE penalty.
    term      = w * (1 - err/tol)           # err = distance on the BAD side of target
    metric    = (term_sync + term_entropy + term_beta) * (elapsed_ms / ref_ms)
    reward    = metric - lambda_charge * charge
-> metric terms positive inside the tolerance band, 0 at the edge, negative
beyond. Time-scaling makes the episode return ~ the time-integral of quality
(semi-MDP correction); the charge term is the per-pulse energy cost.
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


def compute_reward(R, H, beta, charge=0.0, elapsed_ms=None, cfg=None):
    """Compute the scalar reward and a per-term breakdown (for logging).

    Parameters
    ----------
    R : float          - synchrony over the window    (lower is better)
    H : float          - spectral entropy             (higher is better)
    beta : float       - beta-band power in dB         (lower is better)
    charge : float     - |injected charge| this decision (energy cost; >= 0)
    elapsed_ms : float - biological time advanced this decision; scales the metric
                         terms (semi-MDP correction). None => no scaling (factor 1).

    The three metric terms are summed and scaled by (elapsed_ms / ref_ms); the
    charge penalty is applied per decision (not time-scaled).
    """
    cfg = cfg or config.REWARD
    nc = cfg.get("neg_clip")
    r_sync = _tent(max(0.0, R - cfg["target_sync"]), cfg["tol_sync"], cfg["w_sync"], nc)
    r_entropy = _tent(max(0.0, cfg["target_entropy"] - H), cfg["tol_entropy"], cfg["w_entropy"], nc)
    r_beta = _tent(max(0.0, beta - cfg["target_beta"]), cfg["tol_beta"], cfg["w_beta"], nc)

    scale = 1.0 if elapsed_ms is None else (elapsed_ms / cfg.get("ref_ms", 25.0))
    r_metric = (r_sync + r_entropy + r_beta) * scale
    r_charge = -cfg.get("lambda_charge", 0.0) * float(charge)
    total = float(r_metric + r_charge)
    return total, {"r_sync": r_sync, "r_entropy": r_entropy, "r_beta": r_beta,
                   "r_metric": float(r_metric), "r_charge": float(r_charge)}
