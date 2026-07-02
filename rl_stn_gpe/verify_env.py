"""
verify_env.py
=============
Smoke test for STNGPeEnv (event-driven, configurable action space). Checks:
  - the Gymnasium API contract + observation/action spaces, for both
    ACTION_MODE = "continuous" (Box) and "discrete" (MultiDiscrete);
  - finite rewards and in-space observations;
  - the structural max-frequency cap (decoded pulse_period_ms stays within the
    frequency band) for the agent condition;
  - time-based episode truncation.

Run:  python rl_stn_gpe/verify_env.py
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# stngpe.py uses notebook tqdm; patch before importing stn_gpe (no edit to it).
import tqdm
import tqdm.notebook
tqdm.notebook.tqdm = tqdm.std.tqdm

import numpy as np
from rl_stn_gpe.env import STNGPeEnv
from rl_stn_gpe import config


def smoke(condition, n_steps=8, policy="random", seed=0):
    env = STNGPeEnv(condition=condition, seed=seed)
    obs, info = env.reset(seed=seed)
    assert env.observation_space.contains(obs), f"reset obs out of space: {obs}"
    rews = []
    for _ in range(n_steps):
        a = env.action_space.sample() if policy == "random" else 0
        obs, r, term, trunc, info = env.step(a)
        assert env.observation_space.contains(obs), f"step obs out of space: {obs}"
        assert np.isfinite(r), f"non-finite reward: {r}"
        rews.append(r)
    print(f"  [{condition:7}] obs_dim={obs.shape[0]} act={env.action_space} "
          f"mean_reward={np.mean(rews):+.3f} pulses={info['n_pulses']} "
          f"synchrony={info['R']:.3f} entropy={info['H']:.3f} beta={info['beta']:.1f}")
    return env


def check_mode(mode):
    print(f"\n=== ACTION_MODE = {mode} ===")
    config.ACTION_MODE = mode
    smoke("normal", policy="zero")
    smoke("pd", policy="zero")
    smoke("dbs", policy="zero")      # action ignored; open-loop DBS injected
    smoke("rl", policy="random")     # agent acts

    # --- frequency-cap check: decoded period within the band for random actions ---
    env = STNGPeEnv(condition="rl", seed=0)
    p_lo, p_hi = config.period_bounds_ms()
    periods = []
    for _ in range(200):
        vals, _ = env._decode(env.action_space.sample())
        periods.append(vals["pulse_period_ms"])
    pmin, pmax = min(periods), max(periods)
    print(f"  freq cap: period in [{pmin:.2f}, {pmax:.2f}] ms; "
          f"band=[{p_lo:.2f}, {p_hi:.2f}] (<= {config.MAX_FREQ_HZ} Hz)")
    assert pmin >= p_lo - 1e-6 and pmax <= p_hi + 1e-6, "period escaped the freq band!"


if __name__ == "__main__":
    saved_mode = config.ACTION_MODE
    try:
        check_mode("continuous")
        check_mode("discrete")
    finally:
        config.ACTION_MODE = saved_mode

    print("\n=== episode-length / truncation check (rl, continuous) ===")
    config.ACTION_MODE = "continuous"
    env = STNGPeEnv(condition="rl", seed=0)
    env.reset(seed=0)
    env.control_steps = 3000          # shrink horizon so the check is fast
    steps, trunc = 0, False
    while not trunc and steps < 500:
        _, _, _, trunc, _ = env.step(env.action_space.sample())
        steps += 1
    print(f"  truncated after {steps} decisions "
          f"(steps_done={env.steps_done} >= control_steps={env.control_steps})")
    assert trunc and env.steps_done >= env.control_steps, "episode did not truncate"

    config.ACTION_MODE = saved_mode
    print("\nALL ENV CHECKS PASSED")
