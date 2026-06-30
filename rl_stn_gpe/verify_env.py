"""
verify_env.py
=============
Smoke test for STNGPeEnv. Checks the Gymnasium API contract, observation
bounds, finite rewards, and that the post-pulse refractory caps the pulse rate
at STIM_FREQ_HZ.

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


def smoke(condition, n_steps=8, policy="zero", seed=0):
    env = STNGPeEnv(condition=condition, seed=seed)
    obs, info = env.reset(seed=seed)
    assert env.observation_space.contains(obs), f"reset obs out of space: {obs}"
    rews = []
    for _ in range(n_steps):
        if policy == "random":
            a = env.action_space.sample()
        elif policy == "ones":
            a = 1
        else:
            a = 0
        obs, r, term, trunc, info = env.step(a)
        assert env.observation_space.contains(obs), f"step obs out of space: {obs}"
        assert np.isfinite(r), f"non-finite reward: {r}"
        rews.append(r)
    print(f"  [{condition:7}] obs={np.round(obs,3)} "
          f"meanR={np.mean(rews):+.3f} pulses={info['n_pulses']} "
          f"R={info['R']:.3f} H={info['H']:.3f} beta={info['beta']:.1f}")
    return env


if __name__ == "__main__":
    print("=== baseline / agent smoke ===")
    smoke("normal", policy="zero")
    smoke("pd", policy="zero")
    smoke("dbs", policy="zero")     # action ignored; open-loop DBS injected
    smoke("rl", policy="random")    # agent acts

    print("\n=== refractory / max-rate check (rl, always pulse) ===")
    env = STNGPeEnv(condition="rl", seed=0)
    env.reset(seed=0)
    N = 30
    for _ in range(N):
        env.step(1)                 # request a pulse every decision
    expected = len(range(0, N, config.refractory_decisions()))  # one per refractory block
    print(f"  {N} always-pulse decisions -> {env.n_pulses} pulses "
          f"(expected {expected}; refractory={config.refractory_decisions()} decisions)")
    assert env.n_pulses == expected, "refractory did not cap the pulse rate!"

    print("\n=== episode-length / truncation check (rl) ===")
    env = STNGPeEnv(condition="rl", seed=0)
    env.reset(seed=0)
    env.max_decisions = 12          # shrink horizon so the check is fast
    steps, trunc = 0, False
    while not trunc and steps < env.max_decisions + 5:
        _, _, _, trunc, _ = env.step(0)
        steps += 1
    print(f"  truncated after {steps} decisions (horizon set to {env.max_decisions}; "
          f"real horizon = {env.control_steps // env.dec_steps})")
    assert trunc and steps == env.max_decisions, "episode did not truncate at horizon"

    print("\nALL ENV CHECKS PASSED")
