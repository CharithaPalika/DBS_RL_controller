"""
rl_stn_gpe.hparams
==================
Per-algorithm hyperparameter files for the RL-DBS controller. One module per
Stable-Baselines3 algorithm (ppo, sac, td3). Each exports two dicts:

    HPARAMS : passed verbatim (**HPARAMS) to the SB3 algorithm constructor.
              Contains ONLY constructor kwargs for that algorithm.
    RUN     : run-level settings that are NOT constructor kwargs:
                n_envs, total_timesteps, seed, device,
                checkpoint_freq, eval_freq, n_eval_episodes,
                (td3 only) action_noise = {"kind": "normal", "sigma": ...}

train.py selects the file via config.ALGO (or --algo) and calls load_hparams().
Edit these files to tune each algorithm independently — they do not affect each
other. `policy_kwargs` (net_arch / activation) also lives inside HPARAMS here.
"""

import importlib


def load(algo):
    """Return (HPARAMS, RUN) dicts (copies) for the given algo name."""
    algo = algo.lower()
    if algo not in ("ppo", "sac", "td3"):
        raise ValueError(f"Unknown algo '{algo}'. Choose from ppo | sac | td3.")
    m = importlib.import_module(f"rl_stn_gpe.hparams.{algo}")
    return dict(m.HPARAMS), dict(m.RUN)
