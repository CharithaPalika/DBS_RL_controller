"""
TD3 hyperparameters (Stable-Baselines3).
Off-policy, deterministic policy (twin-delayed DDPG). CONTINUOUS actions ONLY
(ACTION_MODE must be "continuous"). Explores via injected ACTION NOISE (not an
entropy bonus, and NOT gSDE), configured under RUN["action_noise"] and built by
train.py to match the action dimension. Uses a replay buffer.

HPARAMS -> TD3(...) constructor kwargs.   RUN -> run-level settings.
`policy_kwargs.activation_fn` is a STRING here; train.py maps it to the torch class.
"""

HPARAMS = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "buffer_size": 200_000,
    "learning_starts": 1_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "policy_delay": 2,                # delayed policy/target updates (the "D")
    "policy_kwargs": {
        "net_arch": [256, 256],
        "activation_fn": "relu",      # 'tanh' | 'relu' | 'elu'
    },
}

RUN = {
    "total_timesteps": 100_000,
    "n_envs": 1,
    "device": "cpu",
    "seed": 0,
    "checkpoint_freq": 20_000,
    "eval_freq": 10_000,
    "n_eval_episodes": 3,
    # Gaussian exploration noise added to actions (sigma is in ACTION units;
    # the action space is Box([-1,1]^k), so 0.1 = 10% of half-range per dim).
    "action_noise": {"kind": "normal", "sigma": 0.1},
}
