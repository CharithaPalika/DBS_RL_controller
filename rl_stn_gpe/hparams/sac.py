"""
SAC hyperparameters (Stable-Baselines3).
Off-policy, entropy-regularized actor-critic. CONTINUOUS actions ONLY
(ACTION_MODE must be "continuous"). Far more sample-efficient than PPO, which
matters here because each env step is an expensive spiking-network simulation.
Supports gSDE. Uses a replay buffer (no n_steps / clip_range like PPO).

HPARAMS -> SAC(...) constructor kwargs.   RUN -> run-level settings.
`policy_kwargs.activation_fn` is a STRING here; train.py maps it to the torch class.
"""

HPARAMS = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "buffer_size": 200_000,           # replay buffer capacity (transitions)
    "learning_starts": 1_000,         # random steps before learning begins
    "batch_size": 256,
    "tau": 0.005,                     # target-network soft-update rate
    "gamma": 0.99,
    "train_freq": 1,                  # gradient step every N env steps
    "gradient_steps": 1,
    "ent_coef": "auto",              # auto-tuned entropy temperature
    "use_sde": True,                  # gSDE smooth exploration
    "policy_kwargs": {
        "net_arch": [256, 256],       # shared actor/critic MLP sizes
        "activation_fn": "relu",      # 'tanh' | 'relu' | 'elu'
    },
}

RUN = {
    "total_timesteps": 100_000,
    "n_envs": 1,                      # off-policy: 1 env is typical/simplest
    "device": "cpu",
    "seed": 0,
    "checkpoint_freq": 20_000,
    "eval_freq": 10_000,
    "n_eval_episodes": 3,
}
