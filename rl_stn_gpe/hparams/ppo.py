"""
PPO hyperparameters (Stable-Baselines3).
On-policy actor-critic with an entropy bonus. Works with continuous (Box) OR
discrete (MultiDiscrete) action spaces. gSDE is continuous-only (train.py
auto-disables it for discrete mode).

HPARAMS -> PPO(...) constructor kwargs.   RUN -> run-level settings (see hparams/__init__).
Tunable knobs used by train.py but NOT PPO kwargs are handled specially:
  - "lr_schedule": "constant" | "linear"  (train.py wraps learning_rate)
  - policy_kwargs.activation_fn given as a STRING here ('tanh'|'relu'|'elu');
    train.py maps it to the torch class.
"""

HPARAMS = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "lr_schedule": "linear",          # 'constant' | 'linear' (decay LR to 0)
    "n_steps": 2048,                  # rollout length per env before each update
    "batch_size": 128,                # minibatch (buffer = n_steps * n_envs)
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,                 # entropy term (exploration)
    "vf_coef": 0.5,                   # value-loss weight
    "max_grad_norm": 0.5,
    "use_sde": True,                  # gSDE smooth exploration (continuous only)
    "policy_kwargs": {
        # pi = actor, vf = critic (independent heads). activation_fn is a string;
        # train.py maps it to the torch module.
        "net_arch": {"pi": [256, 128], "vf": [256, 128]},
        "activation_fn": "tanh",      # 'tanh' | 'relu' | 'elu'
    },
}

RUN = {
    "total_timesteps": 100_000,
    "n_envs": 1,                      # 1 -> DummyVecEnv; >1 -> SubprocVecEnv
    "device": "cpu",                  # 'cpu' recommended on Mac (env-bound)
    "seed": 0,
    "checkpoint_freq": 20_000,
    "eval_freq": 10_000,
    "n_eval_episodes": 3,
}
