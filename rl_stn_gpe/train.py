"""
train.py
========
PPO training for the RL-DBS controller on STNGPeEnv (condition="rl": PD network,
agent-controlled aperiodic pulses).

- Actor-critic PPO (Stable-Baselines3) with separate pi/vf heads and an entropy
  bonus (config.TRAIN).
- Flexible parallelism: n_envs=1 -> DummyVecEnv (single), n_envs>1 ->
  SubprocVecEnv (CPU-parallel). On Mac keep device='cpu' (tiny MLP, env-bound).
- Weights & Biases logging: the full config (RL settings + the simulation YAML
  params) is logged at init; a callback logs per-step reward terms, network
  metrics, and pulse rate.
- Checkpointing + periodic evaluation on a deterministic eval env.

Examples
--------
    # full run (uses config defaults; wandb online)
    python rl_stn_gpe/train.py

    # quick local smoke test (no wandb, tiny budget)
    python rl_stn_gpe/train.py --smoke

    # overrides
    python rl_stn_gpe/train.py --n-envs 4 --timesteps 500000 --run-name "Run 2"
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# stngpe.py uses notebook tqdm; patch before importing stn_gpe (no edit to it).
import tqdm
import tqdm.notebook
tqdm.notebook.tqdm = tqdm.std.tqdm

import numpy as np
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback, CallbackList)

from stn_gpe import load_yaml
from rl_stn_gpe import config
from rl_stn_gpe.env import STNGPeEnv


# ---------------------------------------------------------------------------
def linear_schedule(initial):
    """SB3 schedule: progress_remaining goes 1 -> 0, so LR decays to 0."""
    return lambda progress_remaining: progress_remaining * initial


def make_env(condition="rl", seed=0, params_path=None):
    """Factory returning a thunk that builds a Monitor-wrapped env."""
    def _thunk():
        env = STNGPeEnv(condition=condition, params_path=params_path, seed=seed)
        return Monitor(env)
    return _thunk


def build_vec_env(n_envs, condition="rl", base_seed=0):
    thunks = [make_env(condition=condition, seed=base_seed + i) for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(thunks)
    return SubprocVecEnv(thunks, start_method="spawn")  # spawn: required on macOS


class MetricsLogger(BaseCallback):
    """Logs network metrics, reward terms, and pulse rate from env `info`.

    Everything is recorded via self.logger.record, so it lands in TensorBoard
    and (with sync_tensorboard=True) in wandb on the SAME step stream as SB3's
    own train/* losses (policy_gradient_loss=actor, value_loss=critic,
    entropy_loss). No direct wandb.log here -> no step-counter conflicts.
    """
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        keys = ["R", "beta", "H", "pulse", "r_sync", "r_entropy", "r_beta", "r_stim"]
        for k in keys:
            vals = [i[k] for i in infos if k in i]
            if vals:
                self.logger.record(f"env/{k}", float(np.mean(vals)))
        return True


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="PPO training for RL-DBS controller.")
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=None, dest="n_envs")
    p.add_argument("--device", default=None)
    p.add_argument("--run-name", default=None, dest="run_name")
    p.add_argument("--condition", default="rl")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny local run: no wandb, no eval, small budget.")
    # --- hyperparameter overrides (for manual tuning / wandb sweeps) ---
    p.add_argument("--lr", type=float, default=None, help="learning rate")
    p.add_argument("--ent-coef", type=float, default=None, dest="ent_coef")
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--n-steps", type=int, default=None, dest="n_steps_arg")
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--net-arch", default=None, dest="net_arch",
                   help="Hidden layer sizes for pi & vf, e.g. '128,128' or '256,128'.")
    p.add_argument("--lr-schedule", default=None, dest="lr_schedule",
                   choices=["constant", "linear"])
    args = p.parse_args()

    T = dict(config.TRAIN)
    # apply hyperparameter overrides onto the config copy
    if args.lr is not None:         T["learning_rate"] = args.lr
    if args.ent_coef is not None:   T["ent_coef"] = args.ent_coef
    if args.batch_size is not None: T["batch_size"] = args.batch_size
    if args.gamma is not None:      T["gamma"] = args.gamma
    if args.lr_schedule is not None: T["lr_schedule"] = args.lr_schedule
    if args.net_arch is not None:
        layers = [int(x) for x in args.net_arch.split(",")]
        T["net_arch"] = {"pi": layers, "vf": layers}

    total_timesteps = args.timesteps or T["total_timesteps"]
    n_envs = args.n_envs or T["n_envs"]
    device = args.device or T["device"]
    use_wandb = not (args.no_wandb or args.smoke)
    do_eval = not (args.no_eval or args.smoke)

    # PPO rollout length: override / full default / tiny for the smoke test.
    n_steps = args.n_steps_arg or T["n_steps"]
    if args.smoke:
        total_timesteps, n_envs, n_steps = 64, 1, 16

    os.makedirs(config.CKPT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # --- W&B ---
    run = None
    if use_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        sim_params = load_yaml(config.get_condition(args.condition)["params"])
        wcfg = config.wandb_config(sim_params)
        wcfg["train"] = T   # log the effective (possibly overridden) hyperparameters
        run = wandb.init(
            project=config.WANDB["project"],
            name=args.run_name or config.WANDB["run_name"],
            entity=config.WANDB["entity"],
            mode=config.WANDB["mode"],
            sync_tensorboard=config.WANDB["sync_tensorboard"],
            tags=config.WANDB["tags"],
            config=wcfg,
        )

    # --- envs ---
    vec_env = build_vec_env(n_envs, condition=args.condition, base_seed=T["seed"])

    # --- model (actor-critic PPO + entropy bonus) ---
    act_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}
    policy_kwargs = dict(
        net_arch=T["net_arch"],
        activation_fn=act_map[T.get("activation_fn", "tanh")],
    )
    # learning rate: constant value or a linear decay-to-zero schedule
    lr = T["learning_rate"]
    if T.get("lr_schedule", "constant") == "linear":
        lr = linear_schedule(T["learning_rate"])

    model = PPO(
        T["policy"], vec_env,
        learning_rate=lr, n_steps=n_steps,
        batch_size=T["batch_size"], n_epochs=T["n_epochs"],
        gamma=T["gamma"], gae_lambda=T["gae_lambda"], clip_range=T["clip_range"],
        ent_coef=T["ent_coef"], vf_coef=T.get("vf_coef", 0.5),
        max_grad_norm=T.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
        seed=T["seed"], device=device, verbose=1,
        tensorboard_log=config.LOG_DIR,
    )
    print(f"PPO: net_arch={T['net_arch']} act={T['activation_fn']} "
          f"lr={T['learning_rate']}({T.get('lr_schedule','constant')}) "
          f"ent_coef={T['ent_coef']} batch={T['batch_size']} n_steps={n_steps} "
          f"gamma={T['gamma']} n_envs={n_envs}")

    # --- callbacks ---
    callbacks = [MetricsLogger(),
                 CheckpointCallback(save_freq=max(T["checkpoint_freq"] // n_envs, 1),
                                    save_path=config.CKPT_DIR, name_prefix="ppo_dbs")]
    if do_eval:
        eval_env = DummyVecEnv([make_env(condition=args.condition, seed=T["seed"] + 999)])
        callbacks.append(EvalCallback(
            eval_env, best_model_save_path=config.CKPT_DIR,
            eval_freq=max(T["eval_freq"] // n_envs, 1),
            n_eval_episodes=T["n_eval_episodes"], deterministic=True))
    if use_wandb:
        from wandb.integration.sb3 import WandbCallback
        callbacks.append(WandbCallback(model_save_path=config.CKPT_DIR, verbose=1))

    # --- train ---
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))

    final = os.path.join(config.CKPT_DIR, "ppo_dbs_final.zip")
    model.save(final)
    print(f"saved final model -> {final}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
