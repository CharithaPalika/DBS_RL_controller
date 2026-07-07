"""
train.py
========
RL training for the RL-DBS controller on STNGPeEnv (condition="rl": PD network,
agent-controlled aperiodic pulses).

- Algorithm selectable via config.ALGO or --algo: PPO (on-policy actor-critic),
  SAC or TD3 (off-policy, continuous-only, more sample-efficient). Each algo's
  hyperparameters live in rl_stn_gpe/hparams/<algo>.py (HPARAMS + RUN dicts).
- Optional observation normalization via SB3 VecNormalize (config.USE_VECNORMALIZE);
  running stats are saved to checkpoints/vecnormalize.pkl for eval.
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

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecNormalize)
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback, CallbackList)

from stn_gpe import load_yaml
from rl_stn_gpe import config
from rl_stn_gpe import hparams as hp_loader
from rl_stn_gpe.env import STNGPeEnv


# Algorithm registry (config.ALGO / --algo selects one).
ALGO_REGISTRY = {"ppo": PPO, "sac": SAC, "td3": TD3}
ACT_MAP = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}


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


def maybe_vecnormalize(vec_env, training):
    """Wrap a VecEnv in VecNormalize per config (no-op if disabled).

    training=True for the learning env; training=False (+ norm_reward off) for
    eval so its running stats are frozen and rewards stay on their true scale.
    """
    if not config.USE_VECNORMALIZE:
        return vec_env
    kw = dict(config.VECNORM)
    if not training:
        kw["norm_reward"] = False
    return VecNormalize(vec_env, training=training, **kw)


def build_model(algo, vec_env, hp, run, device, seed):
    """Construct an SB3 model for `algo` from its HPARAMS (hp) + RUN (run) dicts.

    Handles the pieces that are NOT plain constructor kwargs: activation-string ->
    torch class, linear LR schedule, gSDE gating (continuous only), and TD3
    action noise sized to the action dimension.
    """
    hp = dict(hp)
    algo = algo.lower()

    # policy_kwargs: map the activation STRING to a torch module class.
    pk = dict(hp.get("policy_kwargs", {}))
    if isinstance(pk.get("activation_fn"), str):
        pk["activation_fn"] = ACT_MAP[pk["activation_fn"]]
    hp["policy_kwargs"] = pk

    # learning-rate schedule (non-constructor key -> wrap learning_rate).
    if hp.pop("lr_schedule", "constant") == "linear":
        hp["learning_rate"] = linear_schedule(hp["learning_rate"])

    policy = hp.pop("policy", "MlpPolicy")

    if algo in ("sac", "td3"):
        assert config.ACTION_MODE == "continuous", (
            f"{algo.upper()} requires ACTION_MODE == 'continuous' "
            f"(got '{config.ACTION_MODE}'). Use PPO for discrete actions.")
    else:  # ppo: gSDE is continuous-only -> auto-disable for discrete
        if "use_sde" in hp:
            hp["use_sde"] = bool(hp["use_sde"]) and (config.ACTION_MODE == "continuous")

    if algo == "td3":
        an = run.get("action_noise")
        if an and an.get("kind") == "normal":
            k = vec_env.action_space.shape[0]
            hp["action_noise"] = NormalActionNoise(
                mean=np.zeros(k), sigma=an["sigma"] * np.ones(k))

    return ALGO_REGISTRY[algo](
        policy, vec_env, seed=seed, device=device, verbose=1,
        tensorboard_log=config.LOG_DIR, **hp)


class MetricsLogger(BaseCallback):
    """Logs network metrics, reward terms, and pulse rate from env `info`.

    Everything is recorded via self.logger.record, so it lands in TensorBoard
    and (with sync_tensorboard=True) in wandb on the SAME step stream as SB3's
    own train/* losses (policy_gradient_loss=actor, value_loss=critic,
    entropy_loss). No direct wandb.log here -> no step-counter conflicts.
    """
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        keys = ["R", "beta", "H", "pulse", "stim_charge",
                "amplitude", "pulse_period_ms", "phase_width_ms", "interphase_gap_ms",
                "r_sync", "r_entropy", "r_beta", "r_charge"]
        for k in keys:
            vals = [i[k] for i in infos if k in i and np.isfinite(i[k])]
            if vals:
                self.logger.record(f"env/{k}", float(np.mean(vals)))
        return True


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Train an RL-DBS controller (PPO/SAC/TD3).")
    p.add_argument("--algo", default=None, choices=["ppo", "sac", "td3"],
                   help="RL algorithm (default: config.ALGO). Hparams from hparams/<algo>.py.")
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
    p.add_argument("--n-steps", type=int, default=None, dest="n_steps_arg",
                   help="PPO rollout length (ignored by SAC/TD3).")
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--net-arch", default=None, dest="net_arch",
                   help="Hidden layer sizes, e.g. '256,128'.")
    p.add_argument("--lr-schedule", default=None, dest="lr_schedule",
                   choices=["constant", "linear"])
    args = p.parse_args()

    # --- select algorithm + load its hparams (HPARAMS) and run settings (RUN) ---
    algo = (args.algo or config.ALGO).lower()
    HP, RUN = hp_loader.load(algo)

    # apply CLI hyperparameter overrides (only where the key applies to this algo)
    if args.lr is not None:                              HP["learning_rate"] = args.lr
    if args.ent_coef is not None and "ent_coef" in HP:   HP["ent_coef"] = args.ent_coef
    if args.batch_size is not None:                      HP["batch_size"] = args.batch_size
    if args.gamma is not None:                           HP["gamma"] = args.gamma
    if args.lr_schedule is not None:                     HP["lr_schedule"] = args.lr_schedule
    if args.n_steps_arg is not None and "n_steps" in HP: HP["n_steps"] = args.n_steps_arg
    if args.net_arch is not None:
        layers = [int(x) for x in args.net_arch.split(",")]
        HP.setdefault("policy_kwargs", {})
        # PPO uses a dict(pi/vf); off-policy algos use a flat list.
        HP["policy_kwargs"]["net_arch"] = (
            {"pi": layers, "vf": layers} if algo == "ppo" else layers)

    seed = RUN["seed"]
    total_timesteps = args.timesteps or RUN["total_timesteps"]
    n_envs = args.n_envs or RUN["n_envs"]
    device = args.device or RUN["device"]
    use_wandb = not (args.no_wandb or args.smoke)
    do_eval = not (args.no_eval or args.smoke)

    if args.smoke:
        total_timesteps, n_envs = 256, 1
        if "n_steps" in HP:           HP["n_steps"] = 16          # ppo
        if "learning_starts" in HP:   HP["learning_starts"] = 16  # sac/td3
        if "batch_size" in HP:        HP["batch_size"] = min(HP["batch_size"], 32)

    # Global seeding (python / numpy / torch) for a reproducible run.
    set_random_seed(seed)
    print(f"algo = {algo.upper()} | seed = {seed} | vecnormalize = {config.USE_VECNORMALIZE}")

    os.makedirs(config.CKPT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    prefix = f"{algo}_dbs"

    # --- W&B ---
    run = None
    if use_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        sim_params = load_yaml(config.get_condition(args.condition)["params"])
        wcfg = config.wandb_config(sim_params, train={"algo": algo, "hparams": HP, "run": RUN})
        run = wandb.init(
            project=config.WANDB["project"],
            name=args.run_name or config.WANDB["run_name"],
            entity=config.WANDB["entity"],
            mode=config.WANDB["mode"],
            sync_tensorboard=config.WANDB["sync_tensorboard"],
            tags=list(config.WANDB["tags"]) + [algo],
            config=wcfg,
        )

    # --- envs (+ optional VecNormalize) ---
    vec_env = build_vec_env(n_envs, condition=args.condition, base_seed=seed)
    vec_env = maybe_vecnormalize(vec_env, training=True)

    # --- model ---
    model = build_model(algo, vec_env, HP, RUN, device, seed)
    print(f"{algo.upper()}: action_mode={config.ACTION_MODE} "
          f"net_arch={HP.get('policy_kwargs', {}).get('net_arch')} "
          f"lr={HP['learning_rate']} n_envs={n_envs} total={total_timesteps}")

    # --- callbacks ---
    callbacks = [MetricsLogger(),
                 CheckpointCallback(save_freq=max(RUN["checkpoint_freq"] // n_envs, 1),
                                    save_path=config.CKPT_DIR, name_prefix=prefix)]
    if do_eval:
        eval_env = build_vec_env(1, condition=args.condition, base_seed=seed + 999)
        eval_env = maybe_vecnormalize(eval_env, training=False)  # SB3 syncs stats pre-eval
        callbacks.append(EvalCallback(
            eval_env, best_model_save_path=config.CKPT_DIR,
            eval_freq=max(RUN["eval_freq"] // n_envs, 1),
            n_eval_episodes=RUN["n_eval_episodes"], deterministic=True))
    if use_wandb:
        from wandb.integration.sb3 import WandbCallback
        callbacks.append(WandbCallback(model_save_path=config.CKPT_DIR, verbose=1))

    # --- train ---
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks),
                progress_bar=True)

    final = os.path.join(config.CKPT_DIR, f"{prefix}_final.zip")
    model.save(final)
    print(f"saved final model -> {final}")
    # Persist VecNormalize running stats so eval can reproduce the obs scaling.
    if config.USE_VECNORMALIZE:
        vn = model.get_vec_normalize_env()
        if vn is not None:
            vn_path = os.path.join(config.CKPT_DIR, "vecnormalize.pkl")
            vn.save(vn_path)
            print(f"saved VecNormalize stats -> {vn_path}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
