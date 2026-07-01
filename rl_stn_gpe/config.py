"""
config.py - single source of truth for the RL-DBS controller.

Holds paths, run conditions, control timing, pulse shape, episode timing,
observation/reward settings, PPO hyperparameters, and Weights & Biases logging.

IMPORTANT: `dt` and `stn_gpe_units` are NOT defined here. They live in the
params YAML (the validated model's source of truth) and are read at runtime by
the environment. Anything expressed in milliseconds here is converted to
integrator steps using that `dt` via the helpers at the bottom.
"""

import os

# ===========================================================================
# Paths
# ===========================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARAMS_DIR = os.path.join(PROJECT_ROOT, "params", "stn_gpe_params")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "rl_stn_gpe", "outputs")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")


# ===========================================================================
# Run conditions - the normal/pd/dbs/rl switch passed to the environment.
#   stim:
#     "none"     -> no stimulation (baseline rollout)
#     "openloop" -> standard DBS taken VERBATIM from the params YAML
#                   (faithful comparison; uses the YAML's DBS_* + pulseinterval)
#     "agent"    -> RL agent controls aperiodic pulses (uses PULSE below)
# A params path can be overridden at the env call site if desired.
# ===========================================================================
CONDITIONS = {
    "normal": {"params": os.path.join(PARAMS_DIR, "params_Normal.yaml"), "stim": "none"},
    "pd":     {"params": os.path.join(PARAMS_DIR, "params_PD.yaml"),      "stim": "none"},
    "dbs":    {"params": os.path.join(PARAMS_DIR, "params_std_DBS_ms.yaml"), "stim": "openloop"},
    "rl":     {"params": os.path.join(PARAMS_DIR, "params_RL.yaml"),      "stim": "agent"},
}


def get_condition(name):
    if name not in CONDITIONS:
        raise ValueError(f"Unknown condition '{name}'. Choose from {list(CONDITIONS)}.")
    return CONDITIONS[name]


# ===========================================================================
# Control timing  (UNIFORM-STEP model)
#   - The agent decides every DECISION_DT_MS of biological time (fixed).
#   - DECISION_DT_MS MUST be < one pulse period (1000 / STIM_FREQ_HZ).
#   - When a pulse fires, new pulses are blocked for one full period, i.e. for
#     round((1000 / STIM_FREQ_HZ) / DECISION_DT_MS) decisions. This enforces the
#     max pulse rate (<= STIM_FREQ_HZ) while letting timing be aperiodic.
# ===========================================================================
STIM_FREQ_HZ = 40       # maximum pulse rate (Hz)
DECISION_DT_MS = 1.5 #5.0    # agent decision interval (bio ms); must be < 1000/STIM_FREQ_HZ


# ===========================================================================
# DBS pulse (AGENT) - charge-balanced biphasic: +amplitude for W, then
# -amplitude for W (equal widths & opposite amplitudes => net charge = 0).
# Widths in MS, converted to integrator steps at runtime using dt from YAML.
# Defaults seeded from std DBS (params_std_DBS.yaml: A=250, duty 0.052 @130 Hz).
#   width_ms default = duty * period = 0.052 * (1000/130) ~= 0.4 ms.
# NOTE: the std-DBS 'pulseinterval' (=10) is a unitless offset multiplier inside
# GenerateDBS.biphasicDBS, not a millisecond value, so it is NOT ported here.
# interphase_ms is the gap between the +/- phases of the agent pulse (default 0).
# The "dbs" (open-loop) condition still uses the YAML pulseinterval verbatim.
# ===========================================================================
PULSE = {
    "amplitude": 100,        # +amplitude then -amplitude  => net charge 0 (matches std DBS)
    "phase_width_ms": 0.2,   # per-phase width (matches std DBS)
    "interphase_ms": 1.0,    # gap between + and - phase (matches std DBS)
}


# ===========================================================================
# Episode timing (seconds).
# Warmup runs with NO stimulation for ALL conditions (including 'dbs'): the
# network settles and the rolling metric buffer fills; THEN the condition's
# stimulation engages. Warmup steps are discarded from training/eval.
# ===========================================================================
WARMUP_S = 0.25
CONTROL_S = 2.0
METRIC_WINDOW_S = 0.25    # rolling window for obs/reward (in integrator steps)

# Recompute the expensive window metrics (synchrony/entropy/beta) only every N
# decisions and cache them; the per-pulse penalty is still applied every step.
# The 1 s window changes <1% per 5 ms step, so caching is ~N x faster with
# negligible signal loss. Set to 1 for exact per-step metrics.
METRIC_RECOMPUTE_EVERY = 1


# ===========================================================================
# Observation (STN only) + reward
# ===========================================================================
# Frequency settings for the window metrics (Hz).
BETA_BAND = (10.5, 35.5)     # beta-band power range (matches original Analysis full beta)
ENTROPY_FMAX = 35            # upper frequency bound for spectral entropy

OBS_METRICS = ["synchrony", "beta_power", "entropy"]
# (min, max) used to scale each metric into ~[0, 1] for the policy network.
# beta_power comes back in dB (Analysis.power_beta); ranges to be tuned after
# the first PD/Normal/DBS calibration runs.
OBS_NORM = {
    "synchrony":  (0.0, 1.0),
    "beta_power": (40.0, 120.0),   # provisional (observed PD~95, DBS~82 dB); recalibrate
    "entropy":    (0.0, 1.0),
}

# Threshold-band ("tent") reward. For each metric:
#     term = w * max(-1, 1 - err/tol)
# where err is the distance on the BAD side of target. -> positive within the
# tolerance band, 0 at the edge, negative (penalty) beyond. Each term in [-w, w].
# beta_power is included here (lower beta = healthier). Targets/tols are tunable;
# beta target/tol are in dB and should be recalibrated from a metrics pass.
REWARD = {
    "target_sync":    0.2, "tol_sync":    0.15, "w_sync":    3.0,  # lower better
    "target_entropy": 0.65,  "tol_entropy": 0.20, "w_entropy": 0.5,  # higher better
    "target_beta":    70.0,  "tol_beta":    15.0, "w_beta":    0.0,  # lower better (dB)
    "lambda": 0.1,           # penalty per emitted pulse (energy / sparsity)
    # Negative side grows with distance beyond the band (far-from-optimal is
    # punished more). neg_clip floors each term for stability; None = unbounded.
    "neg_clip": None,
}


# ===========================================================================
# PPO  (Stable-Baselines3; actor-critic with an entropy bonus)
#   SB3 PPO is inherently actor-critic; net_arch gives separate pi/vf heads and
#   ent_coef adds the entropy term.
# ===========================================================================
TRAIN = {
    "policy": "MlpPolicy",
    # Network size/depth. Edit these lists to change it: e.g. [128,128] (wider),
    # [256,256,128] (deeper). pi = actor, vf = critic (independent).
    # "net_arch": {"pi": [64, 64], "vf": [64, 64]},
    "net_arch": {"pi": [256, 128], "vf": [256, 128]},
    "activation_fn": "tanh",   # 'tanh' | 'relu' | 'elu'  (hidden-layer activation)
    "ent_coef": 0.01,        # entropy term (exploration)
    "learning_rate": 3e-4,
    "lr_schedule": "linear",  # 'constant' | 'linear' (decay LR to 0 over training)
    "n_steps": 2048,          # rollout length per env before each PPO update
    "batch_size": 128,        # minibatch size (buffer = n_steps * n_envs)
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "vf_coef": 0.5,           # value-loss weight
    "max_grad_norm": 0.5,     # gradient clipping
    "total_timesteps": 50_000, #100_000, #200_000,
    "seed": 0,
    "checkpoint_freq": 10_000,
    # Parallelism / device (MacBook: keep device 'cpu' - tiny MLP, env is the
    # bottleneck; use n_envs>1 for CPU-parallel speedup).
    "n_envs": 1,             # 1 -> DummyVecEnv (single); >1 -> SubprocVecEnv
    "device": "cpu",         # 'cpu' recommended on Mac; 'auto'/'mps' optional
    # Evaluation during training (deterministic PD-rl env, fixed seed).
    "eval_freq": 10_000,
    "n_eval_episodes": 3,
}


# ===========================================================================
# Weights & Biases logging
#   train.py calls wandb.init(project=..., name=..., config=wandb_config(params))
#   and passes WandbCallback to model.learn. The simulation params being run are
#   merged into the logged config (see wandb_config()).
# ===========================================================================
WANDB = {
    "project": "RL controller",
    "run_name": "Run 1",
    "entity": None,
    "mode": "online",
    "sync_tensorboard": True,
    "tags": ["ppo", "stn-gpe", "dbs"],
}


# ===========================================================================
# Helpers
# ===========================================================================
def ms_to_steps(ms, dt_ms):
    """Convert a duration in ms to integer integrator steps."""
    return int(round(ms / dt_ms))


def decision_steps(dt_ms):
    """Integrator steps advanced per agent decision (uniform-step model)."""
    return ms_to_steps(DECISION_DT_MS, dt_ms)


def refractory_decisions():
    """How many decisions a pulse blocks subsequent pulses (max-rate cap)."""
    period_ms = 1000.0 / STIM_FREQ_HZ
    return int(round(period_ms / DECISION_DT_MS))


def validate(dt_ms):
    """Sanity-check the timing/pulse settings against the YAML dt."""
    period_ms = 1000.0 / STIM_FREQ_HZ
    assert DECISION_DT_MS < period_ms, (
        f"DECISION_DT_MS ({DECISION_DT_MS}) must be < one pulse period "
        f"({period_ms} ms = 1000/STIM_FREQ_HZ)."
    )
    pulse_ms = 2 * PULSE["phase_width_ms"] + PULSE["interphase_ms"]
    assert pulse_ms <= DECISION_DT_MS, (
        f"Pulse duration ({pulse_ms} ms) must fit within one decision step "
        f"({DECISION_DT_MS} ms)."
    )


def wandb_config(sim_params=None):
    """Assemble the full config dict to log to W&B.

    Combines the RL settings with the simulation params actually being run so a
    run is fully reproducible from its W&B config. `sim_params` is the loaded
    YAML dict, passed in by train.py at init time.
    """
    cfg = {
        "stim_freq_hz": STIM_FREQ_HZ,
        "decision_dt_ms": DECISION_DT_MS,
        "pulse": PULSE,
        "warmup_s": WARMUP_S,
        "control_s": CONTROL_S,
        "metric_window_s": METRIC_WINDOW_S,
        "obs_metrics": OBS_METRICS,
        "obs_norm": OBS_NORM,
        "beta_band": BETA_BAND,
        "entropy_fmax": ENTROPY_FMAX,
        "reward": REWARD,
        "train": TRAIN,
    }
    if sim_params is not None:
        cfg["sim_params"] = sim_params
    return cfg
