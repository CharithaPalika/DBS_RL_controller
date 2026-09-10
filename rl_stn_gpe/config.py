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
# Control timing
#   - Baseline conditions (normal / pd / dbs) advance in fixed chunks of
#     DECISION_DT_MS of biological time; their action is ignored.
#   - The RL agent is EVENT-DRIVEN: each env step emits ONE biphasic pulse and
#     advances the simulator by the agent-chosen pulse_period_ms (the gap to the
#     next pulse). See the action-space block below.
# ===========================================================================
DECISION_DT_MS = 1.5     # baseline chunk size (bio ms) for non-agent conditions


# ===========================================================================
# Action space (AGENT)
# ---------------------------------------------------------------------------
# The agent controls a charge-balanced biphasic pulse and when the NEXT pulse
# fires. Four controllable parameters (toggle + ranges here):
#   amplitude          +A then -A          (net injected charge = 0)
#   pulse_period_ms    this pulse -> next pulse (= 1000 / frequency). Its range is
#                      DERIVED from the [MIN_FREQ_HZ, MAX_FREQ_HZ] band below, so
#                      the max-rate cap is structural -> its low/high are IGNORED.
#   phase_width_ms     duration of EACH phase
#   interphase_gap_ms  gap between the + and - phase WITHIN a single pulse
#
# ACTION_MODE:
#   "continuous" -> Box([-1, 1]^k); env rescales each enabled dim to its range.
#   "discrete"   -> MultiDiscrete([n_bins, ...]); env maps each bin -> a value.
# Only ENABLED params become action dimensions; disabled ones are held FIXED at
# their "default" value (see below). Set any param's "enabled" to False to fix
# it; flip ACTION_MODE to switch the whole space between continuous and discrete.
#
# FIXED (preset) VALUES WHEN DISABLED
# -----------------------------------
# The "default" field of each param IS its fixed/preset value: whenever
# "enabled" is False, the env holds that param constant at "default" every step
# (it never enters the action vector). So to fix a param, set enabled=False and
# set "default" to the constant value you want. Default config: the agent
# controls AMPLITUDE + PULSE_PERIOD (i.e. amplitude + when-to-fire / frequency),
# with pulse SHAPE (phase width, interphase gap) held at clinical constants.
# Flip either shape param's "enabled" to True to hand it to the agent too.
# ===========================================================================
ACTION_MODE = "continuous"   # "continuous" | "discrete"

# Hard frequency band of the pulse train (you set these). pulse_period_ms is
# bounded to [1000 / MAX_FREQ_HZ, 1000 / MIN_FREQ_HZ]; the agent can never pick a
# period shorter than 1000/MAX_FREQ_HZ, so MAX_FREQ_HZ is a structural safety cap.
MAX_FREQ_HZ = 50        # max instantaneous pulse rate (the safety cap)
MIN_FREQ_HZ = 5          # min rate (longest gap); lower => allows longer silences
INTERVAL_MAP = "frequency"   # "frequency" (even Hz coverage) | "period" (even ms)

ACTION_PARAMS = ["amplitude", "pulse_period_ms", "phase_width_ms", "interphase_gap_ms"]
# "default" = the FIXED value used whenever "enabled" is False.
ACTION_SPACE = {
    "amplitude":         {"enabled": True,  "low": 0.0,  "high": 200.0, "default": 100.0, "n_bins": 11},
    "pulse_period_ms":   {"enabled": True,  "low": None, "high": None,  "default": 25.0,  "n_bins": 16},
    "phase_width_ms":    {"enabled": False, "low": 0.1,  "high": 0.5,   "default": 0.2,   "n_bins": 5},
    "interphase_gap_ms": {"enabled": False, "low": 0.0,  "high": 2.0,   "default": 1.0,   "n_bins": 5},
}


# ===========================================================================
# Episode timing (seconds).
# Warmup runs with NO stimulation for ALL conditions (including 'dbs'): the
# network settles and the rolling metric buffer fills; THEN the condition's
# stimulation engages. Warmup steps are discarded from training/eval.
# ===========================================================================

WARMUP_S = 0.5
CONTROL_S = 1.0
METRIC_WINDOW_S = 0.25   # rolling window (s) for obs/reward. Tunable: shorter =>
                         # more responsive control + noisier metrics (the window
                         # length is also the FFT segment, so it sets frequency
                         # resolution: 0.25 s @10 kHz => ~4 Hz bins; 1.0 s => ~1 Hz).

# Recompute the expensive window metrics (synchrony/entropy/beta) only every N
# decisions and cache them; the per-pulse penalty is still applied every step.
# The 1 s window changes <1% per 5 ms step, so caching is ~N x faster with
# negligible signal loss. Set to 1 for exact per-step metrics.
METRIC_RECOMPUTE_EVERY = 1


# ===========================================================================
# Observation (STN only) + reward
# ===========================================================================
# Frequency settings for the window metrics (Hz).
BETA_BAND = (10.0, 35)     # beta-band power range (matches original Analysis full beta)
ENTROPY_FMAX = 35            # upper frequency bound for spectral entropy

OBS_METRICS = ["synchrony", "beta_power", "entropy"]
# (min, max) used to scale each metric into ~[0, 1] for the policy network.
# beta_power comes back in dB (Analysis.power_beta); ranges to be tuned after
# the first PD/Normal/DBS calibration runs.
# NOTE: this fixed min/max scaling is used ONLY when USE_VECNORMALIZE is False.
# When USE_VECNORMALIZE is True the env emits RAW metric values and VecNormalize
# (running mean/std, robust to drift) does the scaling instead — see below.
OBS_NORM = {
    "synchrony":  (0.0, 1.0),
    "beta_power": (40.0, 120.0),   # provisional (observed PD~95, DBS~82 dB); recalibrate
    "entropy":    (0.0, 1.0),
}

# ---------------------------------------------------------------------------
# Observation normalization strategy
# ---------------------------------------------------------------------------
# USE_VECNORMALIZE True  -> the env returns RAW metrics + raw last-action, the
#                           observation space is unbounded (Box(-inf, inf)), and
#                           SB3 VecNormalize whitens obs with a running mean/std
#                           (stats saved next to the model, reloaded at eval).
# USE_VECNORMALIZE False -> legacy behaviour: env pre-scales metrics to ~[0,1]
#                           via OBS_NORM and the obs space is Box(0, 1).
USE_VECNORMALIZE = True
VECNORM = {
    "norm_obs": True,       # whiten observations (recommended)
    "norm_reward": False,   # keep raw reward scale (targets/charge are meaningful)
    "clip_obs": 10.0,       # clip normalized obs to +/- this many std
}

# Monotonic reward for the project objective:
#   - minimise STN synchrony R
#   - maximise STN spectral entropy H
#   - keep stimulation energy/charge low
#
# reward = (w_entropy * H) - (w_sync * R) - (lambda_charge * |charge|)
#          - bad_state_penalty, when R is high and H is low
# beta is logged/observed but not used by default.
REWARD = {
    "mode": "linear",
    "w_sync": 2.0,
    "w_entropy": 1.0,
    "w_beta": 0.0,
    "lambda_charge": 0.005,
    "bad_state": {
        "enabled": True,
        "sync_threshold": 0.5,
        "entropy_threshold": 0.5,
        "penalty": 1.0,
    },
}

# Previous target-bump reward, kept here for reference.
# REWARD = {
#     "shape": "gauss",          # "gauss" | "laplace"
#     # metric -> {enabled, target, scale, w}. Disable any by enabled=False or w=0.
#     "terms": {
#         "synchrony":  {"enabled": True,  "target": 0.25, "scale": 0.15, "w": 2.0},
#         "beta_power": {"enabled": False, "target": 75.0, "scale": 8.0,  "w": 1.0},
#         "entropy":    {"enabled": True,  "target": 0.70, "scale": 0.15, "w": 1.0},
#     },
#     # energy cost: penalty = -w_charge * lambda_charge * |charge|.
#     # w_charge is the on/off toggle (default 1.0); lambda_charge sets the scale so
#     # the term is balanced against the (0, 1] metric bumps (charge ~ 40 / pulse,
#     # so lambda_charge ~ 0.01 => ~0.4 per pulse). Set w_charge=0 to disable.
#     "w_charge": 1.0, "lambda_charge": 0.01,
# }


# ===========================================================================
# RL algorithm selection
# ---------------------------------------------------------------------------
# Pick which Stable-Baselines3 algorithm train.py builds. Each algorithm's
# hyperparameters live in ITS OWN file under rl_stn_gpe/hparams/ (ppo.py, sac.py,
# td3.py) so you can tune each independently without touching the others. Every
# hparams file exports two dicts: HPARAMS (passed to the SB3 constructor) and RUN
# (run-level: n_envs, total_timesteps, seed, eval/checkpoint cadence, device).
#
#   ppo -> on-policy actor-critic (works with continuous OR discrete actions)
#   sac -> off-policy, sample-efficient, CONTINUOUS actions only (supports gSDE)
#   td3 -> off-policy, deterministic policy, CONTINUOUS actions only (+ action noise)
#
# NOTE: SAC and TD3 require ACTION_MODE == "continuous" (train.py asserts this).
# Override at the CLI with `--algo {ppo,sac,td3}`.
# ===========================================================================
ALGO = "sac"   # "ppo" | "sac" | "td3"


# ===========================================================================
# Weights & Biases logging
#   train.py calls wandb.init(project=..., name=..., config=wandb_config(params))
#   and passes WandbCallback to model.learn. The simulation params being run are
#   merged into the logged config (see wandb_config()).
# ===========================================================================
WANDB = {
    "project": "RL DBS",
    "run_name": "Run 1 SAC",
    "entity": None,
    "mode": "online",
    "sync_tensorboard": True,
    "tags": ["sac", "stn-gpe", "dbs"],
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


def period_bounds_ms():
    """[min, max] pulse period (ms) derived from the frequency band.

    min period = 1000 / MAX_FREQ_HZ (fastest allowed rate = the safety cap),
    max period = 1000 / MIN_FREQ_HZ (slowest allowed rate = longest gap).
    """
    return 1000.0 / MAX_FREQ_HZ, 1000.0 / MIN_FREQ_HZ


def param_bounds(name):
    """(low, high) physical range for an action param. pulse_period is DERIVED
    from the frequency band (its YAML low/high are ignored)."""
    if name == "pulse_period_ms":
        return period_bounds_ms()
    spec = ACTION_SPACE[name]
    return spec["low"], spec["high"]


def enabled_params():
    """Controllable params (the action dimensions), in canonical order."""
    return [p for p in ACTION_PARAMS if ACTION_SPACE[p]["enabled"]]


def _max_value(name):
    """Largest value a param can take (its high if enabled, else its default)."""
    if ACTION_SPACE[name]["enabled"]:
        return param_bounds(name)[1]
    return ACTION_SPACE[name]["default"]


def widest_pulse_ms():
    """Longest possible single biphasic pulse: 2*phase_width + interphase_gap."""
    return 2.0 * _max_value("phase_width_ms") + _max_value("interphase_gap_ms")


def validate(dt_ms):
    """Sanity-check action / frequency / pulse settings against the YAML dt."""
    assert ACTION_MODE in ("continuous", "discrete"), f"bad ACTION_MODE: {ACTION_MODE}"
    assert INTERVAL_MAP in ("frequency", "period"), f"bad INTERVAL_MAP: {INTERVAL_MAP}"
    assert 0 < MIN_FREQ_HZ < MAX_FREQ_HZ, (MIN_FREQ_HZ, MAX_FREQ_HZ)
    assert enabled_params(), "No action params enabled - nothing for the agent to control."
    period_min = 1000.0 / MAX_FREQ_HZ
    assert period_min >= widest_pulse_ms(), (
        f"Min pulse period ({period_min:.3f} ms = 1000/MAX_FREQ_HZ) must be >= the "
        f"widest possible pulse ({widest_pulse_ms():.3f} ms). Lower MAX_FREQ_HZ, or "
        f"narrow phase_width_ms / interphase_gap_ms.")


def wandb_config(sim_params=None, train=None):
    """Assemble the full config dict to log to W&B.

    Combines the RL settings with the simulation params actually being run so a
    run is fully reproducible from its W&B config. `sim_params` is the loaded
    YAML dict; `train` is the effective algo hparams+run dict (passed by
    train.py, since hyperparameters now live in per-algo hparams files).
    """
    cfg = {
        "algo": ALGO,
        "action_mode": ACTION_MODE,
        "max_freq_hz": MAX_FREQ_HZ,
        "min_freq_hz": MIN_FREQ_HZ,
        "interval_map": INTERVAL_MAP,
        "action_space": ACTION_SPACE,
        "decision_dt_ms": DECISION_DT_MS,
        "warmup_s": WARMUP_S,
        "control_s": CONTROL_S,
        "metric_window_s": METRIC_WINDOW_S,
        "obs_metrics": OBS_METRICS,
        "obs_norm": OBS_NORM,
        "use_vecnormalize": USE_VECNORMALIZE,
        "beta_band": BETA_BAND,
        "entropy_fmax": ENTROPY_FMAX,
        "reward": REWARD,
    }
    if train is not None:
        cfg["train"] = train
    if sim_params is not None:
        cfg["sim_params"] = sim_params
    return cfg
