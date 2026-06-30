"""
env.py
======
Gymnasium environment wrapping the STN-GPe simulator for RL-based DBS control.

Design (uniform fixed-DT model)
-------------------------------
- The network integrates at dt (0.1 ms) inside STNGPeStepper. The env never
  changes that.
- One env step == DECISION_DT_MS of biological time (default 5 ms = 50 steps).
- reset() runs a WARMUP_S transient-removal phase (default 1 s) with NO
  stimulation. Those steps fill the rolling metric buffer but are NOT returned
  as RL steps and are not part of the episode return.
- After warm-up, the agent decides every DECISION_DT_MS. Action 1 emits a
  charge-balanced biphasic pulse; after a pulse, the pulse action is MASKED for
  refractory_decisions() steps (= 1/STIM_FREQ_HZ = 50 ms => max rate 20 Hz),
  while the env keeps stepping at DT. Skipped/masked steps inject zeros.
- Observation and reward are computed over the trailing METRIC_WINDOW_S window
  (rolling buffer) using rl_stn_gpe.metrics.

Conditions (config.CONDITIONS):
    "agent"    -> RL controls aperiodic pulses (PD network)   [training]
    "openloop" -> standard DBS train injected, action ignored [baseline]
    "none"     -> no stimulation, action ignored              [normal/pd baseline]

Composition, not inheritance: the env *holds* a STNGPeStepper; the simulator
stays a pure, reusable component. stn_gpe/ is never modified.
"""

from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stn_gpe import load_yaml
from rl_stn_gpe import config
from rl_stn_gpe import metrics as M
from rl_stn_gpe import reward as rwd
from rl_stn_gpe.stepper import STNGPeStepper
from rl_stn_gpe.Generate_DBS_pulse import pulse_in_window, dbs_train


class STNGPeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, condition="rl", params_path=None, seed=0, record=False):
        super().__init__()
        self.record = record           # if True, store full per-step history for plotting
        cfg = config.get_condition(condition)
        self.condition = condition
        self.stim_mode = cfg["stim"]                  # none | openloop | agent
        self.params_path = params_path or cfg["params"]
        self.params = load_yaml(self.params_path)
        self.dt = self.params["dt"]
        config.validate(self.dt)

        # --- timing (in integrator steps) ---
        self.dec_steps = config.decision_steps(self.dt)                       # 50
        self.win_steps = config.ms_to_steps(config.METRIC_WINDOW_S * 1000, self.dt)   # 10000
        self.warmup_steps = config.ms_to_steps(config.WARMUP_S * 1000, self.dt)       # 10000
        self.control_steps = config.ms_to_steps(config.CONTROL_S * 1000, self.dt)     # 50000
        self.refractory_max = config.refractory_decisions()                  # 10
        self.max_decisions = self.control_steps // self.dec_steps            # 1000
        self.metric_every = config.METRIC_RECOMPUTE_EVERY                    # cache window metrics

        # --- spaces ---
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(len(config.OBS_METRICS),), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self._seed = seed

    # ------------------------------------------------------------------
    def _push(self, out):
        self.buf_spk.extend(out["spike_stn"])
        self.buf_lfp.extend(out["lfp_stn"])

    def _record_step(self, out, wave):
        """Append full per-step traces (warmup + control) when record=True."""
        if not self.record:
            return
        h = self.history
        h["spike_stn"].extend(np.asarray(out["spike_stn"], dtype=np.uint8))
        h["spike_gpe"].extend(np.asarray(out["spike_gpe"], dtype=np.uint8))
        h["v_stn"].extend(out["v_stn"])
        h["v_gpe"].extend(out["v_gpe"])
        h["lfp_stn"].extend(out["lfp_stn"])
        h["lfp_gpe"].extend(out["lfp_gpe"])
        h["stim"].extend(np.asarray(wave, dtype=float))

    def get_history(self):
        """Return recorded history as numpy arrays, plus the warmup boundary."""
        arr = {k: np.asarray(v) for k, v in self.history.items()}
        arr["warmup_steps"] = self.warmup_steps
        arr["dt"] = self.dt
        return arr

    def _metrics(self):
        spk = np.asarray(self.buf_spk)
        lfp = np.asarray(self.buf_lfp)
        R = M.synchrony(spk)
        H = M.spectral_entropy(lfp, nperseg=len(lfp), fmax=config.ENTROPY_FMAX)
        beta = M.beta_power(lfp, f_low=config.BETA_BAND[0], f_high=config.BETA_BAND[1])
        return R, beta, H

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Per-episode network seed: use the given seed if provided (deterministic
        # eval), otherwise draw a fresh one from the env RNG (domain randomization
        # across connectivity realizations).
        if seed is not None:
            s = seed
        else:
            s = int(self.np_random.integers(0, 2 ** 31 - 1))
        self.stepper = STNGPeStepper(self.params_path, seed=s)
        self.buf_spk = deque(maxlen=self.win_steps)
        self.buf_lfp = deque(maxlen=self.win_steps)
        self.refractory = 0
        self.decision = 0
        self.n_pulses = 0
        self._since_metric = 0
        if self.record:
            self.history = {k: [] for k in ("spike_stn", "spike_gpe", "v_stn",
                                            "v_gpe", "lfp_stn", "lfp_gpe", "stim")}

        # Pre-build the open-loop DBS train (warm-up portion zeroed: no stim).
        if self.stim_mode == "openloop":
            total = self.warmup_steps + self.control_steps
            self.dbs_full = dbs_train(
                amplitude=self.params["DBS_amplitude"],
                phase_width_ms=self.params["DBS_phase_width_ms"],
                interphase_ms=self.params["DBS_interphase_ms"],
                freq_hz=self.params["DBS_freq_hz"],
                n_steps=total, dt_ms=self.dt)
            self.dbs_full[:self.warmup_steps] = 0.0

        # WARM-UP: advance with zero stim to remove transient + fill buffer.
        t = 0
        while t < self.warmup_steps:
            c = min(self.dec_steps, self.warmup_steps - t)
            zeros = np.zeros(c)
            out = self.stepper.step(c, zeros)
            self._push(out)
            self._record_step(out, zeros)
            t += c

        # Compute and cache the initial window metrics (buffer filled by warm-up).
        self._cacheR, self._cacheBeta, self._cacheH = self._metrics()
        obs = rwd.make_observation(self._cacheR, self._cacheBeta, self._cacheH)
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action):
        emit = False
        if self.stim_mode == "agent":
            if int(action) == 1 and self.refractory == 0:
                wave = pulse_in_window(
                    self.dec_steps, config.PULSE["amplitude"],
                    config.PULSE["phase_width_ms"], config.PULSE["interphase_ms"],
                    self.dt, active=True)
                emit = True
                self.refractory = self.refractory_max
            else:
                wave = np.zeros(self.dec_steps)
        elif self.stim_mode == "openloop":
            start = self.warmup_steps + self.decision * self.dec_steps
            wave = self.dbs_full[start:start + self.dec_steps]
            emit = bool(np.any(wave != 0))
        else:  # "none"
            wave = np.zeros(self.dec_steps)

        # absolute injected charge this step (fair stim-usage metric across modes)
        stim_charge = float(np.sum(np.abs(wave)) * self.dt)

        out = self.stepper.step(self.dec_steps, wave)
        self._push(out)
        self._record_step(out, wave)

        if self.stim_mode == "agent" and self.refractory > 0:
            self.refractory -= 1
        if emit:
            self.n_pulses += 1

        # Recompute the expensive window metrics only every metric_every steps;
        # reuse the cached value otherwise (the per-pulse penalty is exact below).
        self._since_metric += 1
        if self._since_metric >= self.metric_every:
            self._cacheR, self._cacheBeta, self._cacheH = self._metrics()
            self._since_metric = 0
        R, beta, H = self._cacheR, self._cacheBeta, self._cacheH

        obs = rwd.make_observation(R, beta, H)
        reward, breakdown = rwd.compute_reward(R, H, beta, emit)

        self.decision += 1
        truncated = self.decision >= self.max_decisions
        terminated = False
        info = {"R": R, "beta": beta, "H": H, "pulse": int(emit),
                "n_pulses": self.n_pulses, "stim_charge": stim_charge, **breakdown}
        return obs, reward, terminated, truncated, info
