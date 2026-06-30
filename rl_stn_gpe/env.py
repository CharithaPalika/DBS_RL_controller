"""
env.py
======
Gymnasium environment wrapping the STN-GPe simulator for RL-based DBS control.

Design (EVENT-DRIVEN / semi-MDP for the agent)
----------------------------------------------
- The network integrates at dt (0.1 ms) inside STNGPeStepper. The env never
  changes that.
- reset() runs a WARMUP_S transient-removal phase (default 1 s) with NO
  stimulation. Those steps fill the rolling metric buffer but are NOT returned
  as RL steps and are not part of the episode return.
- After warm-up, the AGENT decides one pulse per env step:
    action -> (amplitude, pulse_period_ms, phase_width_ms, interphase_gap_ms)
  The env emits ONE charge-balanced biphasic pulse at the start of the window
  and advances the simulator by pulse_period_ms (the gap to the next pulse).
  pulse_period_ms is bounded to [1000/MAX_FREQ_HZ, 1000/MIN_FREQ_HZ], so the
  max-rate cap is structural. Decisions therefore span VARIABLE biological time.
- BASELINES (none / openloop) ignore the action and advance in fixed chunks of
  DECISION_DT_MS, exactly as before, so the std-DBS comparison stays faithful.
- Observation = trailing-window metrics [sync, beta, entropy] + the agent's
  previous action (normalized to [0,1]). Reward = time-scaled sync+entropy(+beta)
  tent terms minus a per-pulse charge penalty (see rl_stn_gpe.reward).

Action space (config.ACTION_MODE):
    "continuous" -> Box([-1, 1]^k)               (Gaussian PPO policy)
    "discrete"   -> MultiDiscrete([n_bins, ...]) (categorical PPO policy)
where k = number of ENABLED params in config.ACTION_SPACE.

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
from rl_stn_gpe.Generate_DBS_pulse import biphasic_pulse, dbs_train


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
        self.dec_steps = config.decision_steps(self.dt)                              # baseline chunk
        self.win_steps = config.ms_to_steps(config.METRIC_WINDOW_S * 1000, self.dt)
        self.warmup_steps = config.ms_to_steps(config.WARMUP_S * 1000, self.dt)
        self.control_steps = config.ms_to_steps(config.CONTROL_S * 1000, self.dt)
        self.metric_every = config.METRIC_RECOMPUTE_EVERY

        # --- action space (configurable: which params, continuous vs discrete) ---
        self.mode = config.ACTION_MODE
        self.enabled = config.enabled_params()
        self._n_act = len(self.enabled)
        if self.mode == "continuous":
            self.action_space = spaces.Box(low=-1.0, high=1.0,
                                           shape=(self._n_act,), dtype=np.float32)
        else:  # "discrete"
            nvec = [config.ACTION_SPACE[p]["n_bins"] for p in self.enabled]
            self.action_space = spaces.MultiDiscrete(nvec)

        # --- observation space: metrics + previous action (normalized to [0,1]) ---
        obs_dim = len(config.OBS_METRICS) + self._n_act
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(obs_dim,), dtype=np.float32)

        self._seed = seed

    # ------------------------------------------------------------------
    # Action decoding
    # ------------------------------------------------------------------
    def _u_to_value(self, name, u):
        """Map a normalized scalar u in [0,1] to a param's physical value."""
        if name == "pulse_period_ms":
            if config.INTERVAL_MAP == "frequency":
                f = config.MIN_FREQ_HZ + u * (config.MAX_FREQ_HZ - config.MIN_FREQ_HZ)
                return 1000.0 / f                      # u=0 -> slowest, u=1 -> fastest
            lo, hi = config.period_bounds_ms()
            return lo + u * (hi - lo)
        lo, hi = config.param_bounds(name)
        return lo + u * (hi - lo)

    def _decode(self, action):
        """Decode an action into physical params + the normalized u-vector.

        Returns (vals, u) where vals maps EVERY param name -> value (disabled
        params take their default) and u is the [0,1]^k vector for the enabled
        params (used as 'previous action' in the observation).
        """
        vals = {p: config.ACTION_SPACE[p]["default"] for p in config.ACTION_PARAMS}
        u_list = []
        if self.mode == "continuous":
            a = np.asarray(action, dtype=float).reshape(-1)
            for i, p in enumerate(self.enabled):
                u = float(np.clip((a[i] + 1.0) / 2.0, 0.0, 1.0))
                vals[p] = self._u_to_value(p, u)
                u_list.append(u)
        else:  # discrete
            a = np.asarray(action).reshape(-1).astype(int)
            for i, p in enumerate(self.enabled):
                nb = config.ACTION_SPACE[p]["n_bins"]
                idx = int(np.clip(a[i], 0, nb - 1))
                u = idx / (nb - 1) if nb > 1 else 0.0
                vals[p] = self._u_to_value(p, u)
                u_list.append(u)
        return vals, np.asarray(u_list, dtype=np.float32)

    def _neutral_u(self):
        """Placeholder 'previous action' (0.5) for reset / baseline conditions."""
        return np.full(self._n_act, 0.5, dtype=np.float32)

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
        self.decision = 0
        self.n_pulses = 0
        self.steps_done = 0           # integrator steps advanced in the control phase
        self._since_metric = 0
        self.last_u = self._neutral_u()
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
        obs = rwd.make_observation(self._cacheR, self._cacheBeta, self._cacheH,
                                   self.last_u)
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action):
        nan = float("nan")
        amp = period_ms = pw = ip = nan

        if self.stim_mode == "agent":
            vals, u = self._decode(action)
            amp = vals["amplitude"]; period_ms = vals["pulse_period_ms"]
            pw = vals["phase_width_ms"]; ip = vals["interphase_gap_ms"]
            self.last_u = u
            pulse = biphasic_pulse(amp, pw, ip, self.dt)
            period_steps = max(config.ms_to_steps(period_ms, self.dt), len(pulse))
            wave = np.zeros(period_steps)
            wave[:len(pulse)] = pulse
            advanced = period_steps
            emit = bool(amp != 0)

        elif self.stim_mode == "openloop":
            advanced = min(self.dec_steps, self.control_steps - self.steps_done)
            start = self.warmup_steps + self.steps_done
            wave = self.dbs_full[start:start + advanced]
            emit = bool(np.any(wave != 0))

        else:  # "none"
            advanced = min(self.dec_steps, self.control_steps - self.steps_done)
            wave = np.zeros(advanced)
            emit = False

        # absolute injected charge this step (fair stim-usage metric across modes)
        stim_charge = float(np.sum(np.abs(wave)) * self.dt)
        elapsed_ms = advanced * self.dt

        out = self.stepper.step(advanced, wave)
        self._push(out)
        self._record_step(out, wave)
        if emit:
            self.n_pulses += 1

        # Recompute the expensive window metrics only every metric_every steps;
        # reuse the cached value otherwise.
        self._since_metric += 1
        if self._since_metric >= self.metric_every:
            self._cacheR, self._cacheBeta, self._cacheH = self._metrics()
            self._since_metric = 0
        R, beta, H = self._cacheR, self._cacheBeta, self._cacheH

        obs = rwd.make_observation(R, beta, H, self.last_u)
        reward, breakdown = rwd.compute_reward(R, H, beta, charge=stim_charge,
                                               elapsed_ms=elapsed_ms)

        self.decision += 1
        self.steps_done += advanced
        truncated = self.steps_done >= self.control_steps
        terminated = False
        info = {"R": R, "beta": beta, "H": H, "pulse": int(emit),
                "n_pulses": self.n_pulses, "stim_charge": stim_charge,
                "elapsed_ms": elapsed_ms, "amplitude": amp,
                "pulse_period_ms": period_ms, "phase_width_ms": pw,
                "interphase_gap_ms": ip, **breakdown}
        return obs, reward, terminated, truncated, info
