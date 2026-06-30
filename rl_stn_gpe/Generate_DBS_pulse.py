"""
Generate_DBS_pulse.py
=====================
Unit-based (millisecond) DBS pulse generation, shared by:
  - the RL agent's aperiodic pulses (one pulse inside a decision window), and
  - the open-loop "standard DBS" train (pulse tiled at a fixed frequency).

Everything is specified in physical units (ms, Hz) and converted to integrator
steps using the model's dt. The pulse is a CLEAN charge-balanced biphasic:
    +amplitude for `phase_width_ms`, an optional `interphase_ms` gap, then
    -amplitude for `phase_width_ms`  ->  net injected charge = 0.

This module is self-contained (numpy only) and does not modify stn_gpe/.
"""

import numpy as np


def ms_to_steps(ms, dt_ms):
    """Convert a duration in ms to an integer number of integrator steps."""
    return int(round(ms / dt_ms))


def biphasic_pulse(amplitude, phase_width_ms, interphase_ms, dt_ms):
    """A single charge-balanced biphasic pulse.

    Returns a 1-D array: [+A]*w, [0]*gap, [-A]*w  (net charge = 0 since the two
    phases have equal width and opposite amplitude).
    """
    w = max(ms_to_steps(phase_width_ms, dt_ms), 1)
    g = ms_to_steps(interphase_ms, dt_ms)
    A = float(amplitude)
    return np.concatenate([np.full(w, +A), np.zeros(g), np.full(w, -A)])


def pulse_in_window(window_steps, amplitude, phase_width_ms, interphase_ms,
                    dt_ms, active=True):
    """A decision-window-length waveform (for the RL agent).

    Length == window_steps. If `active`, a biphasic pulse is placed at the start
    and the rest is zero; if not active, all zeros (a 'skip').
    """
    win = np.zeros(int(window_steps))
    if active:
        p = biphasic_pulse(amplitude, phase_width_ms, interphase_ms, dt_ms)
        L = min(len(p), int(window_steps))
        win[:L] = p[:L]
    return win


def dbs_train(amplitude, phase_width_ms, interphase_ms, freq_hz, n_steps, dt_ms):
    """An open-loop DBS train: one biphasic pulse every (1000/freq_hz) ms.

    Returns a length-`n_steps` waveform. Pulses are tiled from t=0; a pulse that
    would run past the end is clipped.
    """
    n_steps = int(n_steps)
    period = max(ms_to_steps(1000.0 / freq_hz, dt_ms), 1)
    pulse = biphasic_pulse(amplitude, phase_width_ms, interphase_ms, dt_ms)
    L = len(pulse)
    train = np.zeros(n_steps)
    for onset in range(0, n_steps, period):
        end = min(onset + L, n_steps)
        train[onset:end] = pulse[:end - onset]
    return train
