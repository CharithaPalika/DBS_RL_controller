"""
metrics.py
==========
Fast, self-contained network metrics for the RL environment: synchrony,
spectral entropy, and beta-band power. These mirror stn_gpe.Analysis but are
optimised for the inner RL loop (the env calls them on every decision).

`synchrony` in particular is a vectorised reimplementation of
Analysis.synchrony(): the original has a per-timestep Python loop over each
inter-spike interval, which is far too slow to call thousands of times per
episode. Here that inner loop is replaced by a single np.arange fill, giving
identical output (validated in verify_metrics) at a fraction of the cost.

Inputs follow the Analysis convention:
    spike_window : (T, N) array of 0/1   (time, neurons)
    lfp_window   : (T,)  array           (LFP samples)

stn_gpe/ is not modified; this is a parallel, faster implementation.
"""

import numpy as np
from scipy.signal import welch, savgol_filter
from scipy.stats import entropy as _shannon_entropy
from numpy.fft import fft


def synchrony(spike_window):
    """Kuramoto-style synchrony order parameter R (mean |Rvalue|).

    Vectorised equivalent of Analysis.synchrony()[1].
    """
    sa = np.asarray(spike_window).T          # -> (N, T), as in Analysis
    N, T = sa.shape
    phi = 3000.0 * np.ones((N, T))
    for n in range(N):
        st = np.where(sa[n] == 1)[0]
        # intervals between consecutive spikes, starting at the 2nd spike
        # (matches Analysis: j runs 1..len-2, fill [st[j], st[j+1]-1))
        for j in range(1, len(st) - 1):
            start, end = st[j], st[j + 1]
            idx = np.arange(start, end - 1)
            if idx.size:
                phi[n, idx] = 2 * np.pi * (idx - start) / (end - start)

    tempM = np.mean(phi, axis=0)
    a = np.sqrt(-1 + 0j)                      # imaginary unit (as in Analysis)
    M = np.exp(a * tempM)
    Rvalue = (np.sum(np.exp(a * phi), axis=0) / N) / M
    return float(np.mean(np.abs(Rvalue)))


def spectral_entropy(signal, fs=10000, nperseg=None, fmax=35, normalize=True):
    """Normalised spectral entropy of an LFP segment (matches Analysis)."""
    signal = np.asarray(signal)
    if nperseg is None:
        nperseg = min(10000, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    mask = (freqs >= 0) & (freqs <= fmax)
    psd = psd[mask]
    psd = psd / np.sum(psd)
    psd = np.where(psd == 0, 1e-12, psd)
    se = _shannon_entropy(psd)
    if normalize:
        se /= np.log(len(psd))
    return float(se)


def beta_power(signal, fs=10000, f_low=12.5, f_high=30.5):
    """Average beta-band power in dB (matches Analysis.power_beta full-beta)."""
    signal = np.asarray(signal)
    sm = savgol_filter(signal, window_length=11, polyorder=5)
    fo = fft(sm)
    Nn = len(fo)
    freq = np.arange(Nn) / (Nn / fs)
    power = np.abs(fo) ** 2
    idx = np.where((freq >= f_low) & (freq < f_high))
    return float(10 * np.log10(np.mean(power[idx]) + 1e-12))


def _rescale(rate, target_mean=1.0, max_std=0.335, clip_min=None, clip_max=None):
    """Port of Analysis.rescale: cap per-bin std and shift to target mean."""
    rate_mean = rate.mean(axis=0, keepdims=True)
    rate_std = rate.std(axis=0, keepdims=True)
    scale = np.minimum(1.0, max_std / (rate_std + 1e-12))
    rescaled = (rate - rate_mean) * scale + rate_mean
    rescaled = rescaled + (target_mean - rescaled.mean(axis=0, keepdims=True))
    if clip_min is not None or clip_max is not None:
        rescaled = np.clip(rescaled, a_min=clip_min, a_max=clip_max)
    return rescaled


def spike_rate(spike_window, binsize=100, scaling_factor=8):
    """Spike-to-rate conversion + cross-quadrant std (port of Analysis.spike_rate).

    Vectorised sliding-window rate (the original used a per-timestep Python loop).
    Returns a dict with raw/processed/rescaled rates and the mean/min/max of the
    cross-quadrant std (a desynchronisation measure: higher std = the four STN
    quadrants are more out of step).

    Works for any window length (not just 1 s): the time axis is grouped into
    ~10 ms bins (SAMPLES_PER_BIN samples @10 kHz), with the trailing remainder
    trimmed, and the edge trim for normalisation scales down for short windows.
    """
    SAMPLES_PER_BIN = 100                               # 10 ms @ 10 kHz
    spk = np.asarray(spike_window, dtype=float)        # (T, N)
    time, N = spk.shape
    grid = int(np.sqrt(N))

    # forward sliding sum, truncated at the end (== sum(spikes[i:i+binsize]))
    cs = np.vstack([np.zeros((1, N)), np.cumsum(spk, axis=0)])  # (T+1, N)
    end = np.minimum(np.arange(time) + binsize, time)
    rate_coded = (cs[end] - cs[:time]).T                # (N, T)
    rate_coded = rate_coded.reshape(grid, grid, time)

    half = grid // 2
    q = lambda r0, r1, c0, c1: np.mean(
        rate_coded[r0:r1, c0:c1, :].reshape(half * half, -1), axis=0)
    rate_abcd = [q(0, half, 0, half), q(half, grid, 0, half),
                 q(0, half, half, grid), q(half, grid, half, grid)]
    # edge trim for the min-normalisation: 100 ms (1000 samples) on a long window,
    # scaled down so short windows keep a non-empty interior.
    edge = min(1000, max(1, time // 5))
    rate_abcd = np.array(
        [(i - np.min(i[edge:time - edge])) /
         (np.max(i) - np.min(i[edge:time - edge])) for i in rate_abcd]) * scaling_factor
    # group the time axis into SAMPLES_PER_BIN-sample bins; trim the remainder.
    n_bins = max(1, time // SAMPLES_PER_BIN)
    usable = n_bins * SAMPLES_PER_BIN
    rate_processed = np.mean(
        rate_abcd[:, :usable].reshape(4, n_bins, SAMPLES_PER_BIN), axis=2)
    rate_rescaled = _rescale(rate_processed)
    std_per_bin = np.std(rate_rescaled, axis=0)

    return {
        "raw_rate_data": rate_coded,
        "preprocessed_rate_data": rate_abcd,
        "processed_rate_data": rate_processed,
        "rescaled_rate_data": rate_rescaled,
        "mean_std": float(np.mean(std_per_bin)),
        "min_std": float(np.min(std_per_bin)),
        "max_std": float(np.max(std_per_bin)),
    }


def compute_metrics(spike_window, lfp_window, fmax=35, beta_band=(12.5, 30.5)):
    """Convenience: returns (synchrony R, beta_power dB, entropy H).

    `fmax` (entropy upper bound) and `beta_band` (f_low, f_high) are passed in so
    callers can drive them from config; defaults match the original Analysis.
    """
    R = synchrony(spike_window)
    H = spectral_entropy(lfp_window, nperseg=len(lfp_window), fmax=fmax)
    beta = beta_power(lfp_window, f_low=beta_band[0], f_high=beta_band[1])
    return R, beta, H
