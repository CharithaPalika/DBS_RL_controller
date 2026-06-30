"""
eval.py
=======
Evaluate and compare DBS control strategies on the STN-GPe model:

    normal  - healthy baseline (no stim)
    pd      - parkinsonian, untreated (no stim)
    dbs     - standard open-loop DBS (130 Hz, from params_std_DBS_ms.yaml)
    rl      - the trained PPO controller (aperiodic, <=20 Hz)

For each condition it rolls out an episode (same seed across conditions for a
fair comparison), and reports steady-state network metrics (synchrony, entropy,
beta power) plus stimulation usage (total injected charge, pulse count, rate).
Outputs a printed table, a CSV, and a multi-panel comparison figure.

SB3/torch is imported lazily and only when the 'rl' condition is requested, so
the baselines can be evaluated without those packages installed.

Examples
--------
    python rl_stn_gpe/eval.py                                  # all four conditions
    python rl_stn_gpe/eval.py --conditions pd dbs              # baselines only (no torch)
    python rl_stn_gpe/eval.py --model rl_stn_gpe/outputs/checkpoints/best_model.zip
    python rl_stn_gpe/eval.py --time 20000 --seed 7
"""

import os
import sys
import csv
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# stngpe.py uses notebook tqdm; patch before importing stn_gpe (no edit to it).
import tqdm
import tqdm.notebook
tqdm.notebook.tqdm = tqdm.std.tqdm

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.signal

from rl_stn_gpe import config
from rl_stn_gpe import metrics as M
from rl_stn_gpe.env import STNGPeEnv

COLORS = {"normal": "#2a9d3f", "pd": "#c0392b", "dbs": "#e08214", "rl": "#054b7c"}
DT_DECISION_S = config.DECISION_DT_MS / 1000.0


# ---------------------------------------------------------------------------
def rollout(condition, model, seed, time_override=None, record=False):
    """Run one episode and collect series + steady-state metrics (+history)."""
    env = STNGPeEnv(condition=condition, seed=seed, record=record)
    obs, _ = env.reset(seed=seed)
    if time_override is not None:
        env.max_decisions = int(time_override) // env.dec_steps

    series = {"R": [], "H": [], "beta": [], "pulse": [], "charge": []}
    done = False
    while not done:
        action = 0
        if condition == "rl":
            action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        series["R"].append(info["R"]); series["H"].append(info["H"])
        series["beta"].append(info["beta"]); series["pulse"].append(info["pulse"])
        series["charge"].append(info["stim_charge"])
        done = terminated or truncated

    # steady-state metrics on the final 1 s window (matches the analysis window)
    spk = np.asarray(env.buf_spk)
    lfp = np.asarray(env.buf_lfp)
    R_f, beta_f, H_f = M.compute_metrics(spk, lfp, fmax=config.ENTROPY_FMAX,
                                         beta_band=config.BETA_BAND)
    rate_std = M.spike_rate(spk, binsize=env.params["binsize"])["mean_std"]

    control_s = env.decision * env.dec_steps * env.dt / 1000.0
    # actual pulse count: agent counts exactly; open-loop is freq*time.
    if env.stim_mode == "agent":
        n_pulses = env.n_pulses
    elif env.stim_mode == "openloop":
        n_pulses = int(round(env.params["DBS_freq_hz"] * control_s))
    else:
        n_pulses = 0

    return {
        "condition": condition,
        "R_final": R_f, "H_final": H_f, "beta_final": beta_f, "rate_std": rate_std,
        "total_charge": float(np.sum(series["charge"])),
        "n_pulses": n_pulses,
        "pulse_rate_hz": n_pulses / control_s if control_s > 0 else 0.0,
        "control_s": control_s,
        "series": {k: np.asarray(v) for k, v in series.items()},
        "history": env.get_history() if record else None,
    }


def plot_condition(result, save_path):
    """Per-condition figure: voltage, raster, spectrogram + stim, spanning the
    transient (pre-stim) and stimulated parts, with metrics printed."""
    h = result["history"]
    cond = result["condition"]
    dt = h["dt"]
    fs = 1000.0 / dt                         # 10000 Hz
    onset = h["warmup_steps"]                # stim engages here (transient before)
    T = len(h["lfp_stn"])
    col = COLORS.get(cond, "#555")

    # viewing window: 0.5 s before onset (transient) + up to 2 s after (stim)
    w0 = max(0, onset - int(0.5 * fs))
    w1 = min(T, onset + int(2.0 * fs))
    sl = slice(w0, w1)
    tt = (np.arange(w0, w1) - onset) / fs    # t=0 at stim onset

    fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.06, hspace=0.45)
    metrics_txt = (f"R={result['R_final']:.3f}  H={result['H_final']:.3f}  "
                   f"beta={result['beta_final']:.1f} dB  rate_std={result['rate_std']:.3f}  "
                   f"pulses={result['n_pulses']} ({result['pulse_rate_hz']:.1f} Hz)  "
                   f"charge={result['total_charge']:.0f}")
    fig.suptitle(f"Condition: {cond}\n{metrics_txt}", fontweight="bold", fontsize=11)

    def mark(ax):
        ax.axvline(0.0, color="k", ls="--", lw=1)   # stim onset
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # 1) single-neuron voltage
    axs[0].plot(tt, h["v_stn"][sl], color=col, lw=0.5)
    axs[0].set_ylabel("STN V (mV)"); axs[0].set_title("Single-neuron voltage  (dashed = stim onset)")
    mark(axs[0])

    # 2) STN raster
    spk = h["spike_stn"][sl].astype(np.int16)   # (win, N); int16 avoids uint8 overflow
    for n in range(spk.shape[1]):
        axs[1].scatter(tt, (n + 1) * spk[:, n], color=col, s=0.2)
    axs[1].set_ylim(0.5, spk.shape[1] + 0.5); axs[1].set_ylabel("STN neuron")
    axs[1].set_title("Raster"); mark(axs[1])

    # 3) stimulation signal
    axs[2].plot(tt, h["stim"][sl], color="#444", lw=0.5)
    axs[2].set_ylabel("DBS current"); axs[2].set_title("Stimulation"); mark(axs[2])

    # 4) spectrogram of STN LFP over the FULL run (transient + control)
    sig = scipy.signal.savgol_filter(h["lfp_stn"], 11, 5)
    nperseg = int(min(10000, max(512, len(sig) // 6)))
    f, ts, Sxx = scipy.signal.spectrogram(sig, fs=fs, nperseg=nperseg,
                                          noverlap=int(nperseg * 0.9), window="hamming")
    axs[3].pcolormesh(ts - onset / fs, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
    axs[3].set_ylim(0, 40); axs[3].set_ylabel("Freq (Hz)")
    axs[3].set_xlabel("Time from stim onset (s)")
    axs[3].set_title("STN LFP spectrogram (full run)"); axs[3].axvline(0.0, color="w", ls="--", lw=1)

    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"  per-condition figure -> {save_path}")


# ---------------------------------------------------------------------------
def print_table(results):
    print("\n=== Comparison (steady-state, final 1 s window) ===")
    hdr = ["Condition", "Synchrony", "Entropy", "BetaPwr dB", "RateStd",
           "Pulses", "Rate Hz", "Charge"]
    print("| " + " | ".join(f"{h:^10}" for h in hdr) + " |")
    print("|" + "|".join("-" * 12 for _ in hdr) + "|")
    for r in results:
        row = [r["condition"], f"{r['R_final']:.3f}", f"{r['H_final']:.3f}",
               f"{r['beta_final']:.2f}", f"{r['rate_std']:.3f}", f"{r['n_pulses']}",
               f"{r['pulse_rate_hz']:.1f}", f"{r['total_charge']:.0f}"]
        print("| " + " | ".join(f"{c:^10}" for c in row) + " |")


def save_csv(results, path):
    fields = ["condition", "R_final", "H_final", "beta_final", "rate_std",
              "n_pulses", "pulse_rate_hz", "total_charge", "control_s"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})
    print(f"results CSV -> {path}")


def plot_comparison(results, save_path):
    conds = [r["condition"] for r in results]
    cols = [COLORS.get(c, "#555") for c in conds]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    fig.subplots_adjust(wspace=0.3, hspace=0.4, left=0.07, right=0.97,
                        top=0.92, bottom=0.08)
    fig.suptitle("DBS control comparison", fontweight="bold")

    # Row 0: steady-state bar charts with target lines
    axs[0, 0].bar(conds, [r["R_final"] for r in results], color=cols)
    axs[0, 0].axhline(config.REWARD["target_sync"], ls="--", c="k", lw=1)
    axs[0, 0].set_title("Synchrony (target dashed)"); axs[0, 0].set_ylabel("R")

    axs[0, 1].bar(conds, [r["H_final"] for r in results], color=cols)
    axs[0, 1].axhline(config.REWARD["target_entropy"], ls="--", c="k", lw=1)
    axs[0, 1].set_title("Spectral entropy (target dashed)")

    axs[0, 2].bar(conds, [r["beta_final"] for r in results], color=cols)
    axs[0, 2].set_title("Beta power (dB)")

    # Row 1: time series of R and H, and stim-usage bar
    for r in results:
        t = np.arange(len(r["series"]["R"])) * DT_DECISION_S
        axs[1, 0].plot(t, r["series"]["R"], color=COLORS.get(r["condition"]),
                       label=r["condition"], lw=1)
        axs[1, 1].plot(t, r["series"]["H"], color=COLORS.get(r["condition"]),
                       label=r["condition"], lw=1)
    axs[1, 0].set_title("Synchrony over time"); axs[1, 0].set_xlabel("t (s)")
    axs[1, 0].set_ylabel("R"); axs[1, 0].legend(fontsize=8, frameon=False)
    axs[1, 1].set_title("Entropy over time"); axs[1, 1].set_xlabel("t (s)")

    axs[1, 2].bar(conds, [r["total_charge"] for r in results], color=cols)
    axs[1, 2].set_title("Stim usage (total |charge|)")

    for ax in axs.flat:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"comparison figure -> {save_path}")


def plot_stim_spectrum(results, save_path, fmax_hz=200):
    """Overlay the FFT magnitude spectrum of the stimulation signals used during
    the control phase, for every condition that actually stimulates (dbs / rl).

    Shows the spectral signature of each strategy: standard DBS = sharp peak at
    its fixed rate (+harmonics); RL = aperiodic, energy spread at low frequency.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.15)
    plotted = False
    for r in results:
        h = r["history"]
        if h is None:
            continue
        onset = h["warmup_steps"]
        fs = 1000.0 / h["dt"]
        stim = np.asarray(h["stim"][onset:])          # control-phase stim only
        if stim.size == 0 or not np.any(stim):
            continue                                  # skip no-stim conditions
        n = len(stim)
        mag = np.abs(np.fft.rfft(stim)) / n
        freq = np.fft.rfftfreq(n, d=1.0 / fs)
        m = freq <= fmax_hz
        ax.plot(freq[m], mag[m] + 1e-9, color=COLORS.get(r["condition"]),
                label=r["condition"], lw=1.2)
        plotted = True

    if not plotted:
        plt.close(fig)
        print("  (no stimulating condition -> stim spectrum skipped)")
        return

    # reference lines: RL max rate and (if a dbs condition is present) its rate
    ax.axvline(config.STIM_FREQ_HZ, color="#054b7c", ls=":", lw=1, alpha=0.6)
    if any(r["condition"] == "dbs" and r["history"] is not None for r in results):
        ax.axvline(130, color="#e08214", ls=":", lw=1, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("|FFT| (a.u., log)")
    ax.set_title("Stimulation spectrum (control phase): RL vs standard DBS")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"  stim-spectrum figure -> {save_path}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Compare PD / std-DBS / RL controllers.")
    p.add_argument("--conditions", nargs="+",
                   default=["normal", "pd", "dbs", "rl"])
    p.add_argument("--model", default=None,
                   help="Path to trained PPO .zip (defaults to best/final in CKPT_DIR).")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--time", type=int, default=None, help="Override control steps.")
    p.add_argument("--save", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--no-plots", action="store_true",
                   help="Skip the per-condition figures (comparison figure still saved).")
    args = p.parse_args()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Load the PPO model only if needed.
    model = None
    if "rl" in args.conditions:
        from stable_baselines3 import PPO
        path = args.model
        if path is None:
            for cand in ("best_model.zip", "ppo_dbs_final.zip"):
                c = os.path.join(config.CKPT_DIR, cand)
                if os.path.exists(c):
                    path = c
                    break
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(
                "No trained model found. Pass --model PATH or train first "
                "(or drop 'rl' from --conditions).")
        print(f"loading model: {path}")
        model = PPO.load(path, device="cpu")

    record = not args.no_plots
    results = []
    for cond in args.conditions:
        print(f"rolling out: {cond} ...")
        r = rollout(cond, model, seed=args.seed, time_override=args.time,
                    record=record)
        results.append(r)
        if record:
            plot_condition(r, os.path.join(config.OUTPUT_DIR, f"eval_{cond}.png"))

    print_table(results)
    save_csv(results, args.csv or os.path.join(config.OUTPUT_DIR, "eval_results.csv"))
    plot_comparison(results,
                    args.save or os.path.join(config.OUTPUT_DIR, "eval_comparison.png"))
    if record:
        plot_stim_spectrum(results,
                           os.path.join(config.OUTPUT_DIR, "eval_stim_spectrum.png"))


if __name__ == "__main__":
    main()
