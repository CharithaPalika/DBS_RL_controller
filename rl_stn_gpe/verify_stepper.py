"""
verify_stepper.py
=================
Proves STNGPeStepper reproduces the original STN_GPe_loop *bit-for-bit*.

Strategy
--------
For each test case:
  1. Seed NumPy, run the original STN_GPe_loop -> reference spikes/LFP,
     and grab the exact I_DBS waveform it used.
  2. Re-seed identically, build STNGPeStepper, and advance it in CHUNKS
     (mimicking the RL control cadence) feeding the SAME I_DBS waveform.
  3. Assert spike_stn / spike_gpe are identical and lfp_stn / lfp_gpe match.

Two cases cover the two things that could break equivalence:
  A) noise > 0, DBS off  -> checks the per-timestep RNG draw ORDER.
  B) DBS on (biphasic)   -> checks the DBS injection path (I_DBS[i] -> dbs_cur).

Run:  python rl_stn_gpe/verify_stepper.py
"""

import os
import sys
import argparse
import tempfile

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")  # headless: save figures to file instead of showing
import matplotlib.pyplot as plt
import scipy.signal

# --- make project importable regardless of cwd -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# stngpe.py uses `from tqdm.notebook import tqdm`, which crashes outside Jupyter.
# Patch the notebook tqdm to the plain one BEFORE importing stn_gpe (no edit to stn_gpe).
import tqdm
import tqdm.notebook
tqdm.notebook.tqdm = tqdm.std.tqdm

from stn_gpe import STN_GPe_loop, load_yaml, GenerateDBS, Analysis  # untouched
from rl_stn_gpe.stepper import STNGPeStepper          # our wrapper
from rl_stn_gpe.Generate_DBS_pulse import dbs_train   # ms-based DBS builder
from rl_stn_gpe import metrics as M                   # fast metrics (used by the env)
from rl_stn_gpe import config                          # for the metric frequency bands

PARAMS_DIR = os.path.join(PROJECT_ROOT, "params", "stn_gpe_params")


def _write_temp_yaml(base_yaml, overrides):
    """Load a base yaml, apply overrides, write to a temp file, return path."""
    params = load_yaml(base_yaml)
    params.update(overrides)
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(params, f)
    return path, params


def run_case(name, base_yaml, overrides, seed=0, slot=500):
    print(f"\n=== {name} ===")
    tmp, params = _write_temp_yaml(base_yaml, overrides)
    time = params["time"]

    # --- 1. original ------------------------------------------------------
    np.random.seed(seed)
    ref = STN_GPe_loop(tmp)
    ref_spk_stn = np.array(ref["spike_stn"])
    ref_spk_gpe = np.array(ref["spike_gpe"])
    ref_lfp_stn = np.array(ref["lfp_stn"])
    ref_lfp_gpe = np.array(ref["lfp_gpe"])
    I_DBS = np.asarray(ref["I_DBS"], dtype=float)

    # --- 2. stepper, advanced in chunks ----------------------------------
    np.random.seed(seed)                      # identical RNG state for weights
    stp = STNGPeStepper(tmp, seed=None)        # uses the global seed just set
    spk_stn, spk_gpe, lfp_stn, lfp_gpe = [], [], [], []
    t = 0
    while t < time:
        c = min(slot, time - t)
        out = stp.step(c, I_DBS[t:t + c])
        spk_stn.extend(out["spike_stn"])
        spk_gpe.extend(out["spike_gpe"])
        lfp_stn.extend(out["lfp_stn"])
        lfp_gpe.extend(out["lfp_gpe"])
        t += c
    spk_stn = np.array(spk_stn)
    spk_gpe = np.array(spk_gpe)
    lfp_stn = np.array(lfp_stn)
    lfp_gpe = np.array(lfp_gpe)

    os.remove(tmp)

    # --- 3. compare -------------------------------------------------------
    checks = {
        "spike_stn equal": np.array_equal(ref_spk_stn, spk_stn),
        "spike_gpe equal": np.array_equal(ref_spk_gpe, spk_gpe),
        "lfp_stn equal":   np.array_equal(ref_lfp_stn, lfp_stn),
        "lfp_gpe equal":   np.array_equal(ref_lfp_gpe, lfp_gpe),
    }
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    if not checks["spike_stn equal"]:
        print(f"    spike_stn mismatch count: {np.sum(ref_spk_stn != spk_stn)}")
    if not checks["lfp_stn equal"]:
        print(f"    lfp_stn max abs diff: {np.max(np.abs(ref_lfp_stn - lfp_stn)):.3e}")

    summary = {
        "ref total STN spikes": int(ref_spk_stn.sum()),
        "stepper total STN spikes": int(spk_stn.sum()),
        "DBS nonzero samples": int(np.count_nonzero(I_DBS)),
    }
    print(f"  info: {summary}")
    return all(checks.values())


# ===========================================================================
#  Run-any-params utility: simulate via the stepper, plot, and print metrics
# ===========================================================================

def build_dbs_waveform(args):
    """Build the open-loop DBS current for a params dict, reusing GenerateDBS.

    Mirrors the dispatch in STN_GPe_loop (lines 137-152). Returns zeros when
    DBS is off. Only the pulse shapes used by the bundled param files are
    implemented (biphasic / monophasic); noise variants raise NotImplementedError.
    """
    n_steps = args["time"]
    if not args.get("DBS", False):
        return np.zeros(n_steps)

    dt = args["dt"]
    func = args.get("DBS_func")

    # ms-based clean biphasic train (new, uniform-units path)
    if func == "biphasic_ms":
        return dbs_train(amplitude=args["DBS_amplitude"],
                         phase_width_ms=args["DBS_phase_width_ms"],
                         interphase_ms=args["DBS_interphase_ms"],
                         freq_hz=args["DBS_freq_hz"],
                         n_steps=n_steps, dt_ms=dt)

    # legacy duty/step-based path (original params_std_DBS.yaml), via GenerateDBS
    dbs = GenerateDBS()
    common = dict(sampling_freq=(10 ** 3) / dt,
                  time_sec=args["time"] * dt / (10 ** 3))
    if func == "biphasicDBS":
        return dbs.biphasicDBS(duty=args["DBS_duty"], T=1 / args["DBS_freq"],
                               A1=args["DBS_A1"], A2=args["DBS_A2"],
                               pulseinterval=args["pulseinterval"], **common)
    if func == "monophasicDBS":
        return dbs.monophasicDBS(amplitude=args["DBS_amplitude"],
                                 T=1 / args["DBS_freq"], duty=args["DBS_duty"],
                                 **common)
    raise NotImplementedError(f"DBS_func '{func}' not supported by this utility")


def run_via_stepper(params_path, seed=0, slot=500, time_override=None):
    """Run a full simulation through STNGPeStepper for the given params.

    Returns a results dict with concatenated arrays (spike_stn/gpe, lfp_stn/gpe,
    v_stn/gpe single-neuron traces) plus the args and the DBS waveform used.
    """
    args = dict(load_yaml(params_path))
    if time_override is not None:
        args["time"] = int(time_override)
    time = args["time"]

    # Build the open-loop DBS waveform from a (possibly time-overridden) copy.
    I_DBS = build_dbs_waveform(args)

    np.random.seed(seed)
    stp = STNGPeStepper(params_path, seed=None)

    acc = {"spike_stn": [], "spike_gpe": [], "lfp_stn": [],
           "lfp_gpe": [], "v_stn": [], "v_gpe": []}
    t = 0
    while t < time:
        c = min(slot, time - t)
        out = stp.step(c, I_DBS[t:t + c])
        for k in acc:
            acc[k].extend(out[k])
        t += c

    for k in ("spike_stn", "spike_gpe"):
        acc[k] = np.array(acc[k])
    for k in ("lfp_stn", "lfp_gpe", "v_stn", "v_gpe"):
        acc[k] = np.array(acc[k])
    acc["args"] = args
    acc["I_DBS"] = I_DBS[:time]
    return acc


def compute_metrics(res, t_low, t_high):
    """Synchrony + spectral entropy + beta power for STN and GPe over the window.

    Uses rl_stn_gpe.metrics (the SAME fast functions the env uses), not Analysis.
    """
    spk_s, lfp_s = res["spike_stn"][t_low:t_high], res["lfp_stn"][t_low:t_high]
    spk_g, lfp_g = res["spike_gpe"][t_low:t_high], res["lfp_gpe"][t_low:t_high]
    nperseg = min(10000, t_high - t_low)

    H_stn = M.spectral_entropy(lfp_s, nperseg=nperseg, fmax=config.ENTROPY_FMAX)
    H_gpe = M.spectral_entropy(lfp_g, nperseg=nperseg, fmax=config.ENTROPY_FMAX)
    R_stn, R_gpe = M.synchrony(spk_s), M.synchrony(spk_g)
    b_stn = M.beta_power(lfp_s, f_low=config.BETA_BAND[0], f_high=config.BETA_BAND[1])
    b_gpe = M.beta_power(lfp_g, f_low=config.BETA_BAND[0], f_high=config.BETA_BAND[1])
    return {"entropy": (H_stn, H_gpe), "synchrony": (R_stn, R_gpe),
            "beta": (b_stn, b_gpe)}


def print_metrics(metrics, title):
    H_stn, H_gpe = metrics["entropy"]
    R_stn, R_gpe = metrics["synchrony"]
    b_stn, b_gpe = metrics["beta"]
    print(f"\n--- Metrics ({title}) [via rl_stn_gpe.metrics] ---")
    print(f'|{"Metric":<14}|{"STN":^10}|{"GPe":^10}|')
    print(f'|{"-"*14}|{"-"*10}|{"-"*10}|')
    print(f'|{"Entropy":<14}|{H_stn:^10.3f}|{H_gpe:^10.3f}|')
    print(f'|{"Synchrony":<14}|{R_stn:^10.3f}|{R_gpe:^10.3f}|')
    print(f'|{"Beta power dB":<14}|{b_stn:^10.3f}|{b_gpe:^10.3f}|')


def plot_results(res, t_low, t_high, save_path, title=""):
    """4x2 figure mirroring Run_STN_GPe: voltage, LFP, raster, spectrogram."""
    h = res["args"]["dt"]
    t_chunk = np.linspace(0, (t_high - t_low) * h / 1000, t_high - t_low)
    cS, cG = "#054b7c", "#9f0d03"

    fig, axs = plt.subplots(4, 2, figsize=(9, 8), facecolor="white")
    fig.subplots_adjust(wspace=0.3, hspace=0.5, left=0.08, right=0.97,
                        top=0.93, bottom=0.07)
    if title:
        fig.suptitle(title, fontweight="bold")

    # Row 0: single-neuron voltage (from the stepper's v_stn/v_gpe)
    axs[0, 0].plot(t_chunk, res["v_stn"][t_low:t_high], color=cS, lw=0.6)
    axs[0, 0].set_title("STN single-neuron V"); axs[0, 0].set_ylabel("V (mV)")
    axs[0, 1].plot(t_chunk, res["v_gpe"][t_low:t_high], color=cG, lw=0.6)
    axs[0, 1].set_title("GPe single-neuron V")

    # Row 1: LFP
    axs[1, 0].plot(t_chunk, res["lfp_stn"][t_low:t_high], color=cS, lw=0.6)
    axs[1, 0].set_ylabel("LFP"); axs[1, 0].set_title("STN LFP")
    axs[1, 1].plot(t_chunk, res["lfp_gpe"][t_low:t_high], color=cG, lw=0.6)
    axs[1, 1].set_title("GPe LFP")

    # Row 2: raster
    for ax, key, col in ((axs[2, 0], "spike_stn", cS), (axs[2, 1], "spike_gpe", cG)):
        sp = res[key][t_low:t_high]
        tr = np.linspace(0, sp.shape[0] * h / 1000, sp.shape[0])
        for nrn in range(sp.shape[1]):
            ax.scatter(tr, (nrn + 1) * sp[:, nrn], color=col, s=0.3)
        ax.set_ylim(0.5, sp.shape[1] + 0.5)
        ax.set_ylabel("Neuron")
    axs[2, 0].set_title("STN raster"); axs[2, 1].set_title("GPe raster")

    # Row 3: spectrogram of smoothed LFP over the FULL run.
    # NB: a spectrogram needs signal length >> FFT window. Using the whole run
    # (not just the 1 s metric slice) with an adaptive nperseg + high overlap
    # gives many time columns; a single column appears when nperseg ~= signal.
    for ax, key in ((axs[3, 0], "lfp_stn"), (axs[3, 1], "lfp_gpe")):
        full = res[key]
        L = len(full)
        nperseg = int(min(10000, max(512, L // 6)))   # <=1 s window, >=6 columns
        noverlap = int(nperseg * 0.9)                  # smooth time axis
        sig = scipy.signal.savgol_filter(full, 11, 5)
        f, ts, Sxx = scipy.signal.spectrogram(sig, fs=10000, nperseg=nperseg,
                                               noverlap=noverlap, window="hamming")
        ax.pcolormesh(ts, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
        ax.set_ylim(0, 40); ax.set_ylabel("Freq (Hz)"); ax.set_xlabel("Time (s)")
    axs[3, 0].set_title("STN spectrogram (full run)")
    axs[3, 1].set_title("GPe spectrogram (full run)")

    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"  figure saved -> {save_path}")


def check_metrics(params_path, seed=0, time_override=4000):
    """Confirm rl_stn_gpe.metrics matches stn_gpe.Analysis on a real window."""
    name = os.path.basename(params_path)
    print(f"\n=== CHECK metrics vs Analysis ({name}) ===")
    res = run_via_stepper(params_path, seed=seed, time_override=time_override)
    time = res["spike_stn"].shape[0]
    t_low, t_high = max(0, time - 10000), time
    spk, lfp = res["spike_stn"][t_low:t_high], res["lfp_stn"][t_low:t_high]
    nperseg = min(10000, t_high - t_low)

    A = Analysis(spk)
    pairs = {
        "synchrony":  (A.synchrony()[1],                       M.synchrony(spk)),
        "entropy":    (A.spectral_entropy(lfp, fs=10000, nperseg=nperseg,
                                          fmax=35, normalize=True),
                       M.spectral_entropy(lfp, nperseg=nperseg, fmax=35)),
        "beta_power": (A.power_beta(lfp)[2],                   M.beta_power(lfp)),
    }
    all_ok = True
    for k, (ref, new) in pairs.items():
        ok = bool(np.isclose(ref, new))
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {k:<11} Analysis={ref:.6f}  metrics={new:.6f}")
    print("  " + ("ALL METRICS MATCH" if all_ok else "MISMATCH"))
    return all_ok


def run_and_report(params_path, seed=0, slot=500, time_override=None,
                   save_path=None):
    """End-to-end: simulate via stepper, print metrics, save the figure."""
    name = os.path.basename(params_path)
    print(f"\n=== RUN {name} ===")
    res = run_via_stepper(params_path, seed=seed, slot=slot,
                          time_override=time_override)
    time = res["spike_stn"].shape[0]
    t_low = max(0, time - 10000)   # analyse the last ~1 s (matches notebook)
    t_high = time

    metrics = compute_metrics(res, t_low, t_high)
    print_metrics(metrics, name)
    print(f"  DBS nonzero samples: {int(np.count_nonzero(res['I_DBS']))}")

    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__),
                                 f"run_{os.path.splitext(name)[0]}.png")
    plot_results(res, t_low, t_high, save_path, title=name)
    return res, metrics


# ===========================================================================
def _verify():
    SLOT = 500   # 50 ms @ dt=0.1 ms == one 20 Hz control slot
    TIME = 3000  # short run keeps the double simulation fast

    results = []
    # Case A: noise ON, DBS OFF  -> tests RNG draw order across chunked steps
    results.append(run_case(
        "A: noise=3, DBS off (RNG-order test)",
        os.path.join(PARAMS_DIR, "params_PD.yaml"),
        overrides={"stn_gpe_noise": 3, "time": TIME},
        slot=SLOT,
    ))
    # Case B: DBS ON (biphasic) -> tests the DBS injection path
    results.append(run_case(
        "B: std biphasic DBS on (injection test)",
        os.path.join(PARAMS_DIR, "params_std_DBS.yaml"),
        overrides={"time": TIME},
        slot=SLOT,
    ))

    print("\n" + "=" * 50)
    print("ALL CASES PASSED" if all(results) else "SOME CASES FAILED")
    return all(results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Verify and/or run the STN-GPe stepper.")
    p.add_argument("--run", metavar="PARAMS",
                   help="Path to a params YAML to simulate, plot, and report. "
                        "Accepts a full path or just a name like 'params_Normal.yaml'.")
    p.add_argument("--time", type=int, default=None,
                   help="Override the number of integration steps (e.g. 50000).")
    p.add_argument("--save", default=None, help="Output PNG path for the figure.")
    p.add_argument("--check-metrics", metavar="PARAMS", dest="check_metrics",
                   help="Confirm rl_stn_gpe.metrics matches stn_gpe.Analysis.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    def _resolve(name):
        return name if os.path.exists(name) else os.path.join(PARAMS_DIR, name)

    if args.check_metrics:
        ok = check_metrics(_resolve(args.check_metrics), seed=args.seed,
                           time_override=args.time or 4000)
        sys.exit(0 if ok else 1)

    if args.run:
        run_and_report(_resolve(args.run), seed=args.seed,
                       time_override=args.time, save_path=args.save)
        sys.exit(0)

    sys.exit(0 if _verify() else 1)
