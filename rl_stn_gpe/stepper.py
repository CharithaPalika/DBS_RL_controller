"""
STNGPeStepper
=============
A *steppable* wrapper around the existing `STN_GPe_loop` dynamics
(stn_gpe/stngpe.py).

Why this exists
---------------
`STN_GPe_loop` builds the network, pre-computes the entire DBS waveform, and
runs the full fixed-length integration loop in a single call. An RL controller
needs the opposite: initialise the network once, then advance it a short window
at a time while injecting a DBS current that is decided *online* by the agent.

This class therefore splits that monolithic function into:
    __init__()  -> the setup block (STN_GPe_loop lines 13-133), minus the
                   pre-computed I_DBS.
    step()      -> the per-timestep loop body (STN_GPe_loop lines 199-293),
                   copied VERBATIM, with the single change that `I_DBS[i]` is
                   replaced by the externally supplied `dbs_waveform[k]`.

The neural equations are not modified in any way. stn_gpe/ is imported as a
library and never edited. Bit-for-bit equivalence with the original is checked
by verify_stepper.py.
"""

import numpy as np

# Reuse the validated model code — do NOT duplicate it.
from stn_gpe import GenerateDBS, load_yaml, random_wts_sparse, lfp_dist_matrix


class STNGPeStepper:
    """Stateful, steppable STN-GPe spiking network.

    Parameters
    ----------
    yaml_path : str
        Path to a params YAML (Normal / PD / std-DBS). Only the network and
        DBS-spread parameters are read; the DBS *waveform* is supplied per call
        to :meth:`step`, so DBS_func / DBS_freq etc. in the YAML are ignored.
    seed : int | None
        If given, seeds NumPy's global RNG before weight construction so runs
        are reproducible (and comparable to STN_GPe_loop seeded identically).
    record_stn : tuple(int, int)
        Grid index of the single STN neuron whose voltage trace is returned
        each step (default (15, 15) — matches the Run_STN_GPe notebook plot).
    record_gpe : tuple(int, int)
        Grid index of the single GPe neuron whose voltage trace is returned
        (default (7, 7) — matches the notebook plot).
    """

    def __init__(self, yaml_path, seed=None, record_stn=(15, 15), record_gpe=(7, 7)):
        if seed is not None:
            np.random.seed(seed)

        self.record_stn = record_stn
        self.record_gpe = record_gpe

        args = load_yaml(yaml_path)
        self.args = args

        # ---- params (identical names to STN_GPe_loop) -------------------
        self.I_strd2_gpe = args.get('I_strd2_gpe')
        lat_strength_stn = args.get('lat_strength_stn')
        lat_strength_gpe = args.get('lat_strength_gpe')
        wsg_strength = args.get('wsg_strength')
        wgs_strength = args.get('wgs_strength')
        self.I_gpe_ext = args.get('I_gpe_ext')
        self.I_stn_ext = args.get('I_stn_ext')
        self.dt = args.get('dt')
        self.stn_gpe_noise = args.get('stn_gpe_noise')
        n = args.get('stn_gpe_units')
        self.n = n

        # ---- Izhikevich parameters (verbatim) ---------------------------
        self.a_stn, self.b_stn, self.c_stn, self.d_stn = 0.005, 0.265, -65, 1.5
        self.a_gpe, self.b_gpe, self.c_gpe, self.d_gpe = 0.1, 0.2, -65, 2

        # ---- state matrices (verbatim init) -----------------------------
        self.V_stn = np.zeros((n, n))
        self.V_gpe = np.zeros((n, n))
        self.U_stn = np.zeros((n, n))
        self.U_gpe = np.zeros((n, n))

        self.h_gaba_gpe = np.zeros((n, n))
        self.h_nmda_gpe = np.zeros((n, n))
        self.h_ampa_gpe = np.zeros((n, n))
        self.h_strd2_gpe = np.zeros((n, n))

        self.h_gaba_stn = np.zeros((n, n))
        self.h_nmda_stn = np.zeros((n, n))
        self.h_ampa_stn = np.zeros((n, n))

        self.spk_gpe = np.zeros((n, n))
        self.spk_stn = np.zeros((n, n))
        self.spk_d2 = np.ones((n, n))  # unused downstream, kept for parity

        # ---- decay constants / reversal potentials (verbatim) -----------
        self.tau_gaba, self.tau_nmda, self.tau_ampa = 4, 160, 6
        self.E_gaba, self.E_nmda, self.E_ampa = -60, 0, 0
        self.mg = 1
        self.w_strd2_gpe = 1
        self.vpeak = 30

        # ---- weights (SAME RNG draw order as STN_GPe_loop) --------------
        lat_sparse = args.get('lat_sparse')
        self.w_lat_stn = random_wts_sparse(lat_strength_stn, lat_sparse, (n * n, n, n))
        self.w_lat_gpe = random_wts_sparse(lat_strength_gpe, lat_sparse, (n * n, n, n))

        inter_sparse = args.get('inter_sparse')
        self.wsg = random_wts_sparse(wsg_strength, inter_sparse, (n * n, n, n))
        self.wgs = random_wts_sparse(wgs_strength, inter_sparse, (n * n, n, n))

        # ---- LFP distance matrix (verbatim) -----------------------------
        self.lfp_dist = lfp_dist_matrix(n, 7, 7)

        # ---- DBS spatial spread (Gaussian, verbatim) --------------------
        dbs = GenerateDBS()
        self.dbs_spread = dbs.dbs_gauss_weight(
            n=n,
            c=args.get('center'),
            amplitude=args.get('spread_amplitude'),
            sigma=args.get('sigma'),
        )

        self.t = 0  # global timestep counter

    # ====================================================================
    def step(self, n_steps, dbs_waveform):
        """Advance the network by `n_steps` integration steps.

        Parameters
        ----------
        n_steps : int
            Number of 0.1 ms integration steps to advance.
        dbs_waveform : array-like, shape (n_steps,)
            DBS current injected into STN at each step (replaces I_DBS[i]).
            Pass zeros for no stimulation.

        Returns
        -------
        dict with keys spike_stn, spike_gpe (lists of (n*n,) arrays),
        lfp_stn, lfp_gpe (lists of floats), and v_stn, v_gpe (lists of floats =
        the single recorded-neuron membrane voltage), one entry per step.
        """
        dbs_waveform = np.asarray(dbs_waveform, dtype=float)
        assert dbs_waveform.shape[0] >= n_steps, "dbs_waveform shorter than n_steps"

        n = self.n
        dt = self.dt
        rs, rg = self.record_stn, self.record_gpe

        spike_stn, spike_gpe, lfp_stn, lfp_gpe = [], [], [], []
        v_stn, v_gpe = [], []

        for k in range(n_steps):
            dbs_spread_ = self.dbs_spread
            dbs_cur = dbs_waveform[k]

            # ===== GPe synaptic currents (verbatim) ======================
            dh_gaba_gpe = (-self.h_gaba_gpe + self.spk_gpe) / self.tau_gaba
            self.h_gaba_gpe = self.h_gaba_gpe + dt * dh_gaba_gpe
            I_gabalat_gpe = (np.sum(np.sum((self.w_lat_gpe * self.h_gaba_gpe), axis=1), axis=1).reshape(n, n)) * (self.E_gaba - self.V_gpe)

            B_gpe = 1 / (1 + (self.mg / 3.57) * np.exp(-0.062 * self.V_gpe))
            dh_nmda_gpe = (-self.h_nmda_gpe + self.spk_stn) / self.tau_nmda
            self.h_nmda_gpe = self.h_nmda_gpe + dt * dh_nmda_gpe
            I_nmda_gpe = (np.sum(np.sum((self.wsg * self.h_nmda_gpe), axis=1), axis=1).reshape(n, n)) * (self.E_nmda - self.V_gpe) * B_gpe

            dh_ampa_gpe = (-self.h_ampa_gpe + self.spk_stn) / self.tau_ampa
            self.h_ampa_gpe = self.h_ampa_gpe + dt * dh_ampa_gpe
            I_ampa_gpe = (np.sum(np.sum((self.wsg * self.h_ampa_gpe), axis=1), axis=1).reshape(n, n)) * (self.E_ampa - self.V_gpe)

            cD2 = -1
            I_gpe_t = I_gabalat_gpe + I_nmda_gpe + I_ampa_gpe + self.I_strd2_gpe * cD2

            dV_gpe = (0.04 * self.V_gpe ** 2) + 5 * self.V_gpe - self.U_gpe + 140 + I_gpe_t + self.I_gpe_ext + self.stn_gpe_noise * np.random.randn(n, n)
            dU_gpe = self.a_gpe * ((self.b_gpe * self.V_gpe) - self.U_gpe)
            self.V_gpe = self.V_gpe + dt * dV_gpe
            self.U_gpe = self.U_gpe + dt * dU_gpe

            indx_gpe = np.where(self.V_gpe > self.vpeak)
            self.V_gpe[indx_gpe] = self.c_gpe
            self.U_gpe[indx_gpe] = self.U_gpe[indx_gpe] + self.d_gpe
            self.spk_gpe = np.zeros((n, n))
            self.spk_gpe[indx_gpe] = 1

            v_gpe.append(self.V_gpe[rg])
            spike_gpe.append(self.spk_gpe.reshape(n * n).copy())

            I_syn_lfp_gpe = np.sum(np.multiply(I_gpe_t, self.lfp_dist))
            lfp_gpe.append(I_syn_lfp_gpe)

            # ===== STN synaptic currents (verbatim) ======================
            dh_gaba_stn = (-self.h_gaba_stn + self.spk_gpe) / self.tau_gaba
            self.h_gaba_stn = self.h_gaba_stn + dt * dh_gaba_stn
            I_gaba_stn = (np.sum(np.sum((self.wgs * self.h_gaba_stn), axis=1), axis=1).reshape(n, n)) * (self.E_gaba - self.V_stn)

            dh_nmda_stn = (-self.h_nmda_stn + self.spk_stn) / self.tau_nmda
            self.h_nmda_stn = self.h_nmda_stn + dt * dh_nmda_stn

            B_stn = 1 / (1 + (self.mg / 3.57) * np.exp(-0.062 * self.V_stn))

            temp_nmdalat = self.h_nmda_stn * (self.E_nmda - self.V_stn)
            I_nmdalat_stn = B_stn * (np.sum(np.sum(np.multiply(self.w_lat_stn, temp_nmdalat), axis=1), axis=1)).reshape(n, n)

            dh_ampa_stn = (-self.h_ampa_stn + self.spk_stn) / self.tau_ampa
            self.h_ampa_stn = self.h_ampa_stn + dt * dh_ampa_stn

            temp_ampalat = self.h_ampa_stn * (self.E_ampa - self.V_stn)
            I_ampalat_stn = (np.sum(np.sum(np.multiply(self.w_lat_stn, temp_ampalat), axis=1), axis=1)).reshape(n, n)

            # NOTE: only change vs original — I_DBS[i] -> dbs_cur
            I_stn_t = I_gaba_stn + I_nmdalat_stn + I_ampalat_stn + (dbs_cur * dbs_spread_) + self.stn_gpe_noise * np.random.randn(n, n)

            dV_stn = (0.04 * self.V_stn ** 2) + 5 * self.V_stn - self.U_stn + 140 + I_stn_t + self.I_stn_ext
            dU_stn = self.a_stn * ((self.b_stn * self.V_stn) - self.U_stn)
            self.V_stn = self.V_stn + dt * dV_stn
            self.U_stn = self.U_stn + dt * dU_stn

            indx_stn = np.where(self.V_stn > self.vpeak)
            self.V_stn[indx_stn] = self.c_stn
            self.U_stn[indx_stn] = self.U_stn[indx_stn] + self.d_stn

            self.spk_stn = np.zeros((n, n))
            self.spk_stn[indx_stn] = 1

            v_stn.append(self.V_stn[rs])
            spike_stn.append(self.spk_stn.reshape(n * n).copy())

            I_syn_lfp_stn = np.sum(np.multiply(I_stn_t - (dbs_cur * dbs_spread_), self.lfp_dist))
            lfp_stn.append(I_syn_lfp_stn)

        self.t += n_steps

        return {
            'spike_stn': spike_stn,
            'spike_gpe': spike_gpe,
            'lfp_stn': lfp_stn,
            'lfp_gpe': lfp_gpe,
            'v_stn': v_stn,
            'v_gpe': v_gpe,
        }
