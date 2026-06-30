# SKILLS — RL-DBS Controller Project

Reference of the skills, tools, and libraries needed to build the PPO-based DBS
controller on top of the existing STN-GPe spiking model. Split into (A) Claude
skills available in this session, and (B) technical skills / Python libraries
that the actual code depends on.

---

## A. Claude skills (this environment)

The special skills available here are **output-format helpers**, not needed for
the RL code itself. They become relevant only for the project deliverables
(documentation, reports, comparison tables).

| Skill | When it is used | Maps to deliverable |
|-------|-----------------|---------------------|
| `docx` | Write-up of final state / action / reward definitions; methods + results report | Deliverable #6 (documentation) |
| `pptx` | Slide summary of approach and PD vs std-DBS vs RL results | Optional presentation of results |
| `xlsx` | Metrics comparison table (synchrony, entropy, beta power, stim usage) across conditions | Deliverable #4, #5 |
| `pdf` | Export a final combined report if a PDF is requested | Optional |

**Not needed:** none of the above are required to write or run the RL code.
The controller, environment, training, and plots are all plain Python.
Comparison plots are produced with matplotlib in code, not via a skill.

---

## B. Technical skills / Python libraries (the actual build)

### Already present in the repo (reuse, do not duplicate)
- **NumPy** — core array math; the whole simulator is numpy.
- **SciPy** (`scipy.signal`, `scipy.stats`) — Welch PSD, spectrogram, entropy, Savitzky-Golay filtering. Used inside `analysis.py`.
- **PyYAML** — all model conditions are configured via YAML param files.
- **Matplotlib** — raster plots, spectrograms, comparison figures.
- **tqdm** — progress bars in the simulation loop.

### To be added for the RL work
- **Gymnasium** — defines the RL environment API (`reset`, `step`, observation/action spaces). Required for SB3 compatibility.
- **Stable-Baselines3 (PPO)** — the RL algorithm, per project spec. Pulls in **PyTorch** as a backend.
- **PyTorch** — neural-network backend for PPO (installed automatically with SB3).
- **TensorBoard** *(optional but recommended)* — training curves / logging.

Install (sandbox or local):
```bash
pip install gymnasium "stable-baselines3[extra]" --break-system-packages
```

### Engineering capability needed (not a library)
- **Refactor the monolithic simulator into a steppable form.** `STN_GPe_loop`
  currently precomputes the full DBS waveform and runs the entire fixed-length
  loop in one call. The RL env needs `init_state()` + `step(n_steps, dbs_current)`
  with state carried across control decisions. The neural equations must stay
  byte-for-byte identical (project rule: minimize edits to validated model code).
- **Short-window metric adaptation.** Existing analysis assumes long windows
  (e.g. `welch` with `nperseg=10000`). Per-decision rewards over short windows
  need adapted helpers built on the existing `Analysis` methods.
- **Charge-balanced biphasic pulse generation** on demand (positive then
  negative phase, net zero charge) at the fixed stimulation frequency,
  reusing `dbs.py` logic where possible.

---

## C. Notes / constraints

- **Training location:** the sandbox is suitable for correctness testing and
  short runs only. Full PPO training on a ~5 s-per-episode spiking simulation is
  slow; heavy training is better on a local machine / GPU box.
- **Bottleneck:** the simulator is a pure-Python numpy loop that stores full
  state every timestep. Expect to trim stored state and shorten episodes for
  RL throughput.
- **Separation of concerns (project rule):** keep RL code in its own module;
  do not entangle it with the validated simulator code.
