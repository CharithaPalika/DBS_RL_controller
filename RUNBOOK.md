# RL-DBS Controller Runbook

This project trains and evaluates an adaptive DBS controller for the STN-GPe
spiking model. Run commands from the project root:

```bash
cd "/Users/charithapalika/Desktop/Lab projects/DBS_RL_controller"
```

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate dbs_rl
```

Or install into an existing Python environment:

```bash
pip install -r requirements.txt
```

## Core Files

- `rl_stn_gpe/config.py`
  - condition paths: `normal`, `pd`, `dbs`, `rl`
  - action space and pulse bounds
  - episode timing: `WARMUP_S`, `CONTROL_S`, `METRIC_WINDOW_S`
  - reward weights and bad-state penalty: `REWARD`
  - default algorithm: `ALGO`

- `rl_stn_gpe/env.py`
  - Gymnasium environment used by training and evaluation.

- `rl_stn_gpe/reward.py`
  - scalar reward and reward breakdown logging keys.

- `rl_stn_gpe/train.py`
  - training entry point.

- `rl_stn_gpe/eval.py`
  - rollout and comparison plotting entry point.

## Algorithm Selection

Default algorithm is set in:

```text
rl_stn_gpe/config.py
```

```python
ALGO = "sac"   # "ppo" | "sac" | "td3"
```

You can override it at runtime:

```bash
python rl_stn_gpe/train.py --algo sac
python rl_stn_gpe/train.py --algo ppo
python rl_stn_gpe/train.py --algo td3
```

SAC and TD3 require:

```python
ACTION_MODE = "continuous"
```

## Hyperparameter Files

Each algorithm has its own file under:

```text
rl_stn_gpe/hparams/
```

- `rl_stn_gpe/hparams/ppo.py`
- `rl_stn_gpe/hparams/sac.py`
- `rl_stn_gpe/hparams/td3.py`

Each file exports two dictionaries:

- `HPARAMS`
  - passed to the Stable-Baselines3 algorithm constructor.
  - examples: learning rate, batch size, gamma, policy architecture.

- `RUN`
  - run-level settings.
  - examples: `total_timesteps`, `n_envs`, `device`, `seed`,
    `checkpoint_freq`, `eval_freq`, `n_eval_episodes`.

Example:

```python
RUN = {
    "total_timesteps": 100_000,
    "n_envs": 1,
    "device": "cpu",
    "seed": 0,
    "checkpoint_freq": 20_000,
    "eval_freq": 10_000,
    "n_eval_episodes": 3,
}
```

You can override `total_timesteps` without editing files:

```bash
python rl_stn_gpe/train.py --algo sac --timesteps 200000
```

## Episode Timing

Current timing is set in `rl_stn_gpe/config.py`:

```python
WARMUP_S = 0.25
CONTROL_S = 1.0
METRIC_WINDOW_S = 0.25
```

With `dt = 0.1 ms` from the YAML params:

- warmup = `2,500` simulator integration steps
- control episode = `10,000` simulator integration steps
- metric window = `2,500` simulator integration steps

For the agent, one RL environment step is one pulse decision. With:

```python
MIN_FREQ_HZ = 5
MAX_FREQ_HZ = 50
```

the agent takes roughly `5` to `50` decisions per 1-second control episode.

## Reward

The current reward is configured in `rl_stn_gpe/config.py`:

```python
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
```

The scalar reward is:

```text
H - 2.0 * R - 0.005 * charge
```

with an additional `-1.0` penalty when:

```text
R > 0.5 and H < 0.5
```

The best model is still selected by highest mean evaluation reward, so this
penalty is included automatically in the checkpoint decision.

## Smoke Test

Use this first after code changes:

```bash
python rl_stn_gpe/train.py --algo sac --smoke
```

Smoke mode uses a tiny training budget, disables W&B, and skips eval.

## Training

Train with the default algorithm from `config.ALGO`:

```bash
python rl_stn_gpe/train.py --no-wandb
```

Train SAC explicitly:

```bash
python rl_stn_gpe/train.py --algo sac --no-wandb --timesteps 100000
```

Train with W&B logging:

```bash
python rl_stn_gpe/train.py --algo sac --timesteps 100000 --run-name "sac-linear-reward-001"
```

Manual hyperparameter overrides:

```bash
python rl_stn_gpe/train.py \
  --algo sac \
  --no-wandb \
  --timesteps 200000 \
  --lr 0.0003 \
  --batch-size 256 \
  --gamma 0.99 \
  --net-arch 256,256
```

PPO-specific rollout override:

```bash
python rl_stn_gpe/train.py \
  --algo ppo \
  --no-wandb \
  --timesteps 100000 \
  --n-steps 2048
```

## Outputs

Training outputs are written under:

```text
rl_stn_gpe/outputs/
```

Main locations:

- `rl_stn_gpe/outputs/checkpoints/<algo>/<run_name>_seed<seed>/`
  - model checkpoints, top evaluation models, final model, and VecNormalize
    stats for one training run.

- `rl_stn_gpe/outputs/logs/`
  - TensorBoard logs.

Example SAC run folder:

```text
rl_stn_gpe/outputs/checkpoints/sac/sac-linear-reward-001_seed0/
```

Typical files inside a run folder:

- `sac_dbs_20000_steps.zip`
- `sac_dbs_40000_steps.zip`
- `sac_dbs_final.zip`
- `vecnormalize.pkl`

Top evaluation models are saved under:

```text
rl_stn_gpe/outputs/checkpoints/<algo>/<run_name>_seed<seed>/top_models/
```

The trainer keeps the best four evaluated models by mean eval return. The
`top_models.csv` file lists them in rank order. Model names include the eval
reward and training step, for example:

```text
top_reward_pos12p345678_steps_40000.zip
top_reward_pos12p345678_steps_40000_vecnormalize.pkl
top_models.csv
```

If `USE_VECNORMALIZE = True`, training also saves:

```text
rl_stn_gpe/outputs/checkpoints/<algo>/<run_name>_seed<seed>/vecnormalize.pkl
```

Each top model also gets a paired `*_vecnormalize.pkl` snapshot. Keep the
matching normalization file with the model for correct evaluation.

## Evaluation

Evaluate all conditions:

```bash
python rl_stn_gpe/eval.py --algo sac
```

Evaluate with an explicit model:

```bash
python rl_stn_gpe/eval.py \
  --algo sac \
  --model rl_stn_gpe/outputs/checkpoints/sac/sac-linear-reward-001_seed0/top_models/top_reward_pos12p345678_steps_40000.zip
```

Evaluate only baselines:

```bash
python rl_stn_gpe/eval.py --conditions normal pd dbs
```

Evaluate RL plus baselines with a fixed seed:

```bash
python rl_stn_gpe/eval.py \
  --algo sac \
  --conditions normal pd dbs rl \
  --seed 123
```

Evaluation writes:

- `rl_stn_gpe/outputs/eval_results.csv`
- `rl_stn_gpe/outputs/eval_comparison.png`
- `rl_stn_gpe/outputs/eval_normal.png`
- `rl_stn_gpe/outputs/eval_pd.png`
- `rl_stn_gpe/outputs/eval_dbs.png`
- `rl_stn_gpe/outputs/eval_rl.png`
- `rl_stn_gpe/outputs/eval_rl_actions.png`
- `rl_stn_gpe/outputs/eval_stim_spectrum.png`

## Verification Commands

Check the environment API and action-space behavior:

```bash
python rl_stn_gpe/verify_env.py
```

Check that the steppable wrapper matches the original simulator dynamics:

```bash
python rl_stn_gpe/verify_stepper.py
```

## Recommended Current Workflow

For the current monotonic reward and SAC setup:

```bash
conda activate dbs_rl
python rl_stn_gpe/train.py --algo sac --smoke
python rl_stn_gpe/train.py --algo sac --no-wandb --timesteps 100000
python rl_stn_gpe/eval.py --algo sac
```
