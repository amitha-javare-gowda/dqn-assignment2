# Assignment 2: Deep Q-Network (CartPole-v1 and MsPacman-v5)

PyTorch DQN with a replay buffer, a target network, Double-DQN targets and Huber loss.
CartPole uses gamma = 0.95; MsPacman uses gamma = 0.99 and the handout's preprocessing.

## Layout

```
dqn/                     shared code
  agent.py               DQNAgent: online + target net, epsilon-greedy, TD update, schedules
  buffer.py              replay buffer (stores terminated separately from truncated)
  networks.py            MLP (CartPole) and Nature-style CNN (MsPacman)
  atari.py               handout preprocessing + 4-frame stack
train_cartpole.py        CartPole training
train_mspacman.py        MsPacman training (GPU), checkpoint + resume
evaluate.py              500-episode rollout, histogram, mean/std
plots.py                 max-Q plot and reward + MA-100 plot (+ action heatmap for MsPacman)
make_report.py           builds report.pdf from everything in results/
validation/              experiments for the "edge cases and tests" section (see below)
scripts/                 setup + GPU run scripts (plain shell and SLURM)
results/                 logs, checkpoints, plots
```

## Setup

```bash
bash scripts/setup_env.sh          # creates .venv and installs requirements
source .venv/bin/activate
```

## CartPole-v1 (CPU is fine, about 10 minutes)

```bash
python train_cartpole.py --episodes 800 --out results/cartpole
python evaluate.py --game cartpole --ckpt results/cartpole/best.pt --episodes 500
python plots.py --log results/cartpole/log.json --game cartpole
```

`best.pt` is the checkpoint with the best periodic 5-episode greedy evaluation (ties go to the later one).

## MsPacman-v5 (needs one GPU)

```bash
# inside tmux/screen on the GPU machine
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mspacman.sh
# or, on a SLURM cluster
sbatch scripts/mspacman.slurm
```

On older GPUs such as the GTX 1080 on deepdish (compute capability 6.1), the default PyTorch
wheel fails with "no kernel image is available". Install the CUDA 12.6 build instead:
`pip install torch --index-url https://download.pytorch.org/whl/cu126`.

This trains 5000 episodes, writes the plots, then rolls out 500 evaluation episodes.
If the job stops (time limit, disconnect), run the same command again: it resumes from
`results/mspacman/last.pt` (the replay buffer is refilled before learning resumes).
Memory: the 100k-transition buffer uses about 5.6 GB of RAM (`--buffer` to change).

## Validation experiments

| Script | What it tests | How it validates |
|---|---|---|
| `python -m validation.v1_analytic_mdps` | TD target, gamma, terminal masking, truncation handling | Learns two tiny MDPs with closed-form Q*. Corridor: max error vs Q*. Survival loop with a time limit: correct code converges to 1/(1-γ)=20, a deliberately buggy variant to its own predicted fixed point 12.25 |
| `python -m validation.v2_cartpole_value_checks` | Are trained Q-values meaningful? | Q vs Monte-Carlo return; Q vs steps-to-crash against (1-γ^k)/(1-γ) from perturbed starts; Q ceiling; policy map vs pole physics |
| `python -m validation.v3_value_scale` | Gamma and reward plumbing | Max Q must plateau near r/(1-γ): 10 for γ=0.9, 20 for γ=0.95, 40 with rewards doubled |
| `python -m validation.v4_ablations` | Each DQN component matters | Removes target net, replay buffer, Double+Huber; compares learning and Q stability |
| `python -m validation.v5_first_steps_trace` | First 10 steps | Table of state, Q, action, reward, TD target; one-update sign check |
| `python -m validation.v6_mspacman_checks` | MsPacman pipeline | Preprocessing, frame stack, lives/terminals, random baseline, CNN overfit test |

Outputs go to `results/validation/` (JSON numbers + PNG figures).

## Report

```bash
python make_report.py      # -> report.pdf (includes MsPacman sections once results exist)
```
