"""V4: Do the three required DQN components each matter the way theory predicts?

Same CartPole setup, one component removed at a time:
    full           replay buffer + target network + Double-DQN + Huber
    no_target      targets computed with the online net (moving target)
    no_replay      buffer of size = batch: learns only from the latest
                   transitions (correlated, non-stationary data)
    vanilla_mse    plain max target (no Double) + MSE loss

Predictions: no_replay and no_target learn slower / less stably; the vanilla
max target overestimates, so max Q exceeds the 1/(1-gamma) = 20 ceiling more
often. We measure final MA-100 reward, the fraction of episodes whose max Q
exceeds 21, and the episode at which MA-100 first reaches 150.

Run:  python -m validation.v4_ablations [--episodes 500]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dqn.utils import load_json, moving_average, save_json
from train_cartpole import build_parser, train

OUT = "results/validation"
RUNS = {
    "full": [],
    "no_target": ["--no-target"],
    "no_replay": ["--no-replay"],
    "vanilla_mse": ["--no-double", "--loss", "mse"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--replot", action="store_true", help="redraw from saved logs, no training")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = {}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    for name, extra in RUNS.items():
        if a.only and name not in a.only:
            continue
        if a.replot:
            log = load_json(f"{OUT}/v4_runs/{name}/log.json")
        else:
            args = build_parser().parse_args(["--episodes", str(a.episodes), "--quiet",
                                              "--eval-every", "1000000",
                                              "--out", f"{OUT}/v4_runs/{name}", *extra])
            _, log, _ = train(args)
        r, mq = np.array(log["reward"]), np.array(log["max_q"])
        ma = moving_average(r, 100)
        hit = np.flatnonzero(ma >= 150)
        res[name] = {"final_ma100": float(ma[-1]), "best_ma100": float(ma.max()),
                     "first_ep_ma100_150": int(hit[0]) + 1 if len(hit) else None,
                     "frac_maxQ_above_21": float(np.mean(mq > 21)),
                     "max_Q_ever": float(mq.max())}
        ax[0].plot(ma, label=name)
        ax[1].plot(moving_average(mq, 20), label=name)
        print(name, res[name], flush=True)
    ax[0].set_title("Ablations: moving average (100) of episode reward")
    ax[0].set_xlabel("Training episode"); ax[0].set_ylabel("reward")
    ax[1].axhline(20, color="k", ls="--", lw=1, label="ceiling 20")
    ax[1].set_title("Ablations: max Q (20-episode moving average)")
    ax[1].set_xlabel("Training episode"); ax[1].set_ylabel("max Q")
    ax[1].set_yscale("symlog", linthresh=1)
    ax[1].set_ylim(bottom=0)
    for a_ in ax:
        a_.legend(); a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v4_ablations.png"), dpi=150)
    save_json(res, os.path.join(OUT, "v4_ablations.json"))


if __name__ == "__main__":
    main()
