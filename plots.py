"""Training-curve plots required by the assignment.

    python plots.py --log results/cartpole/log.json --game cartpole
    python plots.py --log results/mspacman/log.json --game mspacman

Produces in the log's folder:
    max_q.png     (i)  max Q-value per episode vs training episode
    rewards.png   (ii) episode reward with 100-episode moving average overlay
    actions.png   (MsPacman only, non-standard) action-usage heatmap over training
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dqn.utils import load_json, moving_average

NAMES = {"cartpole": "CartPole-v1", "mspacman": "MsPacman-v5"}
PAC_ACTIONS = ["NOOP", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT"]


def plot_max_q(log, game, out, gamma=None):
    ep = np.arange(1, len(log["max_q"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(ep, log["max_q"], lw=0.6, alpha=0.45, color="#3b6ea5", label="max Q in episode")
    ax.plot(ep, moving_average(log["max_q"], 50), lw=2, color="#1d3557",
            label="50-episode moving average")
    if gamma is not None and game == "cartpole":
        ax.axhline(1 / (1 - gamma), color="#c0392b", ls="--",
                   label=f"theoretical ceiling 1/(1-γ) = {1/(1-gamma):.0f}")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("max_a Q(s, a) over visited states")
    ax.set_title(f"{NAMES[game]}: maximum Q-value vs training episodes")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "max_q.png"), dpi=150)
    plt.close(fig)


def plot_rewards(log, game, out):
    r = np.asarray(log["reward"], dtype=float)
    ep = np.arange(1, len(r) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(ep, r, lw=0.6, alpha=0.45, color="#7f8c8d", label="episode reward")
    ax.plot(ep, moving_average(r, 100), lw=2.2, color="#c0392b",
            label="moving average (last 100 episodes)")
    if game == "cartpole":
        ax.axhline(500, color="black", ls=":", lw=1, label="max (500)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode reward")
    ax.set_title(f"{NAMES[game]}: episode reward during training")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rewards.png"), dpi=150)
    plt.close(fig)


def plot_actions(log, out, bin_size=50):
    A = np.asarray(log["actions"], dtype=float)
    n = len(A) // bin_size
    if n == 0:
        return
    binned = A[: n * bin_size].reshape(n, bin_size, -1).sum(1)
    frac = binned / binned.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(frac.T, aspect="auto", cmap="magma", origin="lower",
                   extent=[0, n * bin_size, -0.5, 8.5])
    ax.set_yticks(range(9))
    ax.set_yticklabels(PAC_ACTIONS)
    ax.set_xlabel("Training episode")
    ax.set_title(f"MsPacman-v5: share of each action per {bin_size}-episode window "
                 "(uniform under random play = 0.11)")
    fig.colorbar(im, ax=ax, label="fraction of steps")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "actions.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--game", choices=["cartpole", "mspacman"], required=True)
    a = p.parse_args()
    log = load_json(a.log)
    out = os.path.dirname(a.log)
    gamma = 0.95 if a.game == "cartpole" else 0.99
    plot_max_q(log, a.game, out, gamma)
    plot_rewards(log, a.game, out)
    if a.game == "mspacman" and "actions" in log:
        plot_actions(log, out)
    print("plots written to", out)
