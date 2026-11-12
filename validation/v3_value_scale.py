"""V3: Do Q-values scale the way theory says when we change gamma or rewards?

With +1 per step, the value of surviving L more steps is (1 - gamma^L)/(1 - gamma),
so the maximum Q should plateau near 1/(1-gamma) once episodes are long:
    gamma = 0.90  -> 10      gamma = 0.95 -> 20
    gamma = 0.95, rewards x2 -> 40 (Q is linear in reward)
A bug in gamma handling, reward handling or bootstrapping would break these.

We train short runs (default 350 episodes) and compare the plateau of max Q
(median of the last 50 episodes) to the theoretical ceiling, and also to the
finite-horizon prediction from the observed episode lengths.

Run:  python -m validation.v3_value_scale [--episodes 350]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dqn.utils import save_json
from train_cartpole import build_parser, train

OUT = "results/validation"
CONFIGS = [("γ=0.90", 0.90, 1.0), ("γ=0.95", 0.95, 1.0), ("γ=0.95, reward×2", 0.95, 2.0)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=350)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = {}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#3b6ea5", "#2a9d8f", "#e76f51"]
    for (name, g, scale), c in zip(CONFIGS, colors):
        args = build_parser().parse_args([
            "--episodes", str(a.episodes), "--gamma", str(g), "--reward-scale", str(scale),
            "--out", f"{OUT}/v3_runs/g{g}_r{scale}", "--quiet", "--eval-every", "1000000"])
        _, log, _ = train(args)
        mq = np.array(log["max_q"])
        L = np.array(log["reward"][-50:])
        ceiling = scale / (1 - g)
        finite = float(np.median(scale * (1 - g ** L) / (1 - g)))
        plateau = float(np.median(mq[-50:]))
        res[name] = {"ceiling": ceiling, "finite_horizon_pred": finite, "plateau": plateau,
                     "rel_err_vs_finite": abs(plateau - finite) / ceiling,
                     "max_ever": float(mq.max())}
        ax.plot(mq, color=c, lw=0.8, label=f"{name}: plateau {plateau:.1f}")
        ax.axhline(ceiling, color=c, ls="--", lw=1)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("max Q in episode")
    ax.set_title("Value-scale test: dashed lines are the ceilings 1/(1−γ)·r")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v3_value_scale.png"), dpi=150)
    r = res
    res["ratio_reward_x2"] = r["γ=0.95, reward×2"]["plateau"] / r["γ=0.95"]["plateau"]
    save_json(res, os.path.join(OUT, "v3_value_scale.json"))
    print(res)


if __name__ == "__main__":
    main()
