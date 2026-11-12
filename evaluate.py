"""Roll out a trained network for N episodes and plot the reward histogram.

    python evaluate.py --game cartpole --ckpt results/cartpole/best.pt --episodes 500
    python evaluate.py --game mspacman --ckpt results/mspacman/best.pt --episodes 500

CartPole is evaluated fully greedy (epsilon 0). MsPacman uses epsilon 0.01 by
default (standard Atari evaluation practice; also stops a deterministic policy
from getting stuck in a loop until the 108k-frame cap). Seeds for evaluation
never overlap the training seeds.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dqn.agent import DQNAgent
from dqn.utils import pick_device, save_json, set_seed


def rollout_cartpole(ckpt, episodes, eps, seed, device, hidden=256):
    import gymnasium as gym
    from dqn.networks import MLPQNet
    env = gym.make("CartPole-v1")
    agent = DQNAgent(MLPQNet(4, 2, hidden), 2, 0.95, device=device, seed=seed)
    agent.load(ckpt, load_opt=False)
    rets = []
    for k in range(episodes):
        obs, _ = env.reset(seed=seed + k)
        done, ret = False, 0.0
        while not done:
            a, _ = agent.act(obs.astype(np.float32), eps)
            obs, r, term, trunc, _ = env.step(a)
            ret += r
            done = term or trunc
        rets.append(ret)
    return rets


def rollout_mspacman(ckpt, episodes, eps, seed, device, frame_stack=4):
    from dqn.atari import FrameStack, make_env, preprocess_observation
    from dqn.networks import AtariQNet
    env = make_env()
    agent = DQNAgent(AtariQNet(frame_stack, 9), 9, 0.99, device=device, seed=seed)
    agent.load(ckpt, load_opt=False)
    stack = FrameStack(frame_stack)
    rets = []
    for k in range(episodes):
        raw, _ = env.reset(seed=seed + k)
        obs = stack.reset(preprocess_observation(raw))
        done, ret = False, 0.0
        while not done:
            a, _ = agent.act(obs, eps)
            raw, r, term, trunc, _ = env.step(a)
            obs = stack.push(preprocess_observation(raw))
            ret += r
            done = term or trunc
        rets.append(ret)
        if (k + 1) % 50 == 0:
            print(f"{k+1}/{episodes} mean so far {np.mean(rets):.1f}", flush=True)
    return rets


def plot_hist(rets, title, path, bins=30):
    rets = np.asarray(rets)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if np.ptp(rets) == 0:
        # all episodes identical (e.g. every CartPole run hits 500): one visible bar
        ax.bar([rets[0]], [len(rets)], width=max(1.0, abs(rets[0]) * 0.02),
               color="#3b6ea5", edgecolor="black")
        ax.set_xlim(rets[0] - 50, rets[0] + 50)
    else:
        ax.hist(rets, bins=bins, color="#3b6ea5", edgecolor="black")
    ax.axvline(rets.mean(), color="#c0392b", ls="--", lw=2,
               label=f"mean = {rets.mean():.1f}, std = {rets.std():.1f}")
    ax.set_xlabel("Episode reward")
    ax.set_ylabel("Number of episodes")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--game", choices=["cartpole", "mspacman"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--eps", type=float, default=None)
    p.add_argument("--seed", type=int, default=777_000_000)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    set_seed(0)
    device = pick_device(a.device)
    out = a.out or os.path.dirname(a.ckpt)
    if a.game == "cartpole":
        eps = 0.0 if a.eps is None else a.eps
        rets = rollout_cartpole(a.ckpt, a.episodes, eps, a.seed, device)
        title = f"CartPole-v1: {a.episodes} evaluation episodes (greedy)"
    else:
        eps = 0.01 if a.eps is None else a.eps
        rets = rollout_mspacman(a.ckpt, a.episodes, eps, a.seed, device)
        title = f"MsPacman-v5: {a.episodes} evaluation episodes (eps={eps})"
    rets = np.asarray(rets, dtype=float)
    stats = {"episodes": a.episodes, "epsilon": eps, "mean": rets.mean(), "std": rets.std(),
             "min": rets.min(), "max": rets.max(), "median": float(np.median(rets)),
             "frac_500": float(np.mean(rets >= 500)) if a.game == "cartpole" else None,
             "returns": rets.tolist()}
    save_json(stats, os.path.join(out, "eval.json"))
    plot_hist(rets, title, os.path.join(out, "eval_hist.png"))
    print(f"mean {rets.mean():.2f}  std {rets.std():.2f}  min {rets.min()}  max {rets.max()}")
