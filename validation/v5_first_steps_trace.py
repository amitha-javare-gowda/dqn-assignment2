"""V5: Step-by-step trace of the first 10 steps (as suggested in the handout).

For one fixed-seed episode we print, at every step: state, Q(left), Q(right),
epsilon, chosen action, reward, terminated flag, and the TD target that this
transition would produce (r + gamma * max_a' Q_target(s', a') * (1 - terminated)).

Checked expectations:
  * Untrained net: Q-values near 0 and nearly equal; TD target ~ 1 (reward 1 plus
    a tiny bootstrap), confirming reward and gamma enter the target correctly.
  * Trained net: Q ~ 20 = 1/(1-gamma); the TD target equals Q(s,a) to within a
    small error (Bellman consistency on-policy), and the greedy action pushes
    toward the side the pole is leaning.
  * One gradient step on a single transition moves Q(s,a) TOWARD its target
    (sign check of the loss/gradient).

Run:  python -m validation.v5_first_steps_trace --ckpt results/cartpole/best.pt
"""
import argparse
import copy
import os

import gymnasium as gym
import numpy as np
import torch

from dqn.agent import DQNAgent
from dqn.networks import MLPQNet
from dqn.utils import save_json, set_seed

G = 0.95


def trace(agent, seed, steps=10):
    env = gym.make("CartPole-v1")
    o, _ = env.reset(seed=seed)
    rows = []
    for t in range(steps):
        a, q = agent.act(o.astype(np.float32), 0.0)
        o2, r, term, trunc, _ = env.step(a)
        tgt = agent.compute_targets(torch.tensor([r], dtype=torch.float32),
                                    torch.tensor(o2[None], dtype=torch.float32),
                                    torch.tensor([float(term)])).item()
        rows.append({"t": t, "x": round(float(o[0]), 3), "x_dot": round(float(o[1]), 3),
                     "theta": round(float(o[2]), 4), "theta_dot": round(float(o[3]), 3),
                     "Q_left": round(float(q[0]), 3), "Q_right": round(float(q[1]), 3),
                     "action": "R" if a == 1 else "L", "reward": r, "terminated": bool(term),
                     "td_target": round(tgt, 3), "td_error": round(tgt - float(q[a]), 3)})
        o = o2
        if term or trunc:
            break
    return rows


def to_md(rows):
    keys = list(rows[0].keys())
    out = "| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n"
    for r in rows:
        out += "| " + " | ".join(str(r[k]) for k in keys) + " |\n"
    return out


def one_step_sign_check(agent):
    a2 = copy.deepcopy(agent)
    a2.opt = torch.optim.SGD(a2.q.parameters(), lr=1e-3)
    s = np.array([[0.0, 0.0, 0.05, 0.0]], np.float32)
    batch = (s, np.array([1]), np.array([1.0], np.float32), s, np.array([0.0], np.float32))
    before = a2.q(torch.tensor(s))[0, 1].item()
    target = a2.compute_targets(torch.tensor([1.0]), torch.tensor(s), torch.tensor([0.0])).item()
    a2.update(batch)
    after = a2.q(torch.tensor(s))[0, 1].item()
    return {"Q_before": before, "target": target, "Q_after": after,
            "moved_toward_target": abs(after - target) < abs(before - target)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/cartpole/best.pt")
    a = p.parse_args()
    set_seed(0)
    untrained = DQNAgent(MLPQNet(4, 2, 256), 2, G, seed=0)
    trained = DQNAgent(MLPQNet(4, 2, 256), 2, G, seed=0)
    trained.load(a.ckpt, load_opt=False)
    res = {"untrained": trace(untrained, 123), "trained": trace(trained, 123),
           "sign_check_untrained": one_step_sign_check(untrained),
           "sign_check_trained": one_step_sign_check(trained)}
    md = ("## Untrained network\n\n" + to_md(res["untrained"]) +
          "\n## Trained network\n\n" + to_md(res["trained"]) +
          f"\n\nSingle-update sign check (untrained): {res['sign_check_untrained']}\n"
          f"\nSingle-update sign check (trained): {res['sign_check_trained']}\n")
    os.makedirs("results/validation", exist_ok=True)
    open("results/validation/v5_first_steps.md", "w").write(md)
    save_json(res, "results/validation/v5_first_steps.json")
    print(md)


if __name__ == "__main__":
    main()
