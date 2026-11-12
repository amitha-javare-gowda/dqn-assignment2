"""V1: Does the DQN code recover KNOWN optimal Q-values?

We run the exact same DQNAgent / ReplayBuffer used for CartPole on two tiny
deterministic MDPs whose Q* has a closed form. If the TD target, terminal
masking, gamma, or target-network code were wrong, Q would not match.

(a) Corridor (termination test). States 0..9 (one-hot), actions {0: left, 1: right}.
    Reaching state 9 gives +1 and TERMINATES. All other rewards are 0. Left at
    state 0 stays at 0.  Q*(s, right) = gamma^(8-s), Q*(s, left) = gamma^(10-s)
    (for s>0), Q*(0,left) = gamma^9.

(b) Survival loop (truncation + reward-linearity test, mirrors CartPole). One non-absorbing state,
    reward +1 every step, episode TRUNCATED after 30 steps (time limit, never
    terminated). Correct handling: Q* = 1 / (1 - gamma) = 20 for gamma 0.95.
    A buggy agent that treats truncation as termination converges to
    Q = 1 / (1 - gamma (1 - 1/T)) = 12.25 instead (every state looks the same,
    and 1 in 30 samples wrongly has its bootstrap cut). With every reward doubled
    the correct agent must reach exactly 40 (Q is linear in reward).  We train both variants.

Run:  python -m validation.v1_analytic_mdps
Pass criteria: corridor max |Q - Q*| < 0.05; survival correct-variant |Q - 20| < 0.5
and buggy variant within 1.5 of its own predicted fixed point 12.25 (shows the
test can detect the bug). The survival test uses MSE: with Huber loss the bug
is largely masked (reported as survival_Q_buggy_huber).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dqn.agent import DQNAgent
from dqn.buffer import ReplayBuffer
from dqn.networks import MLPQNet
from dqn.utils import save_json, set_seed

OUT = "results/validation"
G = 0.95


def onehot(i, n):
    v = np.zeros(n, np.float32)
    v[i] = 1
    return v


def corridor(steps=6000, seed=0):
    set_seed(seed)
    n = 10
    agent = DQNAgent(MLPQNet(n, 2, 64), 2, G, lr=1e-3, target_period=100, seed=seed)
    buf = ReplayBuffer(5000, (n,), seed=seed)
    s = 0
    for t in range(steps):
        a = int(np.random.randint(2))           # uniform behaviour policy: off-policy test
        s2 = min(s + 1, n - 1) if a == 1 else max(s - 1, 0)
        term = s2 == n - 1
        buf.add(onehot(s, n), a, float(term), onehot(s2, n), term)
        s = 0 if term else s2
        if len(buf) >= 64:
            agent.update(buf.sample(64))
    with torch.no_grad():
        Q = agent.q(torch.eye(n)).numpy()[: n - 1]
    Qstar = np.zeros((n - 1, 2))
    for st in range(n - 1):
        Qstar[st, 1] = G ** (8 - st)
        Qstar[st, 0] = G ** 9 if st == 0 else G ** (10 - st)
    return Q, Qstar


def survival(treat_trunc_as_terminal, steps=12000, T=30, seed=0, loss="mse", reward=1.0):
    set_seed(seed)
    agent = DQNAgent(MLPQNet(1, 2, 32), 2, G, lr=1e-3, target_period=50, loss=loss, seed=seed)
    buf = ReplayBuffer(5000, (1,), seed=seed)
    x = np.ones(1, np.float32)
    t_in_ep, curve = 0, []
    for t in range(steps):
        a = int(np.random.randint(2))
        t_in_ep += 1
        trunc = t_in_ep >= T
        buf.add(x, a, reward, x, trunc if treat_trunc_as_terminal else False)
        if trunc:
            t_in_ep = 0
        if len(buf) >= 64:
            agent.update(buf.sample(64))
        if t % 100 == 0:
            curve.append(float(agent.q_values(x).max()))
    # report the average over the last 2,000 steps: single readings fluctuate
    # around the fixed point because batches contain a random number of cut targets
    return float(np.mean(curve[-20:])), curve


def main():
    os.makedirs(OUT, exist_ok=True)
    Q, Qs = corridor()
    err = float(np.abs(Q - Qs).max())
    greedy_ok = bool((Q.argmax(1) == 1).all())
    q_ok, curve_ok = survival(False)
    q_bug, curve_bug = survival(True)
    # Huber behaves like an L1 (median) fit for large errors, so the 1-in-30
    # wrongly-cut targets barely move it: Huber partially HIDES this bug.
    q_bug_huber, _ = survival(True, loss="huber")
    # linearity in reward: doubling every reward must double Q* (40)
    q_r2, _ = survival(False, reward=2.0)
    # all states look identical, so the bug's fixed point solves
    # Q = 1 + gamma * (1 - 1/T) * Q   ->   Q = 1 / (1 - gamma * (1 - 1/T))
    avg_trunc_ret = 1 / (1 - G * (1 - 1 / 30))

    res = {"corridor_max_abs_err": err, "corridor_greedy_all_right": greedy_ok,
           "survival_Q_correct": q_ok, "survival_Q_buggy": q_bug, "survival_Q_buggy_huber": q_bug_huber, "survival_Q_reward_x2": q_r2,
           "survival_theory": 1 / (1 - G), "buggy_theory": float(avg_trunc_ret),
           "pass": err < 0.05 and abs(q_ok - 20) < 0.5 and abs(q_bug - avg_trunc_ret) < 1.5 and abs(q_r2 - 40) < 1.0}
    save_json(res, os.path.join(OUT, "v1_analytic.json"))
    print(res)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    s = np.arange(9)
    ax[0].plot(s, Qs[:, 1], "k-", lw=2, label="Q*(s, right) closed form")
    ax[0].plot(s, Q[:, 1], "o", color="#c0392b", label="DQN Q(s, right)")
    ax[0].plot(s, Qs[:, 0], "k--", lw=2, label="Q*(s, left) closed form")
    ax[0].plot(s, Q[:, 0], "s", color="#3b6ea5", label="DQN Q(s, left)")
    ax[0].set_xlabel("corridor state s")
    ax[0].set_ylabel("Q")
    ax[0].set_title(f"(a) Corridor: learned vs exact Q*, max error {err:.3f}")
    ax[0].legend()
    x = np.arange(len(curve_ok)) * 100
    ax[1].plot(x, curve_ok, color="#2a9d8f", label="truncation bootstraps (our code)")
    ax[1].plot(x, curve_bug, color="#c0392b", label="truncation treated as terminal (bug)")
    ax[1].axhline(20, color="k", ls="--", label="Q* = 1/(1-γ) = 20")
    ax[1].axhline(avg_trunc_ret, color="#c0392b", ls=":", label=f"bug's fixed point ≈ {avg_trunc_ret:.1f}")
    ax[1].set_xlabel("environment steps")
    ax[1].set_ylabel("Q(s)")
    ax[1].set_title("(b) Survival loop with a 30-step time limit")
    ax[1].legend(fontsize=8)
    for a_ in ax:
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v1_analytic.png"), dpi=150)


if __name__ == "__main__":
    main()
