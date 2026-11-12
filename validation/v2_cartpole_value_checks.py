"""V2: Are the trained CartPole Q-values physically and mathematically sensible?

Uses a trained checkpoint. Four checks:

(a) Q vs true discounted return. CartPole is deterministic given the state, so
    for the greedy policy Q(s_t, a_t) should match the Monte-Carlo return-to-go
    G_t = sum_k gamma^k r_{t+k}. We collect greedy rollouts from normal and from
    perturbed start states (see b) and compare Q to G_t for every step that
    is at least 150 steps before a truncation (so the unobserved tail beyond
    500 contributes < gamma^150 * 20 = 0.01).

(b) Q vs steps-to-failure (non-standard plot). If an episode TERMINATES at T,
    a perfect value function gives Q = (1 - gamma^(T-t)) / (1 - gamma).
    Plotting Q against T - t shows whether the network "sees" a crash coming.

(c) Value ceiling. No Q-value may exceed 1/(1-gamma) = 20 by much; values far
    above 20 would indicate overestimation/divergence.

(c/d) Policy map (non-standard plot). Q(push right) - Q(push left) over pole angle
    and angular velocity (cart centred), and over cart position and velocity
    (pole upright). Physics predicts: push toward the side the pole is falling,
    and with an upright pole, push so the cart returns toward the centre.

Run:  python -m validation.v2_cartpole_value_checks --ckpt results/cartpole/best.pt
"""
import argparse
import os

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dqn.agent import DQNAgent
from dqn.networks import MLPQNet
from dqn.utils import save_json, set_seed

G = 0.95
OUT = "results/validation"


def rollouts(agent, n, eps, seed, perturb=False):
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    eps_data = []
    for k in range(n):
        o, _ = env.reset(seed=seed + k)
        if perturb:
            # start from a random, often unrecoverable, state inside the bounds
            st = rng.uniform([-2.0, -1.5, -0.18, -2.0], [2.0, 1.5, 0.18, 2.0])
            env.unwrapped.state = st
            o = st.astype(np.float32)
        S, A, Q, R = [], [], [], []
        done = False
        while not done:
            a, q = agent.act(o.astype(np.float32), eps)
            S.append(o.copy()); A.append(a); Q.append(q[a])
            o, r, term, trunc, _ = env.step(a)
            R.append(r)
            done = term or trunc
        G_t = np.zeros(len(R))
        acc = 0.0
        for t in reversed(range(len(R))):
            acc = R[t] + G * acc
            G_t[t] = acc
        eps_data.append(dict(S=np.array(S), Q=np.array(Q), G=G_t, terminated=bool(term),
                             T=len(R)))
    return eps_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/cartpole/best.pt")
    p.add_argument("--hidden", type=int, default=256)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    set_seed(0)
    agent = DQNAgent(MLPQNet(4, 2, a.hidden), 2, G, seed=0)
    agent.load(a.ckpt, load_opt=False)

    res = {}
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for name, pert, col in [("greedy, normal starts", False, "#2a9d8f"),
                            ("greedy, perturbed starts", True, "#e76f51")]:
        data = rollouts(agent, 40 if not pert else 400, 0.0,
                        seed=900_000 if not pert else 950_000, perturb=pert)
        q, g = [], []
        for d in data:
            keep = np.arange(d["T"]) if d["terminated"] else np.arange(max(0, d["T"] - 150))
            q.append(d["Q"][keep]); g.append(d["G"][keep])
        q, g = np.concatenate(q), np.concatenate(g)
        res[name] = {"n_states": int(len(q)), "mae_Q_vs_G": float(np.abs(q - g).mean()),
                     "corr": float(np.corrcoef(q, g)[0, 1]) if np.std(g) > 0 else None,
                     "frac_terminated": float(np.mean([d["terminated"] for d in data])),
                     "mean_len": float(np.mean([d["T"] for d in data])),
                     "max_Q": float(max(d["Q"].max() for d in data))}
        q0 = np.array([d["Q"][0] for d in data if d["terminated"]])
        g0 = np.array([d["G"][0] for d in data if d["terminated"]])
        if len(q0):
            res[name]["start_state_mae"] = float(np.abs(q0 - g0).mean())
            res[name]["start_state_corr"] = float(np.corrcoef(q0, g0)[0, 1])
        ax[0].scatter(g, q, s=3, alpha=0.3, color=col, label=f"{name}: MAE {res[name]['mae_Q_vs_G']:.2f}")
        # (b) Q vs steps to failure, only terminated episodes
        for d in data:
            if d["terminated"]:
                k = d["T"] - np.arange(d["T"])
                ax[1].plot(k, d["Q"], color=col, alpha=0.25, lw=0.8)
    kk = np.arange(1, 160)
    ax[0].plot([0, 21], [0, 21], "k--", lw=1)
    ax[0].set_xlabel("Monte-Carlo discounted return-to-go G_t")
    ax[0].set_ylabel("predicted Q(s_t, a_t)")
    ax[0].set_title("(a) Predicted Q vs realised return")
    ax[0].legend(markerscale=4)
    ax[1].plot(kk, (1 - G ** kk) / (1 - G), "k-", lw=2.5, label="theory (1-γ^k)/(1-γ)")
    ax[1].axhline(1 / (1 - G), color="gray", ls=":")
    ax[1].set_xlim(0, 160)
    ax[1].set_xlabel("steps until the episode terminates (k)")
    ax[1].set_ylabel("Q(s_t, a_t)")
    ax[1].set_title("(b) Does the net see the crash coming?")
    ax[1].legend()

    # (d) policy maps
    th = np.linspace(-0.21, 0.21, 121); thd = np.linspace(-2.0, 2.0, 121)
    TH, THD = np.meshgrid(th, thd)
    X1 = np.stack([np.zeros_like(TH), np.zeros_like(TH), TH, THD], -1).reshape(-1, 4)
    x = np.linspace(-2.4, 2.4, 121); xd = np.linspace(-2.0, 2.0, 121)
    XX, XD = np.meshgrid(x, xd)
    X2 = np.stack([XX, XD, np.zeros_like(XX), np.zeros_like(XX)], -1).reshape(-1, 4)
    with torch.no_grad():
        q1 = agent.q(torch.tensor(X1, dtype=torch.float32)).numpy()
        q2 = agent.q(torch.tensor(X2, dtype=torch.float32)).numpy()
    gap1 = (q1[:, 1] - q1[:, 0]).reshape(TH.shape)
    gap2 = (q2[:, 1] - q2[:, 0]).reshape(XX.shape)
    # physics check: where the choice matters (|gap| > 1), push toward the side
    # the pole will be leaning 5 steps (0.1 s) from now: sign(theta + 0.1*theta_dot)
    decisive = np.abs(gap1) > 1.0
    phys = (TH + 0.1 * THD) > 0
    agree1 = float(np.mean((gap1 > 0)[decisive] == phys[decisive]))
    res["decisive_state_fraction"] = float(decisive.mean())
    res["policy_agreement_decisive"] = agree1
    res["ceiling_violation"] = bool(max(v["max_Q"] for k, v in res.items()
                                        if isinstance(v, dict)) > 1 / (1 - G) * 1.1)
    lim = np.abs(gap1).max()
    im = ax[2].imshow(gap1, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                      extent=[th[0], th[-1], thd[0], thd[-1]])
    ax[2].plot(th, -th / 0.1, "k--", lw=1, label="θ + 0.1·θ̇ = 0 (pole upright in 0.1 s)")
    ax[2].set_ylim(thd[0], thd[-1])
    ax[2].set_xlabel("pole angle θ (rad)")
    ax[2].set_ylabel("pole angular velocity θ̇")
    ax[2].set_title(f"(c) Q(right) − Q(left), cart centred\nagreement on decisive states (|gap|>1): {agree1:.1%}")
    ax[2].legend(loc="lower left", fontsize=8)
    fig.colorbar(im, ax=ax[2])
    for a_ in ax[:2]:
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v2_cartpole_values.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4.6))
    lim = np.abs(gap2).max()
    im = ax.imshow(gap2, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=[x[0], x[-1], xd[0], xd[-1]])
    ax.set_xlabel("cart position x")
    ax.set_ylabel("cart velocity ẋ")
    ax.set_title("Q(right) − Q(left), pole upright and still")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v2_cart_drift_map.png"), dpi=150)
    save_json(res, os.path.join(OUT, "v2_cartpole_values.json"))
    print(res)


if __name__ == "__main__":
    main()
