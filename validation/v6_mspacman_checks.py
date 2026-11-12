"""V6: MsPacman pipeline checks that run on a CPU in a few minutes.

(a) Preprocessing. Ms. Pac-Man's own colour (sum 448) must exist in raw frames
    and be mapped to the darkest value (-128) after preprocessing; output shape
    (88, 80, 1), dtype int8, range [-128, 127]. We also save a side-by-side image.
(b) Frame stack at reset is 4 copies of the first frame (no stale frames from
    the previous episode) and shifts correctly on push.
(c) Lives / terminal logic: we play randomly and count life losses per game.
    Every game must end in a real termination, and the life-loss count should
    equal the starting lives (an extra life at 10,000 points is possible but
    random play never reaches it). This is what the life-loss-as-terminal
    training option relies on.
(d) Random-policy baseline over 30 episodes: the score a trained agent must
    clearly beat to show learning (this number is drawn on the training curve).
(e) Overfit test: 32 real MsPacman transitions are given random +/-1 rewards;
    with the target net frozen, the CNN + TD update must drive the loss on this
    fixed batch toward 0 (capacity / gradient-flow check on image inputs; random
    rewards make sure the starting loss is large, so the drop is meaningful).

Run:  python -m validation.v6_mspacman_checks
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dqn.agent import DQNAgent
from dqn.atari import FrameStack, make_env, mspacman_color, preprocess_observation
from dqn.buffer import ReplayBuffer
from dqn.networks import AtariQNet
from dqn.utils import save_json, set_seed

OUT = "results/validation"


def main():
    os.makedirs(OUT, exist_ok=True)
    set_seed(0)
    env = make_env()
    res = {"n_actions": int(env.action_space.n)}
    raw, info = env.reset(seed=0)
    for _ in range(60):  # let the game start moving
        raw, *_ = env.step(0)
    crop = raw[1:176:2, ::2].astype(np.int64).sum(2)
    p = preprocess_observation(raw)
    res["pre"] = {"shape": list(p.shape), "dtype": str(p.dtype), "min": int(p.min()),
                  "max": int(p.max()), "mspacman_pixels_in_crop": int((crop == mspacman_color).sum()),
                  "those_pixels_mapped_to": int(np.unique(p[..., 0][crop == mspacman_color])[0])
                  if (crop == mspacman_color).any() else None}
    fig, ax = plt.subplots(1, 2, figsize=(8, 4.5))
    ax[0].imshow(raw); ax[0].set_title("raw 210x160 RGB")
    ax[1].imshow(p[..., 0], cmap="gray"); ax[1].set_title("preprocessed 88x80 (int8)")
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "v6_preprocess.png"), dpi=130)

    fs = FrameStack(4)
    s0 = fs.reset(p)
    s1 = fs.push(-p)
    res["stack"] = {"reset_all_equal": bool(all((s0[i] == s0[0]).all() for i in range(4))),
                    "push_shifts": bool((s1[:3] == s0[1:]).all() and (s1[3] == -p[..., 0]).all())}

    scores, life_events = [], []
    for ep in range(30):
        raw, info = env.reset(seed=10_000 + ep)
        lives, losses, score, done, t = info["lives"], 0, 0.0, False, 0
        while not done:
            raw, r, term, trunc, info = env.step(env.action_space.sample())
            score += r; t += 1
            if info["lives"] < lives:
                losses += 1
            lives = info["lives"]
            done = term or trunc
        scores.append(score)
        life_events.append({"life_losses": losses, "terminated": bool(term), "steps": t})
    res["random_baseline"] = {"mean": float(np.mean(scores)), "std": float(np.std(scores)),
                              "min": float(np.min(scores)), "max": float(np.max(scores))}
    res["lives"] = {"start_lives": int(make_env().reset(seed=0)[1]["lives"]),
                    "life_losses_per_game_mean": float(np.mean([e["life_losses"] for e in life_events])),
                    "all_games_terminated": bool(all(e["terminated"] for e in life_events))}

    # (e) overfit a fixed batch of 32 real transitions
    buf = ReplayBuffer(256, (4, 88, 80), np.int8)
    raw, _ = env.reset(seed=1)
    fs = FrameStack(4); s = fs.reset(preprocess_observation(raw))
    while len(buf) < 32:
        a = env.action_space.sample()
        raw, r, term, trunc, _ = env.step(a)
        s2 = fs.push(preprocess_observation(raw))
        # random +/-1 rewards: the net must memorise 32 arbitrary targets
        buf.add(s, a, float(np.random.choice([-1.0, 1.0])), s2, term)
        s = s2
    batch = buf.sample(32)
    agent = DQNAgent(AtariQNet(4, 9), 9, 0.99, lr=2.5e-4, target_period=10**9, seed=0)
    losses = [agent.update(batch)[0] for _ in range(300)]
    res["overfit"] = {"loss_first": losses[0], "loss_last": losses[-1],
                      "reduction_factor": losses[0] / max(losses[-1], 1e-12)}
    save_json(res, os.path.join(OUT, "v6_mspacman_checks.json"))
    print(res)


if __name__ == "__main__":
    main()
