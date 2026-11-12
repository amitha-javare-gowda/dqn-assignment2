"""Train DQN on MsPacman-v5 from pixels (gamma = 0.99). Needs a GPU for real runs.

Example (cluster):
    python train_mspacman.py --episodes 5000 --out results/mspacman --max-hours 23
Resume after a time limit:
    python train_mspacman.py --episodes 5000 --out results/mspacman --resume

Techniques: Huber loss, target network (hard copy), Double-DQN target (on by default),
epsilon schedule, step-decay learning-rate scheduler, reward clipping for the TD
target, life loss treated as terminal for bootstrapping only, 4-frame stacking.

Logged per episode (log.json): raw game score, max/mean Q of visited states,
epsilon, learning rate, loss, env steps, and action counts.
"""
import argparse
import os
import time
import warnings

import numpy as np
import torch

from dqn.agent import DQNAgent, linear_schedule
from dqn.atari import FrameStack, make_env, preprocess_observation
from dqn.buffer import ReplayBuffer
from dqn.networks import AtariQNet
from dqn.utils import load_json, pick_device, save_json, set_seed

warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()`")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-step", type=int, default=400_000, help="halve LR every N updates")
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--buffer", type=int, default=100_000,
                   help="transitions; ~56 KB each (obs+next_obs), so 100k ~ 5.6 GB RAM")
    p.add_argument("--learning-starts", type=int, default=20_000)
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--target-period", type=int, default=2_500, help="in gradient updates")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-mid", type=float, default=0.1)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--eps-decay-steps", type=int, default=1_000_000,
                   help="1.0 -> eps-mid over this many env steps, then -> eps-end over 2x more")
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--no-double", action="store_true")
    p.add_argument("--loss", choices=["huber", "mse"], default="huber")
    p.add_argument("--no-clip-reward", action="store_true")
    p.add_argument("--no-life-terminal", action="store_true")
    p.add_argument("--max-hours", type=float, default=1e9, help="stop cleanly after this long")
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results/mspacman")
    return p


def epsilon_at(args, step):
    if step < args.eps_decay_steps:
        return linear_schedule(args.eps_start, args.eps_mid, args.eps_decay_steps, step)
    return linear_schedule(args.eps_mid, args.eps_end, 2 * args.eps_decay_steps,
                           step - args.eps_decay_steps)


def main(args):
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"device={device}", flush=True)

    env = make_env()
    n_actions = env.action_space.n
    assert n_actions == 9, "MsPacman-v5 should expose 9 actions"
    agent = DQNAgent(AtariQNet(args.frame_stack, n_actions), n_actions, args.gamma,
                     lr=args.lr, device=device, double=not args.no_double, loss=args.loss,
                     target_update="hard", target_period=args.target_period, seed=args.seed)
    sched = torch.optim.lr_scheduler.LambdaLR(
        agent.opt, lambda u: max(0.5 ** (u // args.lr_step), args.lr_min / args.lr))
    obs_shape = (args.frame_stack, 88, 80)
    buf = ReplayBuffer(args.buffer, obs_shape, np.int8, seed=args.seed)
    stack = FrameStack(args.frame_stack)

    log = {k: [] for k in ["reward", "max_q", "mean_q", "eps", "lr", "loss", "steps",
                           "length", "actions", "seconds"]}
    start_ep, step = 0, 0
    ck_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(ck_path):
        extra = agent.load(ck_path)
        start_ep, step = extra["episode"], extra["step"]
        log = load_json(os.path.join(args.out, "log.json"))
        # fast-forward the LR schedule to the restored number of updates
        sched.last_epoch = agent.n_updates - 1
        sched.step()
        print(f"resumed at episode {start_ep}, step {step}; buffer refills from scratch",
              flush=True)
        # refill the buffer before learning again so updates are not on a tiny buffer
        refill_until = step + args.learning_starts
    else:
        refill_until = args.learning_starts
    save_json(vars(args), os.path.join(args.out, "config.json"))

    t0 = time.time()
    best_ma = -np.inf
    for ep in range(start_ep, args.episodes):
        raw, _ = env.reset(seed=args.seed * 1_000_000 + ep)
        obs = stack.reset(preprocess_observation(raw))
        lives = None
        done, score, qs, losses, ep_len = False, 0.0, [], [], 0
        acts = np.zeros(n_actions, dtype=int)
        while not done:
            eps = epsilon_at(args, step)
            a, q = agent.act(obs, eps)
            qs.append(float(q.max()))
            acts[a] += 1
            raw, r, term, trunc, info = env.step(a)
            nxt = stack.push(preprocess_observation(raw))
            cur_lives = info.get("lives")
            life_lost = lives is not None and cur_lives is not None and cur_lives < lives
            lives = cur_lives
            # bootstrap cut on real game over, and (by default) on losing a life;
            # truncation (ALE frame cap) is never a terminal for the TD target
            learn_term = term or (life_lost and not args.no_life_terminal)
            learn_r = r if args.no_clip_reward else float(np.sign(r))
            buf.add(obs, a, learn_r, nxt, learn_term)
            obs, score, step, ep_len = nxt, score + r, step + 1, ep_len + 1
            done = term or trunc
            if step >= refill_until and len(buf) >= args.batch and step % args.train_freq == 0:
                l, _ = agent.update(buf.sample(args.batch))
                sched.step()
                losses.append(l)

        log["reward"].append(score)
        log["max_q"].append(max(qs))
        log["mean_q"].append(float(np.mean(qs)))
        log["eps"].append(eps)
        log["lr"].append(agent.opt.param_groups[0]["lr"])
        log["loss"].append(float(np.mean(losses)) if losses else None)
        log["steps"].append(step)
        log["length"].append(ep_len)
        log["actions"].append(acts.tolist())
        log["seconds"].append(time.time() - t0)
        ma = float(np.mean(log["reward"][-100:]))
        if (ep + 1) % 10 == 0:
            print(f"ep {ep+1:5d} step {step:8d} score {score:6.0f} ma100 {ma:7.1f} "
                  f"maxQ {log['max_q'][-1]:6.2f} eps {eps:.3f} lr {log['lr'][-1]:.1e} "
                  f"buf {len(buf)} ({(time.time()-t0)/3600:.2f}h)", flush=True)
        if len(log["reward"]) >= 100 and ma > best_ma:
            best_ma = ma
            agent.save(os.path.join(args.out, "best.pt"), {"episode": ep + 1, "ma100": ma})
        hit_time = (time.time() - t0) / 3600 > args.max_hours
        if (ep + 1) % args.ckpt_every == 0 or hit_time or ep + 1 == args.episodes:
            agent.save(ck_path, {"episode": ep + 1, "step": step})
            save_json(log, os.path.join(args.out, "log.json"))
        if hit_time:
            print("time budget reached; rerun with --resume to continue", flush=True)
            break
    agent.save(os.path.join(args.out, "final.pt"), {"episode": len(log["reward"]), "step": step})
    save_json(log, os.path.join(args.out, "log.json"))


if __name__ == "__main__":
    main(build_parser().parse_args())
