"""Train DQN on CartPole-v1 with gamma = 0.95.

Example:
    python train_cartpole.py --episodes 600 --out results/cartpole

Writes to --out:
    log.json        per-episode reward, max Q, mean Q, epsilon, loss, steps
    best.pt         checkpoint with the best periodic greedy evaluation
    final.pt        checkpoint at the end of training
    config.json     hyperparameters used
"""
import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch

from dqn.agent import DQNAgent, linear_schedule
from dqn.buffer import ReplayBuffer
from dqn.networks import MLPQNet
from dqn.utils import pick_device, save_json, set_seed


def greedy_eval(agent, n, seed):
    env = gym.make("CartPole-v1")
    rets = []
    for k in range(n):
        obs, _ = env.reset(seed=seed + k)
        done, ret = False, 0.0
        while not done:
            a, _ = agent.act(obs.astype(np.float32), 0.0)
            obs, r, term, trunc, _ = env.step(a)
            ret += r
            done = term or trunc
        rets.append(ret)
    env.close()
    return float(np.mean(rets))


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-final-frac", type=float, default=0.05,
                   help="learning rate decays linearly to lr*frac over --lr-decay-steps")
    p.add_argument("--lr-decay-steps", type=int, default=150_000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--buffer", type=int, default=100_000)
    p.add_argument("--learning-starts", type=int, default=1_000)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--target-update", choices=["hard", "soft"], default="hard")
    p.add_argument("--target-period", type=int, default=500)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--eps-decay-steps", type=int, default=15_000)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--loss", choices=["huber", "mse"], default="huber")
    p.add_argument("--no-double", action="store_true", help="ablation: vanilla max target")
    p.add_argument("--no-target", action="store_true", help="ablation: no target net")
    p.add_argument("--no-replay", action="store_true",
                   help="ablation: buffer of size batch, i.e. learn from the latest transitions only")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--reward-scale", type=float, default=1.0,
                   help="validation: multiply every reward by this constant")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results/cartpole")
    p.add_argument("--quiet", action="store_true")
    return p


def train(args):
    set_seed(args.seed)
    torch.set_num_threads(1)
    os.makedirs(args.out, exist_ok=True)
    save_json(vars(args), os.path.join(args.out, "config.json"))
    device = pick_device(args.device)

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNAgent(MLPQNet(obs_dim, n_actions, args.hidden), n_actions, args.gamma,
                     lr=args.lr, device=device, use_target=not args.no_target,
                     double=not args.no_double, loss=args.loss,
                     target_update=args.target_update, target_period=args.target_period,
                     tau=args.tau, seed=args.seed)
    cap = args.batch if args.no_replay else args.buffer
    buf = ReplayBuffer(cap, (obs_dim,), np.float32, seed=args.seed)
    learning_starts = args.batch if args.no_replay else args.learning_starts

    log = {k: [] for k in ["reward", "max_q", "mean_q", "eps", "loss", "td", "steps",
                           "eval_episode", "eval_return", "terminated", "truncated"]}
    step, best_eval, t0 = 0, -np.inf, time.time()
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed * 100_000 + ep)
        obs = obs.astype(np.float32)
        done, ep_ret, qs, losses, tds = False, 0.0, [], [], []
        while not done:
            eps = linear_schedule(args.eps_start, args.eps_end, args.eps_decay_steps, step)
            a, q = agent.act(obs, eps)
            qs.append(q.max())
            nxt, r, term, trunc, _ = env.step(a)
            nxt = nxt.astype(np.float32)
            # store terminated only; truncation at 500 is NOT a terminal state
            buf.add(obs, a, r * args.reward_scale, nxt, term)
            obs, ep_ret, step = nxt, ep_ret + r, step + 1
            done = term or trunc
            if len(buf) >= learning_starts and step % args.train_freq == 0:
                for g in agent.opt.param_groups:
                    g["lr"] = linear_schedule(args.lr, args.lr * args.lr_final_frac,
                                              args.lr_decay_steps, step)
                l, td = agent.update(buf.sample(args.batch))
                losses.append(l)
                tds.append(td)
        log["reward"].append(ep_ret)
        log["max_q"].append(float(np.max(qs)))
        log["mean_q"].append(float(np.mean(qs)))
        log["eps"].append(eps)
        log["loss"].append(float(np.mean(losses)) if losses else None)
        log["td"].append(float(np.mean(tds)) if tds else None)
        log["steps"].append(step)
        log["terminated"].append(bool(term))
        log["truncated"].append(bool(trunc and not term))

        if (ep + 1) % args.eval_every == 0:
            ev = greedy_eval(agent, args.eval_episodes, seed=10_000_000 + ep)
            log["eval_episode"].append(ep + 1)
            log["eval_return"].append(ev)
            # ties go to the later checkpoint (it has seen more data)
            if ev >= best_eval:
                best_eval = ev
                agent.save(os.path.join(args.out, "best.pt"), {"episode": ep + 1, "eval": ev})
            if not args.quiet:
                print(f"ep {ep+1:4d} step {step:7d} ret {ep_ret:5.0f} "
                      f"ma100 {np.mean(log['reward'][-100:]):6.1f} maxQ {log['max_q'][-1]:6.2f} "
                      f"eps {eps:.3f} eval {ev:5.1f} ({time.time()-t0:.0f}s)", flush=True)
        if (ep + 1) % 50 == 0:
            save_json(log, os.path.join(args.out, "log.json"))
    agent.save(os.path.join(args.out, "final.pt"), {"episode": args.episodes})
    save_json(log, os.path.join(args.out, "log.json"))
    env.close()
    return agent, log, buf


if __name__ == "__main__":
    train(build_parser().parse_args())
