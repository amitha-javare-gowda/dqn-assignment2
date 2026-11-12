"""DQN agent: online Q-network, target network, epsilon-greedy policy, TD update.

Options exposed so the validation experiments can ablate them:
  use_target    False -> targets are computed with the online net (ablation)
  double        Double-DQN target (argmax from online net, value from target net)
  loss          'huber' or 'mse'
  target_update 'hard' copy every `target_period` gradient steps, or 'soft'
                Polyak averaging with coefficient `tau`
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F


class DQNAgent:
    def __init__(self, q_net, n_actions, gamma, lr=1e-3, device="cpu",
                 use_target=True, double=False, loss="huber",
                 target_update="hard", target_period=1000, tau=0.005,
                 grad_clip=10.0, seed=0):
        self.device = torch.device(device)
        self.q = q_net.to(self.device)
        self.q_target = copy.deepcopy(self.q).eval()
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.n_actions = n_actions
        self.gamma = gamma
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.use_target = use_target
        self.double = double
        self.loss_name = loss
        self.target_update = target_update
        self.target_period = target_period
        self.tau = tau
        self.grad_clip = grad_clip
        self.n_updates = 0
        self.rng = np.random.default_rng(seed)

    # ---------------- acting ----------------
    @torch.no_grad()
    def q_values(self, obs):
        x = torch.as_tensor(np.asarray(obs), device=self.device)
        if x.dtype == torch.float64:
            x = x.float()
        return self.q(x.unsqueeze(0)).squeeze(0).cpu().numpy()

    def act(self, obs, epsilon):
        """Epsilon-greedy. Returns (action, q-values of obs)."""
        q = self.q_values(obs)
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions)), q
        # break exact ties randomly so a freshly initialised net is not biased
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best)), q

    # ---------------- learning ----------------
    def _to_t(self, a, dtype=None):
        t = torch.as_tensor(a, device=self.device)
        return t.to(dtype) if dtype is not None else t

    def compute_targets(self, rewards, next_obs, terminated):
        with torch.no_grad():
            eval_net = self.q_target if self.use_target else self.q
            q_next_eval = eval_net(next_obs)
            if self.double:
                a_star = self.q(next_obs).argmax(1, keepdim=True)
                v_next = q_next_eval.gather(1, a_star).squeeze(1)
            else:
                v_next = q_next_eval.max(1).values
            return rewards + self.gamma * (1.0 - terminated) * v_next

    def update(self, batch):
        obs, actions, rewards, next_obs, terminated = batch
        obs = self._to_t(obs)
        next_obs = self._to_t(next_obs)
        if obs.dtype == torch.float64:
            obs, next_obs = obs.float(), next_obs.float()
        actions = self._to_t(actions, torch.int64)
        rewards = self._to_t(rewards, torch.float32)
        terminated = self._to_t(terminated, torch.float32)

        q_sa = self.q(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        target = self.compute_targets(rewards, next_obs, terminated)
        if self.loss_name == "huber":
            loss = F.smooth_l1_loss(q_sa, target)
        else:
            loss = F.mse_loss(q_sa, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.opt.step()
        self.n_updates += 1
        self._sync_target()
        td = (target - q_sa).detach()
        return float(loss.item()), td.abs().mean().item()

    def _sync_target(self):
        if self.target_update == "soft":
            with torch.no_grad():
                for p_t, p in zip(self.q_target.parameters(), self.q.parameters()):
                    p_t.mul_(1 - self.tau).add_(p, alpha=self.tau)
        elif self.n_updates % self.target_period == 0:
            self.q_target.load_state_dict(self.q.state_dict())

    # ---------------- io ----------------
    def save(self, path, extra=None):
        torch.save({"q": self.q.state_dict(), "q_target": self.q_target.state_dict(),
                    "opt": self.opt.state_dict(), "n_updates": self.n_updates,
                    "extra": extra or {}}, path)

    def load(self, path, load_opt=True):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.q.load_state_dict(ck["q"])
        self.q_target.load_state_dict(ck["q_target"])
        if load_opt and "opt" in ck:
            self.opt.load_state_dict(ck["opt"])
        self.n_updates = ck.get("n_updates", 0)
        return ck.get("extra", {})


def linear_schedule(start, end, duration, t):
    """Linear interpolation from start to end over `duration` steps, then flat."""
    if duration <= 0:
        return end
    frac = min(max(t / duration, 0.0), 1.0)
    return start + frac * (end - start)
