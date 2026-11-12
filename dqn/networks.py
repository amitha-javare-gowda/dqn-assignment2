import torch
import torch.nn as nn


class MLPQNet(nn.Module):
    """Q-network for low-dimensional states (CartPole)."""

    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class AtariQNet(nn.Module):
    """Nature-DQN style CNN for stacked 88x80 MsPacman frames.

    Input: (B, C, 88, 80) int8 in [-128, 127] from the assignment preprocessing.
    It is rescaled to roughly [-1, 1] inside forward so the replay buffer can
    stay int8.
    """

    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),   # -> 32x21x19
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),            # -> 64x9x8
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),            # -> 64x7x6
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, in_channels, 88, 80)).numel()
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(n_flat, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = x.float() / 128.0
        return self.head(self.conv(x))
