"""MsPacman-v5 environment helpers: assignment preprocessing + frame stacking."""
from collections import deque

import ale_py
import gymnasium as gym
import numpy as np

gym.register_envs(ale_py)

ENV_ID = "ALE/MsPacman-v5"
mspacman_color = 210 + 164 + 74


def preprocess_observation(obs):
    """Exactly the preprocessing given in the assignment handout."""
    img = obs[1:176:2, ::2]           # crop and downsize
    img = img.sum(axis=2)             # to greyscale
    img[img == mspacman_color] = 0    # improve contrast
    img = (img // 3 - 128).astype(np.int8)  # normalize from -128 to 127
    return img.reshape(88, 80, 1)


class FrameStack:
    """Keeps the last k preprocessed frames as a (k, 88, 80) int8 array.

    Edge case: on reset the stack is filled with k copies of the first frame,
    so the network never sees all-zero (or previous-episode) frames.
    """

    def __init__(self, k):
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, frame):
        f = frame[..., 0]
        for _ in range(self.k):
            self.frames.append(f)
        return self.state()

    def push(self, frame):
        self.frames.append(frame[..., 0])
        return self.state()

    def state(self):
        return np.stack(self.frames, axis=0)


def make_env(render_mode=None, **kwargs):
    return gym.make(ENV_ID, render_mode=render_mode, **kwargs)
