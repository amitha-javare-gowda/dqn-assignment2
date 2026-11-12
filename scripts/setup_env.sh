#!/usr/bin/env bash
# One-time setup on the GPU machine. Run from the repo root.
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
python -c "import gymnasium as gym, ale_py; gym.register_envs(ale_py); e=gym.make('ALE/MsPacman-v5'); print('MsPacman ok, actions =', e.action_space.n)"
