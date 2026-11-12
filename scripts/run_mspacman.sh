#!/usr/bin/env bash
# Full MsPacman pipeline on ONE GPU: train -> plots -> 500-episode rollout.
# Usage (inside tmux/screen so it survives logout):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_mspacman.sh
# If it stops early (time limit, crash), run it again: training resumes from last.pt.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
OUT=results/mspacman
mkdir -p $OUT
RESUME=""
[ -f $OUT/last.pt ] && RESUME="--resume"
python train_mspacman.py --episodes 5000 --out $OUT --max-hours ${MAX_HOURS:-47} $RESUME 2>&1 | tee -a $OUT/train.log
python plots.py --log $OUT/log.json --game mspacman
python evaluate.py --game mspacman --ckpt $OUT/best.pt --episodes 500 2>&1 | tee $OUT/eval.log
