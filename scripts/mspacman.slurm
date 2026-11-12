#!/bin/bash
# Only if your cluster uses SLURM. Edit partition/account to match your cluster.
#SBATCH --job-name=dqn-mspacman
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=results/mspacman/slurm-%j.out
##SBATCH --partition=gengpu
##SBATCH --account=YOUR_ALLOCATION
# Resubmit the same script if it hits the time limit; it resumes automatically.
MAX_HOURS=23 bash scripts/run_mspacman.sh
