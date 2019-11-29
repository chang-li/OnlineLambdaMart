#!/bin/sh

#SBATCH --mem=50G
#SBATCH -c2
#SBATCH -o ./out/%A_%a.out
#SBATCH -e ./out/%A_%a.err
#SBATCH --time=24:00:00

PYTHONPATH=/ivi/ilps/personal/cli1/OnlineLambdaMart/oltr/
python oltr_slurm.py
# n_topic 5 10 19
# n_pos 5 10
# normalized: True False
