#!/bin/bash

#SBATCH --job-name=word_count
#SBATCH --output=slurm-%j.out
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --time=01:00:00

args="scripts/word_count.py data/essay.txt -o data/essay-word-count.json"

# The difference between `python $args` and `srun python $args`:
#
# 1. `python $args`:
#    - Runs the Python script directly on the allocated node
#    - Does not use SLURM's srun command for task launching
#    - Suitable for single-node, single-task jobs
#
# 2. `srun python $args`:
#    - Uses SLURM's srun command to launch the Python script
#    - Provides better integration with SLURM's resource management
#    - Allows for more advanced options like multi-node or multi-task jobs
#    - Can provide additional control over resource allocation
#
# For this single-node job, both methods will work similarly.
# However, using `srun` is generally recommended for better SLURM integration.

srun python $args
