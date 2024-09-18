#!/bin/bash
#SBATCH --job-name=index_dataset
#SBATCH --output=/path/to/logs/index_%A_%a.out
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=5:00:00

# Exit on error
set -eo pipefail

# Used for debugging. Prints every command being executed
set -x

# Increase open file limit  
ulimit -Sn 20480

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate qadiff

if [ $# -lt 3 ]; then
    echo "Error: Number of arguments must be at least 3: <input-data> <model-name> <output-dir>"
    exit 1
fi

# HuggingFace dataset
export INPUT_DATA=$1

# HuggingFace model
export MODEL_NAME=$2

# Output directory  
export OUTPUT_DIR=$3

export NUM_GPUS_PER_NODE=2
export NUM_GPUS_TOTAL=$((NUM_GPUS_PER_NODE * SLURM_ARRAY_TASK_COUNT))
echo "NUM_GPUS_PER_NODE: $NUM_GPUS_PER_NODE, SLURM_ARRAY_TASK_COUNT: $SLURM_ARRAY_TASK_COUNT, NUM_GPUS_TOTAL: $NUM_GPUS_TOTAL"

srun --unbuffered --ntasks-per-node=1 --gres=gpu:$NUM_GPUS_PER_NODE --exclusive bash -c '
N_SHARDS=$NUM_GPUS_TOTAL
echo "Total shards: $N_SHARDS"
for ((i=0; i<$NUM_GPUS_PER_NODE; i++)); do
    NODE_INDEX=$SLURM_ARRAY_TASK_ID
    SHARD_ID=$((NODE_INDEX * NUM_GPUS_PER_NODE + i))
    echo "Processing shard $SHARD_ID on GPU $i"
    echo "OUTPUT DIR: $OUTPUT_DIR"
    CUDA_VISIBLE_DEVICES=$i \
    python -u scripts/index_beir.py \
        --model=$MODEL_NAME \
        --shard-id=$SHARD_ID \
        --n-shards=$N_SHARDS \
        --dataset=$INPUT_DATA \
        --output-dir=$OUTPUT_DIR &
done

wait
'
