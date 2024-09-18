#!/bin/bash

# This script launches the indexer job for a given dataset and model.

# *--------------------------- EDIT THIS BLOCK ----------------------------------------*
N_NODES=2
JOB_NAME="index_beir"
TAIL_LOG_FILE=true
SLURM_LOGS_DIR="/fs/clip-scratch/mgor/slurm_logs"

# GRAB the path of the current script and cd to the parent directory
SCRIPT_PATH=$(realpath $0)
echo "SCRIPT_PATH: $SCRIPT_PATH"
SCRIPT_DIR=$(dirname $SCRIPT_PATH)
cd $SCRIPT_DIR
# *------------------------------------------------------------------------------------*

DEFAULT_MODEL="sentence-transformers/all-MiniLM-L6-v2"

DATASET_NAME=$1 # E.g. "dataset/name"
MODEL_PATH=${2:-$DEFAULT_MODEL}
OUTPUT_DIR=${3:-"/fs/clip-scratch/mgor/indexes/beir"}

echo "Dataset: $DATASET_NAME"
echo "Model: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"

SLURM_SCRIPT="index.slurm"
SCRIPT_ARGS="$DATASET_NAME $MODEL_PATH $OUTPUT_DIR"
SBATCH_ARGS="--job-name=$JOB_NAME --array=0-$((N_NODES-1)) --output=$SLURM_LOGS_DIR/index_%A_%a.out"
LAUNCH_CMD="sbatch $SBATCH_ARGS $SLURM_SCRIPT $SCRIPT_ARGS"

echo "Indexer SLURM command: $LAUNCH_CMD"
INDEXER_JOB_ID=$($LAUNCH_CMD | awk '{print $4}')
echo "Indexer SLURM Job ID: $INDEXER_JOB_ID"

if [ "$TAIL_LOG_FILE" = false ]; then
    echo "Not tailing the log file."
    echo "You can tail the log file with:\n\t tail -F $SLURM_LOGS_DIR/index_${INDEXER_JOB_ID}_0.out"
    exit 0
fi

# *--------------------------- OPTIONAL FILE TAILING ----------------------------------*
SLURM_LOG_FILE="$SLURM_LOGS_DIR/index_${INDEXER_JOB_ID}_0.out"
set +x # turn echo off for the loop
echo ""
echo "Next will attempt to tail the slurm log file if it appears with:"
echo "tail -F $SLURM_LOG_FILE"
echo ""
echo "You can now Ctrl-C this launcher at any time w/o any harm if you don't want the auto-tail"
echo ""
echo "*** Waiting for the slurm log file to appear"
POLL_SECS=1
ABORT_SECS=60 # abort waiting after 1min
TOTAL_SECS=0
while [ ! -f $SLURM_LOG_FILE ]; do
    sleep $POLL_SECS
    echo -n "."
    TOTAL_SECS=$(($TOTAL_SECS + $POLL_SECS))

    if [ $TOTAL_SECS -gt $ABORT_SECS ]; then
         echo -e "\nGiving up after $TOTAL_SECS secs of waiting"
         echo "Please check 'sinfo' manually"
         exit 1
    fi
done
echo "."
echo "$SLURM_LOG_FILE appeared in $TOTAL_SECS secs"
tail -f $SLURM_LOG_FILE
# *------------------------------------------------------------------------------------*
