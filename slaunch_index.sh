datasets=("beir/fiqa" "beir/arguana")   
slaunch scripts/index_beir.py --sweep dataset shard_id --exp-name="index_beir" --profile scavenger --gres=gpu:1 \
    --dataset ${datasets[@]} \
    --shard-id 0 1 2 3 \
    --n-shards 4 \
    --model "sentence-transformers/all-MiniLM-L6-v2" \
    --output-dir "/fs/clip-scratch/mgor/indexes/"