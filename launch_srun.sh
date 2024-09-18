srun -A nexus -p tron -q default --output slurm-%j.out \
    python scripts/word_count.py data/essay.txt -o data/essay-word-count.json