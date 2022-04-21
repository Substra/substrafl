#!/bin/bash
set -euxo pipefail

DEFAULT_NROUNDS=2
DEFAULT_SAMPLING=1
DEFAULT_STEPS=10
DEFAULT_BATCH_SIZE=32
DEFAULT_NUM_WORKERS=0
MODE=subprocess

for N_STEP in 10 20 40 80 150 225 300
do
	python benchmarks.py \
    --mode $MODE \
	--n-local-steps $N_STEP \
	--n-rounds $DEFAULT_NROUNDS \
	--sub-sampling $DEFAULT_SAMPLING \
	--batch-size $DEFAULT_BATCH_SIZE \
    --num-workers $DEFAULT_NUM_WORKERS
done

for N_ROUNDS in 3 5 7 10 13 19 25 35 50
do
	python benchmarks.py \
    --mode $MODE \
	--n-local-steps $DEFAULT_STEPS \
	--n-rounds $N_ROUNDS \
	--sub-sampling $DEFAULT_SAMPLING \
	--batch-size $DEFAULT_BATCH_SIZE \
    --num-workers $DEFAULT_NUM_WORKERS
done

for NUM_WORKERS in 0 1 2 3 4
do
    for BATCH_SIZE in 8 16 32 64 128
    do
        python benchmarks.py \
        --mode $MODE \
        --n-local-steps $DEFAULT_STEPS \
        --n-rounds $DEFAULT_NROUNDS \
        --sub-sampling $DEFAULT_SAMPLING \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS
    done
done
