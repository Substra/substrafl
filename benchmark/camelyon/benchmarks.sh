#!/bin/bash
set -euxo pipefail

DEFAULT_NROUNDS=2
DEFAULT_SAMPLING=1
DEFAULT_STEPS=10

for SUB_SAMPLING in 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
do
	python benchmarks.py --n-local-steps $DEFAULT_STEPS --n-rounds $DEFAULT_NROUNDS --sub-sampling $SUB_SAMPLING
done

for N_STEP in 10 20 40 80 150 225 300
do
	python benchmarks.py --n-local-steps $N_STEP --n-rounds $DEFAULT_NROUNDS --sub-sampling $DEFAULT_SAMPLING
done

for N_ROUNDS in 3 5 7 10 13 19 25 35 50
do
	python benchmarks.py --n-local-steps $DEFAULT_STEPS --n-rounds $N_ROUNDS --sub-sampling $DEFAULT_SAMPLING
done
