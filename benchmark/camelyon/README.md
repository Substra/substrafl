# Objectives

Being able to compare connectlib speed to a full torch example.

## Dataset

For this example, we will use the Camelyon dataset already processed per Owkin R&D team. It is automatically downloaded from owkin GCP, please ensure you have access to [this folder](https://console.cloud.google.com/storage/browser/camelyon_0_5?project=connectors-preview)

## Environment

### CPU/GPU

This benchmark has not been tested on GPU even if it is compatible, it is expected to fail on the first try.

### Basic install

In a new python 3.7 environment :

```sh
cd benchmark/camelyon
pip install -r requirements.txt
```

### Common error

Please ensure that your python installation is complete. For Mac users if you get the following warning :

```sh
UserWarning: Could not import the lzma module. Your installed Python is incomplete. Attempting to use lzma compression will result in a RuntimeError.
```

For Mac users, you'll need to uninstall python 3.7, and install the `xz` package :

```sh
brew install xz
```

then reinstall python 3.7.

## Running the benchmark

Some definitions :

- `batch_size`: batch size used during training
- `connectlib_perf`: AUC of connectlib fed avg.
- `connectlib_time`: Connectlib fed avg execution time (s)
- `learning_rate`: learning rate of the optimizer
- `n_centers`: number of centers to execute the strategy on
- `n_local_steps`: the number of times models from each nodes will be trained on a different batch for each round of the strategy
- `n_rounds`: the number of rounds of the strategy
- `seed`: random seed used for results reproducibility
- `pure_torch_perf`: AUC of torch fed avg performances
- `pure_torch_time`: Torch fed avg execution time (s)
- `sub_sampling`: the fraction of the dataset to use for the benchmark

### Execute one experiment

The benchmark has default parameters :

- `batch_size`: 16
- `learning_rate`: 0.01
- `n_centers`: 2
- `n_local_steps`: 50
- `n_rounds`: 11
- `seed`: 42
- `sub_sampling`: 1

Running an experiment with the above parameters is possible with:

```sh
python benchmarks.py
```

The possibility is given to change to some of the parameters from the cli :

- n_local_steps with the --n-local-steps arg
- n_rounds with the --n-rounds arg
- sub_sampling with the --n-rounds arg

for example the following command will run the benchmark with 4 local steps, 2 rounds on half of the dataset.

```sh
python benchmarks.py --n-local-steps 4 --n-rounds 2 --sub-sampling 0.5
```

To run a very complete benchmark (48 hours of compute time):

```sh
sh benchmarks.sh
```

Per default, the [benchmark.py](benchmarks.py) file adds results at the end of the `results.json`.

The results are stored in a json format where every ran experiment is identified by the timestamp of its creation with the following information :

```json
"1642967094.1938958": {
    "batch_size": 16,
    "connectlib_perf": 0.601,
    "connectlib_time": 11404.715334177017,
    "learning_rate": 0.01,
    "n_centers": 2,
    "n_local_steps": 50,
    "n_rounds": 50,
    "num_workers": 0,
    "seed": 42,
    "pure_torch_perf": 0.5555,
    "pure_torch_time": 9618.518035888672,
    "sub_sampling": 1.0,
    "connectlib_version": "0.7.0",
    "substra_version": "0.16.0",
    "substratools_version": "0.9.1",

}
```

## Plot the results

Default comparative graphics can be generated with:

```sh
python plot_results.py
```
