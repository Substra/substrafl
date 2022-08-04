# Substrafl local and remote speed benchmark

## Table of content

- [Introduction](#introduction)
  - [Objectives](#objectives)
  - [Basic install](#basic-install)
- [Local speed benchmark](#local-speed-benchmark)
- [Remote speed benchmark](#remote-speed-benchmark)

## Introduction

### Objectives

Being able to run the benchmark on a deployed connect with a variable amount of data.
Being able to compare substrafl local speed to a full torch example with a variable amount of data.

### Basic install

**All the commands should be run from `benchmark/camelyon`**

In a new python 3.9 environment :

```sh
pip install -r requirements.txt
pip download --no-deps classic-algos==1.6.0
```

### Common Installation error

Please ensure that your python installation is complete. For Mac users if you get the following warning :

```sh
UserWarning: Could not import the lzma module. Your installed Python is incomplete. Attempting to use lzma compression will result in a RuntimeError.
```

You'll need to uninstall python 3.9, and install the `xz` package :

```sh
brew install xz
```

then reinstall python 3.9.

### Getting Started

To check that everything runs smoothly, you can test the local benchmark with the default parameters:

```sh
python benchmarks.py --mode subprocess --n-rounds 2 --n-local-steps 1 --nb-train-data-samples 1 --nb-test-data-samples 1 --batch-size 8
```

### Cli

```txt
--n-centers N_CENTERS               Local only: number of center to execute the benchmark on (default: 2)
--n-rounds N_ROUNDS                 Number of rounds of the strategy to execute (default: 11)
--n-local-steps N_LOCAL_STEPS       Number of batches to learn from at each step of the strategy (default: 50)
--batch-size BATCH_SIZE             Number of samples used in each local step (i.e. each batch) (default: 16)
--num-workers NUM_WORKERS           Number of torch workers to use for data loading (default: 0)
--mode MODE                         Benchmark mode, either `subprocess`, `docker` or `remote` (default: subprocess)
--credentials-path CREDENTIALS      Remote only: relative path from the connect_conf folder to connect credentials (default: remote.yaml)
--assets-keys-path asset_keys_PATH
                                    Remote only: relative path from the connect_conf folder to a file where to fill in the connect assets to be reused (default: keys.json)
--nb-train-data-samples NB_TRAIN_DATA_SAMPLES
                                    Number of data sample of 400 Mb to use for each train task on each center (default: 5)
--nb-test-data-samples NB_TEST_DATA_SAMPLES
                                    Number of data sample of 400 Mb to use for each test task on each center (default: 2)
```

### Dataset

For this example, we use 400 mb of the Camelyon dataset already processed by the Owkin R&D team. The dataset is automatically downloaded from owkin GCP, please ensure you have access to [this folder](https://console.cloud.google.com/storage/browser/camelyon_0_5?project=connectors-preview)

Those 400 mb of data are saved as one data sample. It is duplicated to create other data samples and form a large dataset (eg 400Gb per organization).

**IMPORTANT NOTE:** those folder are populated with `symlinks` hence no useless space will be used on your device.

## Local speed benchmark

The local benchmark aims at comparing the execution time between

- a substrafl fed avg experiment (with substra debug mode as a backend)
- a pure torch implementation of a fed avg experiment

Substra local mode can be chosen from the CLI:

```sh
python benchmarks.py --mode subprocess
python benchmarks.sh --mode docker
```

### Results

After each run the [result file](./results/results.json) is populated with a new entry:

```json
 "1651743200.005445": {  // timestamp of the experiment
        "asset_keys": "keys.json",
        "batch_size": 4,
        "substrafl_perf": {
        // substrafl performances on each center (should be the same and the same as the torch results)
            "0": 0.25000000000000006,
            "1": 0.25000000000000006
        },
        "substrafl_time": 179.76791310310364, // substrafl execution time (do not take the data registration into account)
        "substrafl_version": "0.11.0",
        "credentials": "remote.yaml",
        "learning_rate": 0.01,
        "mode": "subprocess",
        "n_centers": 2,
        "n_local_steps": 2,
        "n_rounds": 2,
        "nb_test_data_samples": 5,
        "nb_train_data_samples": 5,
        "num_workers": 3,
        "pure_torch_perf": {
        // pure torch performances on each center (should be the same and the same as the substrafl results)
            "0": 0.25520833333333337,
            "1": 0.25520833333333337
        },
        "pure_torch_time": 110.0057361125946, // pure torch execution time
        "seed": 42,
        "substra_version": "0.18.0",
        "substratools_version": "0.11.0"
    }
```

### Full local benchmark

To run a very complete benchmark (48 hours of compute time):

```sh
sh benchmarks.sh
```

## Remote speed benchmark

The benchmark can be run on remote with

```sh
python benchmarks.py --mode remote
```

### Configuration

Modify the [configuration file](./connect_conf/remote.yaml) with the needed credentials
If you already have a configuration file, you can specify its relative path to the `./connect_conf` folder as
an optional arg when running the benchmark:

```sh
python benchmarks.py --mode remote --credentials my_credentials.yaml
```

### Connect assets management

In remote, one can chose to reuse some assets by passing their keys in the [keys.json](./connect_conf/keys.json), e.g.:

```json
{
    "MyOrg1MSP": {
        "dataset_key": "b8d754f0-40a5-4976-ae16-8dd4eca35ffc",
        "test_data_sample_keys": ["1238452c-a1dd-47ef-84a8-410c0841693a"],
        "train_data_sample_keys": ["38071944-c974-4b3b-a671-aa4835a0ae62"]
    },
    "MyOrg2MSP": {
        "dataset_key": "fa8e9bf7-5084-4b59-b089-a459495a08be",
        "test_data_sample_keys": ["73715d69-9447-4270-9d3f-d0b17bb88a87"],
        "train_data_sample_keys": ["766d2029-f90b-440e-8b39-2389ab04041d"]
    },
    "metric_key": "e5a99be6-0138-461a-92fe-23f685cdc9e1"
}
```

If an asset is not filled in, it will be created and registered to connect, otherwise it will be reused.

At the end of the asset registration, this file will be completed with the new registered asset keys.

If you want to use a custom asset file, you can pass its relative path to the connect_conf folder as argument when running the benchmark:

```sh
python benchmarks.py --assets-keys my_keys.json
```

The two arguments :

- `--nb-train-data-samples`
- `--nb-train-data-samples`

are not overwritten by the value within the `asset_keys.json` file.

This means that if the number of data samples passed within each `train_data_sample_keys` field is:

- greater than `--nb-train-data-samples`, only the first `--nb-train-data-samples` will be used

- lower than `--nb-train-data-samples`, the right number of data sample will be added to connect so `--nb-train-data-samples` will be used
