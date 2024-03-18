# Substrafl local and remote speed benchmark

## Table of content

- [Introduction](#introduction)
  - [Objectives](#objectives)
  - [Basic install](#basic-install)
- [Local speed benchmark](#local-speed-benchmark)
- [Remote speed benchmark](#remote-speed-benchmark)

## Introduction

### Objectives

Being able to run the benchmark on a deployed Substra with a variable amount of data.
Being able to compare substrafl local speed to a full torch example with a variable amount of data.

### Basic install

**All the commands should be run from `benchmark/camelyon`**

In a new python 3.11 environment :

```sh
pip install -r requirements.txt
```

### Common Installation error

Please ensure that your python installation is complete. For Mac users if you get the following warning :

```sh
UserWarning: Could not import the lzma module. Your installed Python is incomplete. Attempting to use lzma compression will result in a RuntimeError.
```

You'll need to uninstall python 3.11, and install the `xz` package :

```sh
brew install xz
```

then reinstall python 3.11.

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
--credentials-path CREDENTIALS      Remote only: relative path from the substra_conf folder to Substra credentials (default: remote.yaml)
--assets-keys-path asset_keys_PATH
                                    Remote only: relative path from the substra_conf folder to a file where to fill in the Substra assets to be reused (default: keys.json)
--nb-train-data-samples NB_TRAIN_DATA_SAMPLES
                                    Number of data sample of 400 Mb to use for each train task on each center (default: 5)
--nb-test-data-samples NB_TEST_DATA_SAMPLES
                                    Number of data sample of 400 Mb to use for each test task on each center (default: 2)
--torch-gpu                         Use PyTorch with GPU/CUDA support (default: False)
--cp-name CP_NAME                   Compute Plan name to display (default: None)
```

### Dataset

For this example, we use a few samples from the Camelyon dataset already processed by the [FLamby team](https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_camelyon16).

There are 4 Camelyon slide data (i.e. extracted features from slides), for a total of 392 Mb.

Those 392 mb of data are saved as one Substra data sample. It is duplicated to create other data samples and form a large dataset (eg 400Gb per organization).

**IMPORTANT NOTE:** those folder are populated with `symlinks` hence no useless space will be used on your device.

### Model

The model is named `Weldon`. The implementation is taken from [this paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Durand_WELDON_Weakly_Supervised_CVPR_2016_paper.pdf). Increasing the performances of the experiment could be done by implementing more complex models like [Chowder](https://arxiv.org/pdf/1802.02212.pdf)

## Local speed benchmark

The local benchmark aims at comparing the execution time between

- a substrafl fed avg experiment (with substra local mode as a backend)
- a pure torch implementation of a fed avg experiment

Substra local mode can be chosen from the CLI:

```sh
python benchmarks.py --mode subprocess
python benchmarks.sh --mode docker
```

### Results

After each run the [result file](./results/results.json) is populated with a new entry:

```json
{
    "2023-08-07T09:18:50": {
        "asset_keys": "keys.json",
        "batch_size": 4,
        "credentials": "remote.yaml",
        "learning_rate": 0.01,
        "mode": "subprocess",
        "n_centers": 2,
        "n_local_steps": 1,
        "n_rounds": 2,
        "nb_test_data_samples": 2,
        "nb_train_data_samples": 2,
        "num_workers": 0,
        "results": {
            "substrafl": {
                "exec_time": 28.036304,
                "n_clients": 2,
                "performances": {
                    "Accuracy": 0.5,
                    "ROC AUC": 1.0
                }
            },
            "torch": {
                "exec_time": 2.4744441509246826,
                "n_clients": 2,
                "performances": {
                    "Accuracy": 0.5,
                    "ROC AUC": 1.0
                }
            }
        },
        "seed": 42,
        "substra_version": "0.45.0",
        "substrafl_version": "0.38.0",
        "substratools_version": "0.20.0"
    }
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

Modify the [configuration file](./substra_conf/remote.yaml) with the needed credentials
If you already have a configuration file, you can specify its relative path to the `./substra_conf` folder as
an optional arg when running the benchmark:

```sh
python benchmarks.py --mode remote --credentials my_credentials.yaml
```

_..note:: If you're using a locally deployed environment, make sure your backend nginx configurations are ok with body sizes involved during the benchmark:_

```yaml
server:
  ingress:
    enabled: true
    hostname: "substra-backend.org-1.com"
    annotations:
      nginx.ingress.kubernetes.io/proxy-body-size: 0m  # 0 for infinite
```

### Substra assets management

In remote, one can choose to reuse some assets by passing their keys in the [keys.json](./substra_conf/keys.json), e.g.:

```json
{
  "MyOrg1MSP": {
    "dataset_key": "b8d754f0-40a5-4976-ae16-8dd4eca35ffc",
    "data_sample_keys": ["1238452c-a1dd-47ef-84a8-410c0841693a"],
    "train_data_sample_keys": ["38071944-c974-4b3b-a671-aa4835a0ae62"]
  },
  "MyOrg2MSP": {
    "dataset_key": "fa8e9bf7-5084-4b59-b089-a459495a08be",
    "data_sample_keys": ["73715d69-9447-4270-9d3f-d0b17bb88a87"],
    "train_data_sample_keys": ["766d2029-f90b-440e-8b39-2389ab04041d"]
  },
  "metric_key": "e5a99be6-0138-461a-92fe-23f685cdc9e1"
}
```

If an asset is not filled in, it will be created and registered to Substra, otherwise it will be reused.

At the end of the asset registration, this file will be completed with the new registered asset keys.

If you want to use a custom asset file, you can pass its relative path to the substra_conf folder as argument when running the benchmark:

```sh
python benchmarks.py --assets-keys my_keys.json
```

The two arguments :

- `--nb-train-data-samples`
- `--nb-train-data-samples`

are not overwritten by the value within the `asset_keys.json` file.

This means that if the number of data samples passed within each `train_data_sample_keys` field is:

- greater than `--nb-train-data-samples`, only the first `--nb-train-data-samples` will be used

- lower than `--nb-train-data-samples`, the right number of data sample will be added to Substra so `--nb-train-data-samples` will be used
