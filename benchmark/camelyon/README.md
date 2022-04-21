# Objectives

Being able to run the benchmark on a deployed connect.
Being able to compare connectlib local speed to a full torch example.

## Dataset

For this example, we will use the Camelyon dataset already processed per Owkin R&D team. It is automatically downloaded from owkin GCP, please ensure you have access to [this folder](https://console.cloud.google.com/storage/browser/camelyon_0_5?project=connectors-preview)

## Environment

### CPU/GPU

This benchmark has not been tested on GPU even if it is compatible, it is expected to fail on the first try.

### Basic install

In a new python 3.9 environment :

```sh
cd benchmark/camelyon
pip install -r requirements.txt
```

also do:

```sh
cd benchmark/camelyon
pip download --no-deps classic-algos==1.6.0
```

### Common error

Please ensure that your python installation is complete. For Mac users if you get the following warning :

```sh
UserWarning: Could not import the lzma module. Your installed Python is incomplete. Attempting to use lzma compression will result in a RuntimeError.
```

You'll need to uninstall python 3.9, and install the `xz` package :

```sh
brew install xz
```

then reinstall python 3.9.

### Dataset generation

Generating the needed data folders can be done with the dataset_manager script:

For two centers only, you can create the train and test data folders for each center passing the exact number of tiles you need e.g.:

```sh
python -m common.dataset_manager --train-0 300 --train-1 80 --test-0 60 --test-1 20
```

will create in the [data folder](./data/) the following folders :

- train_0 with 300 tiles and an index.csv file
- train_1 with 80 tiles and an index.csv file
- test_0 with 60 tiles and an index.csv file
- test_1 with 20 tiles and an index.csv file

Regardless of the number of centers defined, you can populate each one of your center data with a multiple of the original data set (130 test and 280 train tiles).
The number of centers is defined with the `--n-centers` arg (default to 2.)

```sh
python -m common.dataset_manager --sub-sampling 0.5 --n-centers 3
```

will create in the [data folder](./data/) the following folders :

- train_0 with 140 tiles and an index.csv file
- train_1 with 140 tiles and an index.csv file
- train_2 with 140 tiles and an index.csv file
- test_0 with 65 tiles and an index.csv file
- test_1 with 65 tiles and an index.csv file
- test_2 with 65 tiles and an index.csv file

```sh
python -m common.dataset_manager --sub-sampling 2
```

will create in the [data folder](./data/) the following folders :

- train_0 with 560 tiles and an index.csv file
- train_1 with 560 tiles and an index.csv file
- test_0 with 260 tiles and an index.csv file
- test_1 with 260 tiles and an index.csv file

**IMPORTANT NOTE:** those folder are populated with `symlinks` hence no useless space will be used on your device.

## Running the benchmark on remote

All the results should be reported [here](https://docs.google.com/spreadsheets/d/1WCEh1svnvUE-u9tuSQau_cWB5FdEDOPuXzwwkQ1FrNQ/edit#gid=1147891156)

### Configuration

Modify the [configuration file](./connect_conf/remote.yaml) with the needed credentials
If you already have a configuration file, you can specify it's relative path to the `./connect_conf` folder as
an optional arg when running the benchmark:

```sh
python benchmarks.py --mode remote --credentials my_credentials.yaml
```

### Upload the data to connect

As the data samples are too big, they need to be added on the `server-media` pod of connect:

```sh
# Set the cluster names
cluster1=cg-cluster-conlib-bm-2
cluster2=cg-cluster-conlib-bm-3

# Connect to connect cluster org-1
gcloud container clusters get-credentials --zone europe-west1 $cluster1

# Retrieve the server media pod name
POD=$(kubectl get pods --namespace=connect --no-headers -o custom-columns=":metadata.name" | grep "server")

# Copy train_0 and test_0 on the server media pod
kubectl cp train_0 connect/$POD:/var/substra/servermedias/train
kubectl cp test_0 connect/$POD:/var/substra/servermedias/test

# Check that all the needed files are at the right place
kubectl exec -it --namespace connect $POD -- /bin/sh
cd /var/substra/servermedias/ && ls train && ls test

# retrieve the data samples size and report it to the benchmark results
du -hs train
du -hs test

# Connect to connect cluster org-2
gcloud container clusters get-credentials --zone europe-west1 $cluster2

# Retrieve the server media pod name-
POD=$(kubectl get pods --namespace=connect --no-headers -o custom-columns=":metadata.name" | grep "server")

# Copy train_1 and test_1 on the server media pod
kubectl cp train_1 connect/$POD:/var/substra/servermedias/
kubectl cp test_1 connect/$POD:/var/substra/servermedias/

# Check that all the needed files are at the right place
kubectl exec -it --namespace connect $POD -- /bin/sh
cd /var/substra/servermedias/ && ls train && ls test

# retrieve the data samples size and report it to the benchmark results
du -hs train
du -hs test
```

INFO: in remote mode, the benchmark automatically seek the data for each client under the "train" and the "test" folder
of the servermedia pod. There is nothing to modify in the code if you copy the data at the right place.

If you want to artificially increase the size of the data sample without having to re upload data, you can execute:

```sh
for FILENAME in *.npy; do cp $FILENAME copy_$FILENAME; done
```

in both the train and test folders of each server pod. This will multiply by 2 the size of your data samples.
Then, remove the corresponding data sample keys for the [keys.json](./connect_conf/keys.json) file and relaunch the experiment.

### Connect assets management

In remote, one can chose to reuse some assets by passing their keys in the [keys.json](./connect_conf/keys.json), e.g.:

```json
{
    "MyOrg1MSP": {
        "dataset_key": "b8d754f0-40a5-4976-ae16-8dd4eca35ffc",
        "test_data_sample_key": "1238452c-a1dd-47ef-84a8-410c0841693a",
        "train_data_sample_key": "38071944-c974-4b3b-a671-aa4835a0ae62"
    },
    "MyOrg2MSP": {
        "dataset_key": "fa8e9bf7-5084-4b59-b089-a459495a08be",
        "test_data_sample_key": "73715d69-9447-4270-9d3f-d0b17bb88a87",
        "train_data_sample_key": "766d2029-f90b-440e-8b39-2389ab04041d"
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

### Launching the benchmark

If the benchmark runs in remote mode, it will search for the `train` and `test` folders on the server media pod for each client.

Some definitions :

- `batch_size`: batch size used during training
- `learning_rate`: learning rate of the optimizer
- `n_local_steps`: the number of times models from each nodes will be trained on a different batch for each round of the strategy
- `n_rounds`: the number of rounds of the strategy
- `seed`: random seed used for results reproducibility

All the above arguments can be defined through the cli.

```sh
python benchmarks.py --mode remote --n-local-steps 50 --n-rounds 11 --batch-size 16
```

## Running local benchmark

The local benchmark aims at comparing connectlib debug mode and a pure torch implementation.

Some additional definitions:

- `connectlib_perf`: AUC of connectlib fed avg.
- `connectlib_time`: Connectlib fed avg execution time (s)
- `n_centers`: number of centers to execute the strategy on
- `pure_torch_perf`: AUC of torch fed avg performances
- `pure_torch_time`: Torch fed avg execution time (s)
- `sub_sampling`: the fraction of the dataset to use for the benchmark - this has no impact on the duration of the training, but it impacts the size of the test set and the duration of the metric calculation.

WARNING with the sub_sampling: if any center has less data than the batch size, the benchmarks throws an error. Since in that case the batch size would get automatically changed to be equal to the number of samples, it would skew the results.

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
python benchmarks.py --mode subprocess
```

The possibility is given to change to some of the parameters from the cli :

- n_local_steps with the --n-local-steps arg
- n_rounds with the --n-rounds arg
- sub_sampling with the --sub-sampling arg
- batch size with the --batch-size arg
- number of workers with the --num-workers arg

for example the following command will run the benchmark with 4 local steps, 2 rounds on half of the dataset.

```sh
python benchmarks.py --mode subprocess --n-local-steps 4 --n-rounds 2 --sub-sampling 0.5
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

## Plot the results of the local benchmarks

Default comparative graphics can be generated with:

```sh
python plot_results.py
```
