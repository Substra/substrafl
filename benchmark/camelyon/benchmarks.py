import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

import substra
import substratools
from common.dataset_manager import fetch_camelyon
from common.dataset_manager import reset_data_folder
from common.dataset_manager import split_dataset
from workflows import connectlib_fed_avg
from workflows import torch_fed_avg

import connectlib

RESULTS_FOLDER = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_FOLDER / "results.json"


def fed_avg(params: dict):
    """Running both connectlib and pure_torch fed avg strategy with the given parameters

    Args:
        params (dict): Strategy and algo parameters.

    Returns:
        dict: Results of both experiment with their computation time and the used parameters
    """
    exp_params = params.copy()

    reset_data_folder()
    trains_folders, test_folder = split_dataset(
        n_centers=exp_params["n_centers"], sub_sampling=exp_params["sub_sampling"]
    )

    # Check on the number of samples
    for train_folder in trains_folders:
        len_data = len((Path(train_folder) / "index.csv").read_text().splitlines()) - 1
        if len_data < exp_params["batch_size"]:
            raise ValueError(
                "The length of the dataset is smaller than the batch size, not allowed as it"
                "skews the benchmark results (the batch size gets automatically adjusted in that case)."
            )
    run_keys = ["n_rounds", "seed", "batch_size", "n_centers", "learning_rate", "n_local_steps", "num_workers"]

    cl_start = time.time()
    cl_perf = connectlib_fed_avg(
        trains_folders=trains_folders,
        test_folder=test_folder,
        **{k: v for k, v in exp_params.items() if k in run_keys},
    )
    cl_end = time.time()

    sa_start = time.time()
    sa_perf = torch_fed_avg(
        trains_folders=trains_folders,
        test_folder=test_folder,
        **{k: v for k, v in exp_params.items() if k in run_keys},
    )
    sa_end = time.time()

    exp_params.update(
        {
            "connectlib_time": cl_end - cl_start,
            "pure_torch_time": sa_end - sa_start,
            "connectlib_perf": cl_perf,
            "pure_torch_perf": sa_perf,
        }
    )

    return {str(time.time()): exp_params}


def parse_params() -> dict:
    """Init experiment parameters from passed args.

    Returns:
        dict: Experiment parameters
    """

    params = {
        "seed": 42,
        "n_centers": 2,
        "learning_rate": 0.01,
        "connectlib_version": connectlib.__version__,
        "substra_version": substra.__version__,
        "substratools_version": substratools.__version__,
    }

    parser = argparse.ArgumentParser("Default parser.")
    parser.add_argument("--sub-sampling", type=float, default=1)
    parser.add_argument("--n-rounds", type=int, default=2)
    parser.add_argument("--n-local-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()
    params["sub_sampling"] = args.sub_sampling
    params["n_rounds"] = args.n_rounds
    params["n_local_steps"] = args.n_local_steps
    params["batch_size"] = args.batch_size
    params["num_workers"] = args.num_workers

    return params


def read_results() -> dict:
    """Read previous results from file if exists.

    Returns:
        dict: Previous results from former benchmark.
    """

    RESULTS_FOLDER.mkdir(exist_ok=True)

    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text())

    else:
        results = {}

    return results


def main():
    # https://github.com/pytest-dev/pytest-flask/issues/104
    # necessary on OS X, Python >= 3.8 to run multiprocessing
    multiprocessing.set_start_method("fork")

    os.environ["DEBUG_SPAWNER"] = "docker"

    # Get dataset
    fetch_camelyon()

    # Parse experiment params from the cli and system configuration
    params = parse_params()

    # Read old benchmark results from file
    results = read_results()

    # Execute experiment
    res = fed_avg(params)

    # Update results
    results.update(res)

    # Save results
    RESULTS_FILE.write_text(json.dumps(results, sort_keys=True, indent=4))


if __name__ == "__main__":
    main()
