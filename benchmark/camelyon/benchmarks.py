import argparse
import json
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
    trains_folders, test_folder = split_dataset(n_centers=params["n_centers"], sub_sampling=params["sub_sampling"])

    cl_start = time.time()
    cl_perf = connectlib_fed_avg(
        trains_folders=trains_folders,
        test_folder=test_folder,
        seed=params["seed"],
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        n_centers=params["n_centers"],
        learning_rate=params["learning_rate"],
        n_rounds=params["n_rounds"],
        n_local_steps=params["n_local_steps"],
    )
    cl_end = time.time()

    sa_start = time.time()
    sa_perf = torch_fed_avg(
        trains_folders=trains_folders,
        test_folder=test_folder,
        seed=params["seed"],
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        n_centers=params["n_centers"],
        learning_rate=params["learning_rate"],
        n_rounds=params["n_rounds"],
        n_local_steps=params["n_local_steps"],
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
        "batch_size": 32,
        "num_workers": 0,
        "n_centers": 2,
        "learning_rate": 0.01,
        "n_rounds": 11,
        "n_local_steps": 50,
    }

    parser = argparse.ArgumentParser("Default parser.")
    parser.add_argument("--sub-sampling", type=float, default=0.1)
    parser.add_argument("--n-rounds", type=int, default=2)
    parser.add_argument("--n-local-steps", type=int, default=2)

    args = parser.parse_args()
    params["sub_sampling"] = float(args.sub_sampling)
    params["n_rounds"] = int(args.n_rounds)
    params["n_local_steps"] = int(args.n_local_steps)

    params["connectlib_version"] = connectlib.__version__
    params["substra_version"] = substra.__version__
    params["substratools_version"] = substratools.__version__

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
    os.environ["DEBUG_SPAWNER"] = "subprocess"

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
