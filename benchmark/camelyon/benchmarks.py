import json
import multiprocessing
import time
from pathlib import Path

from common.dataset_manager import creates_data_folders
from common.dataset_manager import fetch_camelyon
from common.dataset_manager import reset_data_folder
from common.utils import parse_params
from common.utils import read_results
from workflows import connectlib_fed_avg
from workflows import torch_fed_avg

RESULTS_FOLDER = Path(__file__).parent / "results"
LOCAL_RESULTS_FILE = RESULTS_FOLDER / "results.json"


def fed_avg(params: dict, train_folders, test_folders):
    """If remote, only running the benchmark on the connect platform (from the remote.yaml file) else running
    both connectlib and pure_torch fed avg strategy with the given parameters.

    Args:
        params (dict): Strategy and algo parameters.

    Returns:
        dict: Results of both experiment with their computation time and the used parameters
    """
    exp_params = params.copy()

    run_keys = ["n_rounds", "seed", "batch_size", "n_centers", "learning_rate", "n_local_steps", "num_workers", "mode"]

    cl_start = time.time()
    cl_perf = connectlib_fed_avg(
        train_folders=train_folders,
        test_folders=test_folders,
        **{k: v for k, v in exp_params.items() if k in run_keys},
        credentials_path=exp_params["credentials"],
        assets_keys_path=exp_params["assets_keys"],
    )
    cl_end = time.time()
    exp_params.update(
        {
            "connectlib_time": cl_end - cl_start,
            "connectlib_perf": cl_perf,
        }
    )

    if params["mode"] != "remote":

        run_keys.pop()
        sa_start = time.time()
        sa_perf = torch_fed_avg(
            train_folders=train_folders,
            test_folders=test_folders,
            **{k: v for k, v in exp_params.items() if k in run_keys},
        )
        sa_end = time.time()

        exp_params.update(
            {
                "pure_torch_time": sa_end - sa_start,
                "pure_torch_perf": sa_perf,
            }
        )

    return {str(time.time()): exp_params}


def main():
    # https://github.com/pytest-dev/pytest-flask/issues/104
    # necessary on OS X, Python >= 3.8 to run multiprocessing
    multiprocessing.set_start_method("fork")

    # Get dataset
    fetch_camelyon()

    # Parse experiment params from the cli and system configuration
    params = parse_params()

    # Read old benchmark results from file if run in local
    if params["mode"] != "remote":
        results = read_results(LOCAL_RESULTS_FILE)

    # Not use in remote, TODO: refactor at some point
    reset_data_folder()
    train_folders = creates_data_folders(
        n_centers=params.get("n_centers"),
        sub_sampling=params.get("sub_sampling"),
        data_samples_size=params.get("data_samples_size"),
        batch_size=params.get("batch_size"),
        kind="train",
    )
    test_folders = creates_data_folders(
        n_centers=params.get("n_centers"),
        sub_sampling=params.get("sub_sampling"),
        data_samples_size=params.get("data_samples_size"),
        batch_size=params.get("batch_size"),
        kind="test",
    )

    # Execute experiment
    res = fed_avg(params, train_folders, test_folders)

    if params["mode"] != "remote":
        # Update results
        results.update(res)

        # Save results
        LOCAL_RESULTS_FILE.write_text(json.dumps(results, sort_keys=True, indent=4))


if __name__ == "__main__":
    main()
