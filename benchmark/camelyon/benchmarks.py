import json
import multiprocessing
import shutil
import time
from pathlib import Path

from classic_algos.nn import Weldon
from common.dataset_manager import DATA_DIR
from common.dataset_manager import creates_data_folder
from common.dataset_manager import fetch_camelyon
from common.dataset_manager import reset_data_folder
from common.utils import parse_params
from common.utils import read_results
from workflows import substrafl_fed_avg
from workflows import torch_fed_avg

from substrafl.index_generator import NpIndexGenerator

RESULTS_FOLDER = Path(__file__).parent / "results"
LOCAL_RESULTS_FILE = RESULTS_FOLDER / "results.json"


def fed_avg(params: dict, train_folder: Path, test_folder: Path):
    """If remote, only running the benchmark on the Substra platform (from the remote.yaml file) else running
    both substrafl and pure_torch fed avg strategy with the given parameters.

    Args:
        params (dict): Strategy and algo parameters.
        train_folder (Path): Path to the data sample that will be used and duplicate for the benchmark.
        test_folder (Path):  Path to the data sample that will be used and duplicate for the benchmark.

    Returns:
        dict: Results of both experiment with their computation time and the used parameters
    """
    exp_params = params.copy()

    index_generator = NpIndexGenerator(
        batch_size=exp_params["batch_size"], num_updates=exp_params["n_local_steps"], drop_last=True, shuffle=False
    )

    model = Weldon(
        in_features=2048,
        out_features=1,
        n_extreme=10,
        n_top=10,
        n_bottom=10,
    )

    run_keys = [
        "n_rounds",
        "seed",
        "n_centers",
        "learning_rate",
        "num_workers",
        "nb_train_data_samples",
        "nb_test_data_samples",
    ]

    cl_start = time.time()
    cl_perf = substrafl_fed_avg(
        train_folder=train_folder,
        test_folder=test_folder,
        **{k: v for k, v in exp_params.items() if k in run_keys},
        credentials_path=exp_params["credentials"],
        asset_keys_path=exp_params["asset_keys"],
        index_generator=index_generator,
        model=model,
        mode=exp_params["mode"],
    )
    cl_end = time.time()
    exp_params.update(
        {
            "substrafl_time": cl_end - cl_start,
            "substrafl_perf": cl_perf,
        }
    )

    sa_start = time.time()
    sa_perf = torch_fed_avg(
        train_folder=train_folder,
        test_folder=test_folder,
        **{k: v for k, v in exp_params.items() if k in run_keys},
        index_generator=index_generator,
        model=model,
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

    # Not used in remote, TODO: refactor at some point
    reset_data_folder()
    train_folder = creates_data_folder(dest_folder=DATA_DIR / "train", index_path=DATA_DIR / "index.csv")
    test_folder = creates_data_folder(dest_folder=DATA_DIR / "test", index_path=DATA_DIR / "index.csv")

    try:
        # Execute experiment
        res = fed_avg(params, train_folder, test_folder)

        if params["mode"] != "remote":
            # Update results
            results.update(res)

            # Save results
            LOCAL_RESULTS_FILE.write_text(json.dumps(results, sort_keys=True, indent=4))
    finally:
        # Delete the temporary experiment folders at the end of the benchmark
        shutil.rmtree("local-worker", ignore_errors=True)
        shutil.rmtree("benchmark_cl_experiment_folder", ignore_errors=True)


if __name__ == "__main__":
    main()
