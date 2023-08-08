import multiprocessing
import shutil
import sys
from pathlib import Path

from common.benchmark_metrics import assert_expected_results
from common.dataset_manager import creates_data_folder
from common.dataset_manager import fetch_camelyon
from common.dataset_manager import reset_data_folder
from common.utils import load_benchmark_summary
from common.utils import parse_params
from common.weldon import Weldon
from workflows import substrafl_fed_avg
from workflows import torch_fed_avg

from substrafl.index_generator import NpIndexGenerator

PARENT_FOLDER = Path(__file__).parent
LOCAL_RESULTS_FILE = PARENT_FOLDER / "results" / "results.json"


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
        n_top=10,
        n_bottom=10,
    )

    run_keys = [
        "n_rounds",
        "n_centers",
        "learning_rate",
        "num_workers",
        "nb_train_data_samples",
        "nb_test_data_samples",
        "seed",
    ]

    substrafl_metrics = substrafl_fed_avg(
        train_folder=train_folder,
        test_folder=test_folder,
        **{k: v for k, v in exp_params.items() if k in run_keys},
        credentials_path=exp_params["credentials"],
        asset_keys_path=exp_params["asset_keys"],
        index_generator=index_generator,
        model=model,
        mode=exp_params["mode"],
    )

    torch_metrics = torch_fed_avg(
        train_folder=train_folder,
        test_folder=test_folder,
        **{k: v for k, v in exp_params.items() if k in run_keys},
        index_generator=index_generator,
        model=model,
    )

    results = {**exp_params, **{"results": {**substrafl_metrics.to_dict, **torch_metrics.to_dict}}}
    load_benchmark_summary(file=LOCAL_RESULTS_FILE, experiment_summary=results)
    assert_expected_results(substrafl_metrics=substrafl_metrics, torch_metrics=torch_metrics, exp_params=exp_params)


def main():
    # https://github.com/pytest-dev/pytest-flask/issues/104
    # necessary on OS X, Python >= 3.8 to run multiprocessing
    if sys.platform == "win32":
        multiprocessing.set_start_method("spawn")
    else:
        multiprocessing.set_start_method("fork")

    # Parse experiment params from the cli and system configuration
    params = parse_params()

    # Not used in remote, TODO: refactor at some point
    data_path = params.pop("data_path").resolve()
    exp_data_path = data_path / "tmp"
    fetch_camelyon(data_path)
    reset_data_folder(exp_data_path)
    train_folder = creates_data_folder(
        img_dir_path=data_path / "tiles_0.5mpp",
        dest_folder=exp_data_path / "train",
        index_path=data_path / "index.csv",
    )
    test_folder = creates_data_folder(
        img_dir_path=data_path / "tiles_0.5mpp",
        dest_folder=exp_data_path / "test",
        index_path=data_path / "index.csv",
    )

    try:
        # Execute experiment
        fed_avg(params, train_folder, test_folder)
    finally:
        # Delete the temporary experiment folders at the end of the benchmark
        shutil.rmtree("local-worker", ignore_errors=True)
        shutil.rmtree(PARENT_FOLDER / "benchmark_cl_experiment_folder", ignore_errors=True)


if __name__ == "__main__":
    main()
