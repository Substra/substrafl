import argparse
import json
from pathlib import Path

import substra
import substratools

import substrafl

DEST_DIR = Path(__file__).parents[1]
DATA_DIR = DEST_DIR / "data"


def parse_params() -> dict:
    """Init experiment parameters from passed args.

    Returns:
        dict: Experiment parameters
    """

    params = {
        "seed": 42,
        "n_centers": 2,
        "learning_rate": 0.01,
        "substrafl_version": substrafl.__version__,
        "substra_version": substra.__version__,
        "substratools_version": substratools.__version__,
    }

    parser = argparse.ArgumentParser("Default parser.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--n-centers", type=int, default=2, help="Local only: number of center to execute the benchmark on"
    )
    parser.add_argument("--n-rounds", type=int, default=11, help="Number of rounds of the strategy to execute")
    parser.add_argument(
        "--n-local-steps", type=int, default=50, help="Number of batch to learn from at each step of the strategy"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Number of sample to use learn from for each local step"
    )
    parser.add_argument("--num-workers", type=int, default=0, help="Number of torch worker to use for data loading")
    parser.add_argument(
        "--mode",
        type=str,
        default="subprocess",
        help="Benchmark mode, either `subprocess`, `docker` or `remote`",
        choices=["subprocess", "docker", "remote"],
    )
    parser.add_argument(
        "--credentials-path",
        type=str,
        default="remote.yaml",
        help="Remote only: relative path from the substra_conf folder to Substra credentials",
    )
    parser.add_argument(
        "--asset-keys-path",
        type=str,
        default="keys.json",
        help="""Remote only: relative path from the substra_conf folder to a
file where to fill in the Substra assets to be reused""",
    )
    parser.add_argument(
        "--nb-train-data-samples",
        type=int,
        default=5,
        help="Number of data sample of 400 Mb to use for each train task on each center",
    )
    parser.add_argument(
        "--nb-test-data-samples",
        type=int,
        default=2,
        help="Number of data sample of 400 Mb to use for each test task on each center",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_DIR,
        help="Path to the data folder",
    )

    args = parser.parse_args()
    params["n_centers"] = args.n_centers
    params["n_rounds"] = args.n_rounds
    params["n_local_steps"] = args.n_local_steps
    params["batch_size"] = args.batch_size
    params["num_workers"] = args.num_workers
    params["mode"] = args.mode
    params["credentials"] = args.credentials_path
    params["asset_keys"] = args.asset_keys_path
    params["nb_train_data_samples"] = args.nb_train_data_samples
    params["nb_test_data_samples"] = args.nb_test_data_samples
    params["data_path"] = args.data_path

    return params


def read_results(results_file) -> dict:
    """Read previous results from file if exists.

    Returns:
        dict: Previous results from former benchmark.
    """

    results_file.parent.mkdir(exist_ok=True)

    if results_file.exists():
        results = json.loads(results_file.read_text())

    else:
        results = {}

    return results
