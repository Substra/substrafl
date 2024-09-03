import argparse
import datetime
import json
from pathlib import Path

import substra
import substratools

import substrafl


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
        default=Path(__file__).resolve().parents[1] / "data",
        help="Path to the data",
    )
    parser.add_argument(
        "--cancel-cp",
        action="store_true",
        default=False,
        help="Remote only: cancel the CP after registration",
    )
    parser.add_argument("--use-gpu", action="store_true", help="Use PyTorch with GPU/CUDA support")
    parser.add_argument(
        "--skip-pure-torch",
        action="store_true",
        help="Skip the pure torch computation part to only test substrafl implementation",
    )
    parser.add_argument(
        "--cp-name",
        type=str,
        default=None,
        help="Compute Plan name to display",
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
    params["cancel_cp"] = args.cancel_cp
    params["use_gpu"] = args.use_gpu
    params["skip_pure_torch"] = args.skip_pure_torch
    params["cp_name"] = args.cp_name

    return params


def load_benchmark_summary(file: Path, experiment_summary: dict, n_experiment_limit: int = 10) -> None:
    """
    Load benchmark summary and results.

    Args:
        file (Path): result filepath
        experiment_summary (dict): benchmark summary
        n_experiment_limit (int): limit of experiment to keep in the result file

    """
    n_experiment_limit = -1 * max(n_experiment_limit, 0)
    file.parent.mkdir(exist_ok=True)
    experiment_summary = {datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"): experiment_summary}
    experiment_summary = (json.loads(file.read_text()) if file.exists() else []) + [experiment_summary]
    file.write_text(json.dumps(experiment_summary[n_experiment_limit:], sort_keys=True, indent=4))
