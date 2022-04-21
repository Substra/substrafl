import argparse
import json

import substra
import substratools

import connectlib


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
    parser.add_argument("--n-centers", type=int, default=2)
    parser.add_argument("--sub-sampling", type=float, default=1)
    parser.add_argument("--n-rounds", type=int, default=11)
    parser.add_argument("--n-local-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mode", type=str, default="remote")
    parser.add_argument("--train-0", type=int)
    parser.add_argument("--test-0", type=int)
    parser.add_argument("--train-1", type=int)
    parser.add_argument("--test-1", type=int)
    parser.add_argument("--credentials", type=str, default="remote.yaml")
    parser.add_argument("--assets-keys", type=str, default="keys.json")

    args = parser.parse_args()
    params["n_centers"] = args.n_centers
    params["sub_sampling"] = args.sub_sampling
    params["n_rounds"] = args.n_rounds
    params["n_local_steps"] = args.n_local_steps
    params["batch_size"] = args.batch_size
    params["num_workers"] = args.num_workers
    params["mode"] = args.mode
    params["credentials"] = args.credentials
    params["assets_keys"] = args.assets_keys

    # Pass the exact size of the needed samples
    if args.train_0:
        params["data_samples_size"] = [
            {"train": args.train_0, "test": args.test_0},
            {"train": args.train_1, "test": args.test_1},
        ]

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
