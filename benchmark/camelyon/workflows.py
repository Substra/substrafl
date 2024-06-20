import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import substra
import torch
import weldon_fedavg
from common import benchmark_metrics
from common.data_managers import CamelyonDataset
from common.data_managers import Data
from pure_substrafl import register_assets
from pure_substrafl.register_assets import get_clients
from pure_substrafl.register_assets import load_asset_keys
from pure_substrafl.register_assets import save_asset_keys
from pure_torch.strategies import basic_fed_avg
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from substrafl import execute_experiment
from substrafl.dependency import Dependency
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.index_generator import NpIndexGenerator
from substrafl.strategies import FedAvg


def substrafl_fed_avg(
    train_folder: Path,
    test_folder: Path,
    nb_train_data_samples: int,
    nb_test_data_samples: int,
    mode: str,
    seed: int,
    n_centers: int,
    learning_rate: int,
    n_rounds: int,
    num_workers: int,
    index_generator: NpIndexGenerator,
    model: torch.nn.Module,
    credentials_path: Path,
    asset_keys_path: Path,
    cp_name: Optional[str],
    cancel_cp: bool = False,
    torch_gpu: bool = False,
) -> benchmark_metrics.BenchmarkResults:
    """Execute Weldon algorithm for a fed avg strategy with substrafl API.

    Args:
        train_folder (Path): Path to the data sample that will be used and duplicate for the benchmark.
        test_folder (Path):  Path to the data sample that will be used and duplicate for the benchmark.
        nb_train_data_samples (int): Number of data samples to run the experiment with. Each data sample is
            a duplicate of the passed train folder.
        nb_test_data_samples (int): Number of data samples to run the experiment with. Each data sample is
            a duplicate of the passed test folder.
        mode (str): The Substra execution mode, must be either subprocess, docker, remote.
        seed (int): Random seed.
        n_centers (int): Number of centers to be used for the fed avg strategy.
        learning_rate (int): Learning rate to be used.
        n_rounds (int): Number of rounds for the strategy to be executed.
        num_workers (int): Number of workers for the torch data loader.
        index_generator (NpIndexGenerator): index generator to be used by the algo.
        model (nn.Module): model template to be used by the algo.
        credentials_path (Path): Remote only: file to Substra credentials configuration path.
        asset_keys_path (Path): Remote only: path to asset key file. If un existent, it will be created.
            Otherwise, all present keys in this fill will be reused per Substra in remote mode.
        cp_name ben): (Optional[str]): Compute Plan name to display
        cancel_cp (bool): if set to True, the CP will be canceled as soon as it's registered. Only work for remote mode.
        torch_gpu (bool): Use GPU default index for pytorch
    Returns:
        dict: Results of the experiment.
    """
    clients = get_clients(credentials=credentials_path, mode=mode, n_centers=n_centers)
    asset_keys = load_asset_keys(asset_keys_path, mode)

    # Substrafl asset registration
    train_data_nodes = register_assets.get_train_data_nodes(
        clients=clients, train_folder=train_folder, asset_keys=asset_keys, nb_data_sample=nb_train_data_samples
    )
    test_data_nodes = register_assets.get_test_data_nodes(
        clients=clients, test_folder=test_folder, asset_keys=asset_keys, nb_data_sample=nb_test_data_samples
    )

    aggregation_node = register_assets.get_aggregation_node(client=clients[0])

    if mode == "remote":
        save_asset_keys(asset_keys_path, asset_keys)

    my_algo = weldon_fedavg.get_weldon_fedavg(
        seed=seed, learning_rate=learning_rate, num_workers=num_workers, index_generator=index_generator, model=model
    )

    pypi_dependencies = [
        "numpy==1.24.3",
        "torch==2.0.1",
        "scikit-learn==1.3.1",
    ]
    if not torch_gpu:
        pypi_dependencies += ["--extra-index-url https://download.pytorch.org/whl/cpu"]

    # Dependencies
    base = Path(__file__).parent
    dependencies = Dependency(
        pypi_dependencies=pypi_dependencies,
        local_code=[base / "common", base / "weldon_fedavg.py"],
        # Keeping editable_mode=True to ensure nightly test benchmarks are ran against main substrafl git ref
        editable_mode=True,
        compile=True,
    )

    # Metrics
    def auc(data_from_opener, predictions):
        """AUC"""

        y_true = data_from_opener.y_true
        return roc_auc_score(y_true, predictions) if len(set(y_true)) > 1 else 0

    def accuracy(data_from_opener, predictions):
        """Accuracy"""

        y_true = data_from_opener.y_true
        return accuracy_score(y_true, np.round(predictions)) if len(set(y_true)) > 1 else 0

    # Custom Strategy used for the data loading (from custom_torch_function.py file)
    strategy = FedAvg(algo=my_algo, metric_functions={"ROC AUC": auc, "Accuracy": accuracy})

    # Evaluation strategy
    evaluation = EvaluationStrategy(test_data_nodes=test_data_nodes, eval_rounds=[n_rounds])

    # Launch experiment
    compute_plan = execute_experiment(
        # Choosing first client because usually builder is on Org1
        client=clients[0],
        strategy=strategy,
        train_data_nodes=train_data_nodes,
        evaluation_strategy=evaluation,
        aggregation_node=aggregation_node,
        num_rounds=n_rounds,
        dependencies=dependencies,
        experiment_folder=Path(__file__).resolve().parent / "benchmark_cl_experiment_folder",
        name=cp_name,
    )

    if cancel_cp and clients[0].backend_mode == substra.BackendType.REMOTE:
        clients[0].cancel_compute_plan(key=compute_plan.key)

    clients[0].wait_compute_plan(key=compute_plan.key, raise_on_failure=True)

    performances = clients[1].get_performances(key=compute_plan.key)

    return benchmark_metrics.get_metrics_from_substra_performances(raw_performances=performances)


def torch_fed_avg(
    train_folder: Path,
    test_folder: Path,
    nb_train_data_samples: int,
    nb_test_data_samples: int,
    seed: int,
    n_centers: int,
    learning_rate: int,
    n_rounds: int,
    num_workers: int,
    index_generator: NpIndexGenerator,
    model: torch.nn.Module,
) -> benchmark_metrics.BenchmarkResults:
    """Execute Weldon algorithm for a fed avg strategy implemented in pure torch and python.

    Args:
        train_folder (Path): Path to the data sample that will be used and duplicate for the benchmark.
        test_folder (Path):  Path to the data sample that will be used and duplicate for the benchmark.
        nb_train_data_samples (int): Number of data samples to run the experiment with. Each data sample is
            a duplicate of the passed train folder.
        nb_test_data_samples (int): Number of data samples to run the experiment with. Each data sample is
            a duplicate of the passed test folder.
        seed (int): Random seed.
        n_centers (int): Number of centers to be used for the fed avg strategy.
        learning_rate (int): Learning rate to use.
        n_rounds (int): Number of rounds for the strategy to be executed.
        num_workers (int): Number of workers for the torch dataloader.
        index_generator (NpIndexGenerator): index generator to be used by the algo.
        model (nn.Module): model template to be used by the algo.

    Returns:
        Tuple[float, dict]: Result of the experiment and more details on the speed.
    """
    start = time.time()

    train_camelyon = Data(paths=[train_folder] * nb_train_data_samples)
    train_datasets = [CamelyonDataset(data_from_opener=train_camelyon) for _ in range(n_centers)]

    batch_samplers = list()
    for train_dataset in train_datasets:
        batch_sampler = deepcopy(index_generator)
        batch_sampler.n_samples = len(train_dataset)
        batch_samplers.append(batch_sampler)

    multiprocessing_context = None
    if num_workers != 0:
        multiprocessing_context = torch.multiprocessing.get_context("spawn")

    train_dataloaders = [
        DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            multiprocessing_context=multiprocessing_context,
        )
        for batch_sampler, train_dataset in zip(batch_samplers, train_datasets)
    ]

    test_camelyon = Data(paths=[test_folder] * nb_test_data_samples)

    test_datasets = [CamelyonDataset(data_from_opener=test_camelyon) for _ in range(n_centers)]

    batch_samplers = list()
    for test_dataset in test_datasets:
        batch_sampler = deepcopy(index_generator)
        batch_sampler.n_samples = len(test_dataset)
        batch_samplers.append(batch_sampler)

    test_dataloaders = [
        DataLoader(
            test_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            multiprocessing_context=multiprocessing_context,
        )
        for batch_sampler, test_dataset in zip(batch_samplers, test_datasets)
    ]

    # Models definition
    models = []
    # Each model must be instantiated with the same parameters
    for _ in range(n_centers):
        models.append(deepcopy(model))

    criteria = [torch.nn.BCELoss() for _ in range(n_centers)]
    optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate) for model in models]

    basic_fed_avg(
        nets=models,
        optimizers=optimizers,
        criteria=criteria,
        dataloaders_train=train_dataloaders,
        num_rounds=n_rounds,
        batch_samplers=batch_samplers,
    )

    performances = {}

    with torch.no_grad():
        for k, test_dataloader in enumerate(tqdm(test_dataloaders, desc="predict: ")):
            y_pred = []
            y_true = np.array([])
            for X, y in test_dataloader:
                y_pred.append(models[k](X))
                y_true = np.append(y_true, y.numpy())

            # Fusion, sigmoid and to numpy
            y_pred = torch.cat(y_pred).numpy()
            auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0
            acc = accuracy_score(y_true, np.round(y_pred)) if len(set(y_true)) > 1 else 0
            performances.update({str(k): {"ROC AUC": auc, "Accuracy": acc}})

    metrics = benchmark_metrics.BenchmarkResults(name="torch", exec_time=time.time() - start, performances=performances)
    return metrics
