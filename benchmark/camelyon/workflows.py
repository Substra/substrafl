import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import weldon_fedavg
from classic_algos.nn import Weldon
from common.data_managers import CamelyonDataset
from pure_connectlib import register_assets
from pure_connectlib.register_assets import get_clients
from pure_connectlib.register_assets import load_assets_keys
from pure_connectlib.register_assets import save_assets_keys
from pure_torch.strategies import basic_fed_avg
from sklearn.metrics import roc_auc_score
from substra.sdk.models import ComputePlanStatus
from torch.utils.data import DataLoader
from tqdm import tqdm

from connectlib import execute_experiment
from connectlib.dependency import Dependency
from connectlib.evaluation_strategy import EvaluationStrategy
from connectlib.index_generator import NpIndexGenerator
from connectlib.strategies import FedAvg


def connectlib_fed_avg(
    train_folders: List[Path],
    test_folders: Path,
    mode: str,
    seed: int,
    batch_size: int,
    n_centers: int,
    learning_rate: int,
    n_rounds: int,
    n_local_steps: int,
    num_workers: int,
    credentials_path: Path,
    assets_keys_path: Path,
) -> dict:
    """Execute Weldon algorithm for a fed avg strategy with connectlib API.

    Args:
        train_folders (List[Path]): List of the used folders to train the data. There should be one folder
            per center ending with `train_k` where k is the node number. Those folder can be generated with the
            register_assets.split_dataset function.
        test_folder (Path): The folder containing the test data.
        mode (str): The connect execution mode, must be either subprocess, docker, remote.
        seed (int): Random seed.
        batch_size (int): Batch size to use for the training.
        n_centers (int): Number of centers to be used for the fed avg strategy.
        learning_rate (int): Learning rate to be used.
        n_rounds (int): Number of rounds for the strategy to be executed.
        n_local_steps (int): Number of updates for each step of the strategy.
        num_workers (int): Number of workers for the torch data loader.
        credentials_path (Path): Remote only: file to connect credentials configuration path.
        assets_keys_path (Path): Remote only; path to asset key file. If un existent, it will be created.
            Otherwise, all present keys in this fill will be reused per connect in remote mode.

    Returns:
        dict: Results of the experiment.
    """

    clients = get_clients(credentials=credentials_path, mode=mode, n_centers=n_centers)
    assets_keys = load_assets_keys(assets_keys_path, mode)

    # Connectlib asset registration
    train_data_nodes = register_assets.get_train_data_nodes(
        clients=clients, train_folders=train_folders, assets_keys=assets_keys
    )
    test_data_nodes = register_assets.get_test_data_nodes(
        clients=clients, test_folders=test_folders, assets_keys=assets_keys
    )
    aggregation_node = register_assets.get_aggregation_node(client=clients[0])

    if mode == "remote":
        save_assets_keys(assets_keys_path, assets_keys)

    my_algo = weldon_fedavg.get_weldon_fedavg(
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_local_steps=n_local_steps,
        num_workers=num_workers,
    )

    # Algo dependencies
    # Classic algos must be installed locally in editable mode
    # for the Dockerfile mode
    base = Path(__file__).parent
    algo_deps = Dependency(
        pypi_dependencies=["torch", "numpy", "sklearn"],
        local_code=[base / "common" / "data_managers.py", base / "weldon_fedavg.py"],
        local_dependencies=[base / "classic_algos-1.6.0-py3-none-any.whl"],
        editable_mode=True,
    )

    # Custom Strategy used for the data loading (from custom_torch_algo.py file)
    strategy = FedAvg()

    # Evaluation strategy
    evaluation = EvaluationStrategy(test_data_nodes=test_data_nodes, rounds=[n_rounds])

    # Launch experiment
    compute_plan = execute_experiment(
        client=clients[1],
        algo=my_algo,
        strategy=strategy,
        train_data_nodes=train_data_nodes,
        evaluation_strategy=evaluation,
        aggregation_node=aggregation_node,
        num_rounds=n_rounds,
        dependencies=algo_deps,
        experiment_folder=Path(__file__).resolve().parent / "experiment_folder",
    )

    # Wait for the compute plan to finish
    # Read the results from saved performances
    running = True
    while running:
        if clients[0].get_compute_plan(compute_plan.key).status in (
            ComputePlanStatus.done.value,
            ComputePlanStatus.failed.value,
            ComputePlanStatus.canceled.value,
        ):
            running = False

        else:
            time.sleep(1)

    testtuples = clients[1].list_testtuple(filters=[f"testtuple:compute_plan_key:{compute_plan.key}"])

    # Resetting the clients
    # BUG: if we don't reset clients, substra tries to find an algo that do not exist
    # TODO: investigate
    del clients

    # Returning performances
    return {k: list(testtuple.test.perfs.values())[0] for k, testtuple in enumerate(testtuples)}


def torch_fed_avg(
    train_folders: List[Path],
    test_folders: List[Path],
    seed: int,
    batch_size: int,
    n_centers: int,
    learning_rate: int,
    n_rounds: int,
    n_local_steps: int,
    num_workers: int,
) -> float:
    """Execute Weldon algorithm for a fed avg strategy implemented in pure torch and python.

    Args:
        train_folders (List[Path]): List of the used folders to train the data. There should be one folder
            per center ending with `train_k` where k is the number of the node. Those folder can be generated with the
            register_assets.split_dataset function.
        test_folders (List[Path]): The folder containing the test data.
        seed (int): Random seed.
        batch_size (int): Batch size to use for the training.
        n_centers (int): Number of centers to be used for the fed avg strategy.
        learning_rate (int): Learning rate to use.
        n_rounds (int): Number of rounds for the strategy to be executed.
        n_local_steps (int): Number of updates for each step of the strategy.
        num_workers (int): Number of workers for the torch dataloader.

    Returns:
        Tuple[float, dict]: Result of the experiment and more details on the speed.
    """
    train_datasets = [
        CamelyonDataset(
            data_indexes=np.loadtxt(Path(train_folder) / "index.csv", delimiter=",", dtype=str),
            img_path=train_folder,
        )
        for train_folder in train_folders
    ]
    batch_samplers = list()
    for train_dataset in train_datasets:
        batch_sampler = NpIndexGenerator(
            batch_size=batch_size,
            num_updates=n_local_steps,
            shuffle=True,
            drop_last=True,
            seed=42,
        )
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

    test_datasets = [
        CamelyonDataset(
            data_indexes=np.loadtxt(Path(test_folder) / "index.csv", delimiter=",", dtype=str),
            img_path=test_folder,
        )
        for test_folder in test_folders
    ]

    test_dataloaders = [
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=num_workers,
            multiprocessing_context=multiprocessing_context,
        )
        for test_dataset in test_datasets
    ]

    # Models definition

    models = []
    # Each model must be instantiated with the same parameters
    for _ in range(n_centers):
        torch.manual_seed(seed=seed)
        models.append(
            Weldon(
                in_features=2048,
                out_features=1,
                n_extreme=10,
                n_top=10,
                n_bottom=10,
            )
        )

    criteria = [torch.nn.BCEWithLogitsLoss() for _ in range(n_centers)]
    optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate) for model in models]

    basic_fed_avg(
        nets=models,
        optimizers=optimizers,
        criteria=criteria,
        dataloaders_train=train_dataloaders,
        num_rounds=n_rounds,
        batch_samplers=batch_samplers,
    )

    metrics = {}

    with torch.no_grad():
        for k, test_dataloader in enumerate(tqdm(test_dataloaders, desc="predict: ")):
            y_pred = []
            y_true = np.array([])
            for X, y in test_dataloader:
                y_pred.append(models[k](X)[0].reshape(-1))
                y_true = np.append(y_true, y.numpy())

            # Fusion, sigmoid and to numpy
            y_pred = torch.sigmoid(torch.cat(y_pred)).numpy()
            metric = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0
            metrics.update({k: metric})

    return metrics
