from pathlib import Path
from typing import List

import numpy as np
import substra
import torch
from classic_algos.nn import Weldon
from common.data_managers import CamelyonDataset
from pure_connectlib import register_assets
from pure_torch.strategies import basic_fed_avg
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from connectlib import execute_experiment
from connectlib.algorithms.pytorch import TorchFedAvgAlgo
from connectlib.dependency import Dependency
from connectlib.evaluation_strategy import EvaluationStrategy
from connectlib.index_generator import NpIndexGenerator
from connectlib.strategies import FedAVG


def connectlib_fed_avg(
    trains_folders: List[Path],
    test_folder: Path,
    seed: int,
    batch_size: int,
    n_centers: int,
    learning_rate: int,
    n_rounds: int,
    n_local_steps: int,
    num_workers: int,
) -> dict:
    """Execute Weldon algorithm for a fed avg strategy with connectlib API.

    Args:
        trains_folders (List[Path]): List of the used folders to train the data. There should be one folder
            per center ending with `train_k` where k is the node number. Those folder can be generated with the
            register_assets.split_dataset function.
        test_folder (Path): The folder containing the test data.
        seed (int): Random seed.
        batch_size (int): Batch size to use for the training.
        n_centers (int): Number of centers to be used for the fed avg strategy.
        learning_rate (int): Learning rate to be used.
        n_rounds (int): Number of rounds for the strategy to be executed.
        n_local_steps (int): Number of updates for each step of the strategy.
        num_workers (int): Number of workers for the torch dataloader.

    Returns:
        dict: Results of the experiment.
    """
    # Debug clients
    clients = [substra.Client(debug=True)] * n_centers

    # Connectlib asset registration
    train_data_nodes = register_assets.get_train_data_nodes(clients=clients, trains_folders=trains_folders)
    test_data_nodes = [register_assets.get_test_data_node(client=clients[0], test_folder=test_folder)]
    aggregation_node = register_assets.get_aggregation_node()

    # Connectlib will instantiate each center with a copy of this model, hence we need to set the seed at
    # initialization
    torch.manual_seed(seed)

    # Model definition
    model = Weldon(
        in_features=2048,
        out_features=1,
        n_extreme=10,
        n_top=10,
        n_bottom=10,
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Connectlib formatted Algo
    class MyAlgo(TorchFedAvgAlgo):
        def __init__(
            self,
        ):
            torch.manual_seed(seed)
            super().__init__(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                num_updates=n_local_steps,
                batch_size=batch_size,
                get_index_generator=NpIndexGenerator,
            )

        def _local_train(self, x, y):
            # The opener only give all the paths in x and nothin in y
            dataset = CamelyonDataset(data_indexes=x.indexes, img_path=x.path)

            multiprocessing_context = None
            if num_workers != 0:
                multiprocessing_context = torch.multiprocessing.get_context("spawn")

            dataloader = DataLoader(
                dataset,
                batch_sampler=self._index_generator,
                num_workers=num_workers,
                multiprocessing_context=multiprocessing_context,
            )

            # Train the model
            for x_batch, y_batch in dataloader:

                # Forward pass
                y_pred = self._model(x_batch)[0].reshape(-1)

                # Compute Loss
                loss = self._criterion(y_pred, y_batch)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if self._scheduler is not None:
                    self._scheduler.step()

        def _local_predict(self, x):
            dataset = CamelyonDataset(data_indexes=x.indexes, img_path=x.path)

            multiprocessing_context = None
            if num_workers != 0:
                multiprocessing_context = torch.multiprocessing.get_context("spawn")

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=False,
                num_workers=num_workers,
                multiprocessing_context=multiprocessing_context,
            )

            y_pred = []
            y_true = np.array([])
            with torch.no_grad():
                for X, y in dataloader:
                    y_pred.append(self._model(X)[0].reshape(-1))
                    y_true = np.append(y_true, y.numpy())

            y_pred = torch.sigmoid(torch.cat(y_pred)).numpy()

            return y_pred

    my_algo = MyAlgo()

    # Algo dependencies
    # Classic algos must be installed locally in editable mode
    # for the Dockerfile mode
    base = Path(__file__).parent
    algo_deps = Dependency(
        pypi_dependencies=["torch", "numpy", "sklearn"],
        local_code=[
            base / "common" / "data_managers.py",
        ],
        local_dependencies=[base / "classic_algos-1.6.0-py3-none-any.whl"],
    )

    # Custom Strategy used for the data loading (from custom_torch_algo.py file)
    strategy = FedAVG()

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

    # Read the results from saved performances
    testtuples = clients[1].list_testtuple(filters=[f"testtuple:compute_plan_key:{compute_plan.key}"])

    # Resetting the clients
    # BUG: if we don't reset clients, substra tries to find an algo that do not exist
    # TODO: investigate
    del clients

    # Returning performances
    return list(testtuples[0].test.perfs.values())[0]


def torch_fed_avg(
    trains_folders: List[Path],
    test_folder: Path,
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
        trains_folders (List[Path]): List of the used folders to train the data. There should be one folder
            per center ending with `train_k` where k is the number of the node. Those folder can be generated with the
            register_assets.split_dataset function.
        test_folder (Path): The folder containing the test data.
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
        for train_folder in trains_folders
    ]
    batch_samplers = [
        NpIndexGenerator(
            n_samples=len(train_dataset),
            batch_size=batch_size,
            num_updates=n_local_steps,
            shuffle=True,
            drop_last=True,
            seed=42,
        )
        for train_dataset in train_datasets
    ]

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

    test_dataset = CamelyonDataset(
        data_indexes=np.loadtxt(Path(test_folder) / "index.csv", delimiter=",", dtype=str),
        img_path=test_folder,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        multiprocessing_context=multiprocessing_context,
    )

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

    y_pred = []
    y_true = np.array([])

    with torch.no_grad():
        for X, y in test_dataloader:
            y_pred.append(models[0](X)[0].reshape(-1))
            y_true = np.append(y_true, y.numpy())

    # Fusion, sigmoid and to numpy
    y_pred = torch.sigmoid(torch.cat(y_pred)).numpy()
    metric = roc_auc_score(y_true, y_pred)

    return metric
