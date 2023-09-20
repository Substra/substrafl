from copy import deepcopy

import torch
from common.data_managers import CamelyonDataset
from torch.utils.data import DataLoader

from substrafl.algorithms.pytorch import TorchFedAvgAlgo
from substrafl.index_generator import NpIndexGenerator


def get_weldon_fedavg(
    seed: int, learning_rate: float, num_workers: int, index_generator: NpIndexGenerator, model: torch.nn.Module
):
    """Generates a substrafl compatible model for the fed avg strategy

    Args:
        seed (int): Seed to fix the random generators (for reproducibility reasons)
        learning_rate (float): learning rate of the optimizer
        num_workers (int): number of worker to be used per torch.
        index_generator (NpIndexGenerator): index generator to be used by the algo.
        model (nn.Module): model template to be used by the algo.

    Returns:
        TorchFedAvgAlgo: To be submit to a substrafl execute experiment function.
    """

    # Model definition
    my_model = deepcopy(model)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)

    # Substrafl formatted Algo
    class MyAlgo(TorchFedAvgAlgo):
        def __init__(
            self,
        ):
            super().__init__(
                model=my_model,
                criterion=criterion,
                optimizer=optimizer,
                index_generator=index_generator,
                seed=seed,
                dataset=CamelyonDataset,
            )

        def _local_train(self, train_dataset):
            # The opener only give all the paths in x and nothin in y
            multiprocessing_context = None
            if num_workers != 0:
                multiprocessing_context = torch.multiprocessing.get_context("spawn")

            dataloader = DataLoader(
                train_dataset,
                batch_sampler=self._index_generator,
                num_workers=num_workers,
                multiprocessing_context=multiprocessing_context,
            )

            # Train the model
            for x_batch, y_batch in dataloader:
                # Forward pass
                y_pred = self._model(x_batch)

                # Compute Loss
                loss = self._criterion(y_pred, y_batch)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if self._scheduler is not None:
                    self._scheduler.step()

        def _local_predict(self, predict_dataset, predictions_path, return_predictions=False):
            multiprocessing_context = None
            if num_workers != 0:
                multiprocessing_context = torch.multiprocessing.get_context("spawn")

            dataloader = DataLoader(
                predict_dataset,
                batch_size=self._index_generator.batch_size,
                drop_last=False,
                num_workers=num_workers,
                multiprocessing_context=multiprocessing_context,
            )

            y_pred = []
            with torch.no_grad():
                for X in dataloader:
                    y_pred.append(self._model(X))

            y_pred = torch.cat(y_pred).numpy()

            self._save_predictions(y_pred, predictions_path)

    return MyAlgo()
