from copy import deepcopy

import torch
from common.data_managers import CamelyonDataset
from torch.utils.data import DataLoader

from connectlib.algorithms.pytorch import TorchFedAvgAlgo
from connectlib.index_generator import NpIndexGenerator
from connectlib.remote import remote_data


def get_weldon_fedavg(
    seed: int, learning_rate: float, num_workers: int, index_generator: NpIndexGenerator, model: torch.nn.Module
):
    """Generates a connectlib compatible model for the fed avg strategy

    Args:
        seed (int): Seed to fix the random generators (for reproducibility reasons)
        learning_rate (float): learning rate of the optimizer
        num_workers (int): number of worker to be used per torch.
        index_generator (NpIndexGenerator): index generator to be used by the algo.
        model (nn.Module): model template to be used by the algo.

    Returns:
        TorchFedAvgAlgo: To be submit to a connectlib execute experiment function.
    """

    # Connectlib will instantiate each center with a copy of this model, hence we need to set the seed at
    # initialization
    torch.manual_seed(seed)

    # Model definition
    my_model = deepcopy(model)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Connectlib formatted Algo
    class MyAlgo(TorchFedAvgAlgo):
        def __init__(
            self,
        ):
            torch.manual_seed(seed)
            super().__init__(
                model=my_model,
                criterion=criterion,
                optimizer=optimizer,
                index_generator=index_generator,
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
                y_pred = self._model(x_batch)[0].reshape(-1)

                # Compute Loss
                loss = self._criterion(y_pred, y_batch)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if self._scheduler is not None:
                    self._scheduler.step()

        @remote_data
        def predict(self, x, shared_state):
            dataset = CamelyonDataset(x=x)

            multiprocessing_context = None
            if num_workers != 0:
                multiprocessing_context = torch.multiprocessing.get_context("spawn")

            dataloader = DataLoader(
                dataset,
                batch_size=self._index_generator._batch_size,
                drop_last=False,
                num_workers=num_workers,
                multiprocessing_context=multiprocessing_context,
            )

            y_pred = []
            with torch.no_grad():
                for X, _ in dataloader:
                    y_pred.append(self._model(X)[0].reshape(-1))

            y_pred = torch.sigmoid(torch.cat(y_pred)).numpy()

            return y_pred

    return MyAlgo()
