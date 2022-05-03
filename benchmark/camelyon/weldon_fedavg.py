import torch
from classic_algos.nn import Weldon
from common.data_managers import CamelyonDataset
from torch.utils.data import DataLoader

from connectlib.algorithms.pytorch import TorchFedAvgAlgo
from connectlib.index_generator import NpIndexGenerator


def get_weldon_fedavg(seed: int, batch_size: int, learning_rate: float, n_local_steps: int, num_workers: int):
    """Generates a connectlib compatible model for the fed avg strategy

    Args:
        seed (int): Seed to fix the random generators (for reproducibility reasons)
        batch_size (int): Batch size to be used during the experiment
        learning_rate (float): learning rate of the optimizer
        n_local_steps (int): number of updates to perform at each step of the strategy.
        num_workers (int): number of worker to be used per torch.

    Returns:
        TorchFedAvgAlgo: To be submit to a connectlib execute experiment function.
    """

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

    nig = NpIndexGenerator(
        batch_size=batch_size,
        num_updates=n_local_steps,
        shuffle=True,
        drop_last=True,
        seed=42,
    )

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
                index_generator=nig,
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
            with torch.no_grad():
                for X, _ in dataloader:
                    y_pred.append(self._model(X)[0].reshape(-1))

            y_pred = torch.sigmoid(torch.cat(y_pred)).numpy()

            return y_pred

    return MyAlgo()
