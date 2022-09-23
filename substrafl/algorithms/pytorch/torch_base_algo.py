import abc
import inspect
import logging
import os
import shutil
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union

import numpy as np
import torch

from substrafl.algorithms.algo import Algo
from substrafl.exceptions import BatchSizeNotFoundError
from substrafl.exceptions import DatasetSignatureError
from substrafl.exceptions import DatasetTypeError
from substrafl.exceptions import OptimizerValueError
from substrafl.index_generator import BaseIndexGenerator
from substrafl.remote.decorators import remote_data

logger = logging.getLogger(__name__)


class TorchAlgo(Algo):
    """Base TorchAlgo class, all the torch algo classes
    inherit from it.

    To implement a new strategy:

        - add the strategy specific parameters in the ``__init__``
        - implement the :py:func:`~substrafl.algorithms.pytorch.torch_base_algo.TorchAlgo.train`
          function: it must use the
          :py:func:`~substrafl.algorithms.pytorch.torch_base_algo.TorchAlgo._local_train` function, which can be
          overridden by the user and must contain as little strategy-specific code as possible
        - Reimplement the :py:func:`~substrafl.algorithms.pytorch.torch_base_algo.TorchAlgo._update_from_checkpoint`
          and :py:func:`~substrafl.algorithms.pytorch.torch_base_algo.TorchAlgo._get_state_to_save` functions to add
          strategy-specific variables to the local state
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.modules.loss._Loss,
        index_generator: Union[BaseIndexGenerator, None],
        dataset: torch.utils.data.Dataset,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        seed: Optional[int] = None,
        use_gpu: bool = True,
        *args,
        **kwargs,
    ):
        """The ``__init__`` functions is called at each call of the `train()` or `predict()` function
        For round>2, some attributes will then be overwritten by their previous states in the `load()` function,
        before the `train()` or `predict()` function is ran.
        """
        super().__init__(*args, **kwargs)

        if seed is not None:
            torch.manual_seed(seed)

        self._device = self._get_torch_device(use_gpu=use_gpu)

        self._model = model.to(self._device)
        self._optimizer = optimizer
        # Move the optimizer to GPU if needed
        # https://github.com/pytorch/pytorch/issues/8741#issuecomment-496907204
        if self._optimizer is not None:
            self._optimizer.load_state_dict(self._optimizer.state_dict())
        self._criterion = criterion
        self._scheduler = scheduler

        self._index_generator = index_generator
        self._dataset: torch.utils.data.Dataset = dataset
        self._check_torch_dataset()

    @property
    def model(self) -> torch.nn.Module:
        """Model exposed when the user downloads the model

        Returns:
            torch.nn.Module: model
        """
        return self._model

    @abc.abstractmethod
    def train(
        self,
        datasamples: Any,
        shared_state: Any = None,
    ) -> Any:
        # Must be implemented in the child class
        raise NotImplementedError()

    @remote_data
    def predict(self, datasamples: Any, shared_state: Any, predictions_path: os.PathLike = None) -> Any:
        """Executes the following operations:

            * Create the test torch dataset.
            * Execute and return the results of the ``self._local_predict`` method

        Args:
            datasamples (typing.Any): Input data
            shared_state (Any): Latest train task shared state (output of the train method)
        """

        # Create torch dataset
        predict_dataset = self._dataset(datasamples, is_inference=True)
        self._local_predict(predict_dataset=predict_dataset, predictions_path=predictions_path)

    def _save_predictions(self, predictions: torch.Tensor, predictions_path: os.PathLike):
        """Save the predictions under the numpy format.

        Args:
            predictions (torch.Tensor): predictions to save.
            predictions_path (os.PathLike): destination file to save predictions.
        """
        if predictions_path is not None:
            np.save(predictions_path, predictions)
            shutil.move(str(predictions_path) + ".npy", predictions_path)

    def _local_predict(self, predict_dataset: torch.utils.data.Dataset, predictions_path):
        """Executes the following operations:

            * Create the torch dataloader using the index generator batch size.
            * Sets the model to `eval` mode
            * Save the predictions using the
              :py:func:`~substrafl.algorithms.pytorch.torch_base_algo.TorchAlgo._save_predictions` function.

        Args:
            predict_dataset (torch.utils.data.Dataset): predict_dataset build from the x returned by the opener.

        Important:
            The onus is on the user to ``save`` the compute predictions. Substrafl provides the
            :py:func:`~substrafl.algorithms.pytorch.torch_base_algo.TorchAlgo._save_predictions` to do so.
            The user can load those predictions from a metric file with the command:
            ``y_pred = np.load(inputs['predictions'])``.

        Raises:
            BatchSizeNotFoundError: No default batch size have been found to perform local prediction.
                Please overwrite the predict function of your algorithm.
        """
        if self._index_generator is not None:
            predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=self._index_generator.batch_size)
        else:
            raise BatchSizeNotFoundError(
                "No default batch size has been found to perform local prediction. "
                "Please overwrite the _local_predict function of your algorithm."
            )

        self._model.eval()

        predictions = torch.Tensor([])
        with torch.inference_mode():
            for x in predict_loader:
                x = x.to(self._device)
                predictions = torch.cat((predictions, self._model(x)), 0)

        predictions = predictions.cpu().detach()
        self._save_predictions(predictions, predictions_path)

    def _local_train(
        self,
        train_dataset: torch.utils.data.Dataset,
    ):
        """Local train method. Contains the local training loop.

        Train the model on ``num_updates`` minibatches, using the ``self._index_generator generator`` as batch sampler
        for the torch dataset.

        Args:
            train_dataset (torch.utils.data.Dataset): train_dataset build from the x and y returned by the opener.

        Important:

            You must use ``next(self._index_generator)`` as batch sampler,
            to ensure that the batches you are using are correct between 2 rounds
            of the federated learning strategy.

        Example:

            .. code-block:: python

                # Create torch dataloader
                train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=self._index_generator)

                for x_batch, y_batch in train_data_loader:

                    # Forward pass
                    y_pred = self._model(x_batch)

                    # Compute Loss
                    loss = self._criterion(y_pred, y_batch)
                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()

                    if self._scheduler is not None:
                        self._scheduler.step()
        """
        if self._optimizer is None:
            raise OptimizerValueError(
                "No optimizer found. Either give one or overwrite the _local_train method from the used torch"
                "algorithm."
            )

        # Create torch dataloader
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=self._index_generator)

        for x_batch, y_batch in train_data_loader:

            x_batch = x_batch.to(self._device)
            y_batch = y_batch.to(self._device)

            # Forward pass
            y_pred = self._model(x_batch)

            # Compute Loss
            loss = self._criterion(y_pred, y_batch)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            if self._scheduler is not None:
                self._scheduler.step()

    def _get_torch_device(self, use_gpu: bool) -> torch.device:
        """Get the torch device, CPU or GPU, depending
        on availability and user input.

        Args:
            use_gpu (bool): whether to use GPUs if available or not.

        Returns:
            torch.device: Torch device
        """
        device = torch.device("cpu")
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        return device

    def _update_from_checkpoint(self, path: Path) -> dict:
        """Load the checkpoint and update the internal state
        from it.
        Pop the values from the checkpoint so that we can ensure that it is empty at the
        end, ie all the values have been used.

        Args:
            path (pathlib.Path): path where the checkpoint is saved

        Returns:
            dict: checkpoint

        Example:

            .. code-block:: python

                def _update_from_checkpoint(self, path: Path) -> dict:
                    checkpoint = super()._update_from_checkpoint(path=path)
                    self._strategy_specific_variable = checkpoint.pop("strategy_specific_variable")
                    return checkpoint
        """
        assert path.is_file(), f'Cannot load the model - does not exist {list(path.parent.glob("*"))}'
        checkpoint = torch.load(path, map_location=self._device)
        self._model.load_state_dict(checkpoint.pop("model_state_dict"))

        if self._optimizer is not None:
            self._optimizer.load_state_dict(checkpoint.pop("optimizer_state_dict"))

        if self._scheduler is not None:
            self._scheduler.load_state_dict(checkpoint.pop("scheduler_state_dict"))

        self._index_generator = checkpoint.pop("index_generator")

        if self._device == torch.device("cpu"):
            torch.set_rng_state(checkpoint.pop("rng_state").to(self._device))
        else:
            torch.cuda.set_rng_state(checkpoint.pop("rng_state").to("cpu"))

        return checkpoint

    def load(self, path: Path) -> "TorchAlgo":
        """Load the stateful arguments of this class.
        Child classes do not need to override that function.

        Args:
            path (pathlib.Path): The path where the class has been saved.

        Returns:
            TorchAlgo: The class with the loaded elements.
        """
        checkpoint = self._update_from_checkpoint(path=path)
        assert len(checkpoint) == 0, f"Not all values from the checkpoint have been used: {checkpoint.keys()}"
        return self

    def _get_state_to_save(self) -> dict:
        """Create the algo checkpoint: a dictionary
        saved with ``torch.save``.
        In this algo, it contains the state to save for every strategy.
        Reimplement in the child class to add strategy-specific variables.

        Example:

            .. code-block:: python

                def _get_state_to_save(self) -> dict:
                    local_state = super()._get_state_to_save()
                    local_state.update({
                        "strategy_specific_variable": self._strategy_specific_variable,
                    })
                    return local_state

        Returns:
            dict: checkpoint to save
        """
        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "index_generator": self._index_generator,
        }
        if self._optimizer is not None:
            checkpoint["optimizer_state_dict"] = self._optimizer.state_dict()

        if self._scheduler is not None:
            checkpoint["scheduler_state_dict"] = self._scheduler.state_dict()

        if self._device == torch.device("cpu"):
            checkpoint["rng_state"] = torch.get_rng_state()
        else:
            checkpoint["rng_state"] = torch.cuda.get_rng_state()

        return checkpoint

    def _check_torch_dataset(self):
        # Check that the given Dataset is not an instance
        try:
            issubclass(self._dataset, torch.utils.data.Dataset)
        except TypeError:
            raise DatasetTypeError(
                "``dataset`` should be non-instantiate torch.utils.data.Dataset class. "
                "This means that calling ``dataset(datasamples, is_inference=False)`` must "
                "returns a torch dataset object. "
                "You might have provided an instantiate dataset or an object of the wrong type."
            )

        # Check the signature of the __init__() function of the torch dataset class
        signature = inspect.signature(self._dataset.__init__)
        init_parameters = signature.parameters

        if "datasamples" not in init_parameters:
            raise DatasetSignatureError(
                "The __init__() function of the torch Dataset must contain datasamples as parameter."
            )
        elif "is_inference" not in init_parameters:
            raise DatasetSignatureError(
                "The __init__() function of the torch Dataset must contain is_inference as parameter."
            )

    def save(self, path: Path):
        """Saves all the stateful elements of the class to the specified path.
        Child classes do not need to override that function.

        Args:
            path (pathlib.Path): A path where to save the class.
        """
        torch.save(
            self._get_state_to_save(),
            path,
        )
        assert path.is_file(), f'Did not save the model properly {list(path.parent.glob("*"))}'

    def summary(self):
        """Summary of the class to be exposed in the experiment summary file.
        Implement this function in the child class to add strategy-specific variables. The variables
        must be json-serializable.

        Example:

            .. code-block:: python

                def summary(self):
                    summary = super().summary()
                    summary.update(
                        "strategy_specific_variable": self._strategy_specific_variable,
                    )
                    return summary

        Returns:
            dict: a json-serializable dict with the attributes the user wants to store
        """
        summary = super().summary()
        summary.update(
            {
                "model": str(type(self._model)),
                "criterion": str(type(self._criterion)),
                "optimizer": None
                if self._optimizer is None
                else {
                    "type": str(type(self._optimizer)),
                    "parameters": self._optimizer.defaults,
                },
                "scheduler": None if self._scheduler is None else str(type(self._scheduler)),
            }
        )
        return summary
