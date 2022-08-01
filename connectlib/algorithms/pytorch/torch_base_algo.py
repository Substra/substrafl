import abc
import logging
from pathlib import Path
from typing import Any
from typing import Optional

import torch

from connectlib.algorithms.algo import Algo
from connectlib.exceptions import OptimizerValueError
from connectlib.index_generator import BaseIndexGenerator
from connectlib.remote.decorators import remote_data

logger = logging.getLogger(__name__)


class TorchAlgo(Algo):
    """Base TorchAlgo class, all the torch algo classes
    inherit from it.

    To implement a new strategy:

        - add the strategy specific parameters in the ``__init__``
        - implement the :py:func:`~connectlib.algorithms.pytorch.torch_base_algo.TorchAlgo.train`
          function: it must use the
          :py:func:`~connectlib.algorithms.pytorch.torch_base_algo.TorchAlgo._local_train` and
          :py:func:`~connectlib.algorithms.pytorch.torch_base_algo.TorchAlgo._local_predict` functions, which are
          overridden by the user and must contain as little strategy-specific code as possible
        - Reimplement the :py:func:`~connectlib.algorithms.pytorch.torch_base_algo.TorchAlgo._update_from_checkpoint`
          and :py:func:`~connectlib.algorithms.pytorch.torch_base_algo.TorchAlgo._get_state_to_save` functions to add
          strategy-specific variables to the local state
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.modules.loss._Loss,
        index_generator: BaseIndexGenerator,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_gpu: bool = True,
        *args,
        **kwargs,
    ):
        """The ``__init__`` functions is called at each call of the `train()` or `predict()` function
        For round>2, some attributes will then be overwritten by their previous states in the `load()` function,
        before the `train()` or `predict()` function is ran.
        """
        super().__init__(*args, **kwargs)

        self._device = self._get_torch_device(use_gpu=use_gpu)

        self._model = model.to(self._device)
        self._optimizer = optimizer
        # Move the optimizer to GPU if needed
        # https://github.com/pytorch/pytorch/issues/8741#issuecomment-496907204
        if self._optimizer is not None:
            self._optimizer.load_state_dict(self._optimizer.state_dict())
        self._criterion = criterion
        self._scheduler = scheduler

        self._index_generator: BaseIndexGenerator = index_generator

    @property
    def model(self) -> torch.nn.Module:
        """Model exposed when the user downloads the model

        Returns:
            torch.nn.Module: model
        """
        return self._model

    def _get_len_from_x(self, x: Any) -> int:
        """Get the length of the dataset from x as returned by the opener.

        Default: returns ``len(x)``. Overwrite if needed.

        Args:
            x (typing.Any): x returned by the opener
                get_X function

        Returns:
            int: Number of samples in the dataset
        """
        return len(x)

    @abc.abstractmethod
    def train(
        self,
        x: Any,
        y: Any,
        shared_state: Any = None,
    ) -> Any:
        # Must be implemented in the child class
        raise NotImplementedError()

    @remote_data
    def predict(
        self,
        x: Any,
        shared_state: Any,
    ):
        """Executes the following operations:

            * Sets the model to `eval` mode
            * Applies the `self._local_predict` function
            * Returns the predictions

        Args:
            x (typing.Any): Input data
            shared_state (Any): Latest train task shared state (output of the train method)

        Returns:
            typing.Any: Model prediction.
        """
        self._model.eval()
        predictions = self._local_predict(x)
        return predictions

    @abc.abstractmethod
    def _local_train(
        self,
        x: Any,
        y: Any,
    ):
        """Local train method, the user must override it, this function
        contains the local training loop with the data pre-processing.

        Train the model on ``num_updates`` minibatches, using the
        ``self._index_generator generator`` to generate the batches.

        Args:
            x (typing.Any): x as returned by the opener
            y (typing.Any): y as returned by the opener

        Important:

            You must use ``next(self._index_generator)`` at each minibatch,
            to ensure that the batches you are using are correct between 2 rounds
            of the federated learning strategy.

        Example:

            .. code-block:: python

                for batch_index in self._index_generator:
                    x_batch, y_batch = x[batch_index], y[batch_index]

                    # Do the pre-processing here

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

        for batch_index in self._index_generator:
            x_batch, y_batch = x[batch_index], y[batch_index]

            # Forward pass
            y_pred = self._model(x_batch)

            # Compute Loss
            loss = self._criterion(y_pred, y_batch)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            if self._scheduler is not None:
                self._scheduler.step()

    @abc.abstractmethod
    def _local_predict(
        self,
        x: Any,
    ) -> Any:
        """Local predict method, the user must override it. This function
        contains the local predict with the data pre-processing and
        post-processing.

        Args:
            x (typing.Any): x as returned by the opener

        Returns:
            typing.Any: predictions in the format saved then loaded by the opener
            to calculate the metric

        Example:

            .. code-block:: python

                with torch.inference_mode():
                    # Do the pre-processing here
                    y = self._model(x)
                    # Do the post-processing here
                return y
        """
        with torch.inference_mode():
            y = self._model(x)
        return y

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
