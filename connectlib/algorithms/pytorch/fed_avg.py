import abc
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type

import numpy as np
import torch

from connectlib.algorithms import Algo
from connectlib.exceptions import IndexGeneratorUpdateError
from connectlib.index_generator import BaseIndexGenerator
from connectlib.index_generator import NpIndexGenerator
from connectlib.remote import remote_data

from . import weight_manager

logger = logging.getLogger(__name__)

# TODO/INFO : for the next strategy, all methods and args of this class could be wrapped into a generic TorchAlgo
# class. Every strategy class could inherit from it.


class TorchFedAvgAlgo(Algo):
    """To be inherited. Wraps the necessary operation so a torch model can be trained in the Federated Averaging strategy.
    The child class must at least defines : `model`, `criterion`, `optimizer`, `num_updates` as arguments of the
    :func:`super().__init__` function within the `__init__` method of the child class.
    E.g.:

    ```python
    class MyAlgo(TorchFedAvgAlgo):
        def __init__(
            self,
        ):
            super().__init__(
                model=perceptron,
                criterion=torch.nn.MSELoss(),
                optimizer=optimizer,
                num_updates=100,
            )

    my_algo = MyAlgo()
    algo_deps = Dependency(pypi_dependencies=["torch", "numpy"])
    strategy = FedAVG()
    ```

    It will inherit of the following default arguments : `get_index_generator = NpIndexGenerator`, `scheduler = None`,
    `batch_size = None` and `with_batch_norm_parameters = False`
    which can be overwritten as arguments of the :func:`super().__init__` function within the `__init__` method of
    the child class. E.g.:

    ```python
    class MyAlgo(TorchFedAvgAlgo):
        def __init__(
            self,
        ):
            super().__init__(
                model=perceptron,
                criterion=torch.nn.MSELoss(),
                optimizer=optimizer,
                num_updates=100,
                batch_size = 128,
            )
    ```

    It will inherit of the following default methods : `train`, `predict`, `load`
    and `save` which can be overwritten in the child class.
    It must define the `_local_train` and `_local_predict` methods.

    The `train` method updates the weights of the model with the aggregated weights, initialises or
    loads the index generator, calls the `_local_train` method to do the local training then gets
    the weight updates from the models and sends them to the aggregator.
    The `predict` method calls the `_local_predict` method to generate the predictions.

    ```python
    class MyAlgo(TorchFedAvgAlgo):
        def __init__(
            self,
        ):
            super().__init__(
                model=perceptron,
                criterion=torch.nn.MSELoss(),
                optimizer=optimizer,
                num_updates=100,
            )
        def _local_train(
            self,
            x: Any,
            y: Any,
        ):
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

        def _local_predict(self, x: Any) -> Any:
            with torch.inference_mode():
                y = self._model(x)
            return y
    ```

    The algo needs to get the number of samples in the dataset from the x sent by the opener.
    By default, it uses len(x). If that is not the proper way of getting the number of samples,
    override the `_get_len_from_x` function:

    ```python
    class MyAlgo(TorchFedAvgAlgo):
        def _get_len_from_x(self, x):
            return len(x)
    ```

    As development tools, the `train` and `predict` method comes with a default argument : _skip.
    If _skip is set to True, only the function will be executed and not all the code related to connect.
    This allows to quickly debug code and use the defined algorithm as is.

    Attributes:
        model (torch.nn.Modules): A torch model.
        criterion (torch.nn.modules.loss._Loss): A torch criterion (loss).
        optimizer (torch.optim.Optimizer): A torch optimizer linked to the model.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional). A torch scheduler that will be called at every
        batch. If None, no scheduler will be used. Defaults to None.
        num_updates (int): The number of times the model will be trained at each step of the strategy (i.e. of the
            train function).
        batch_size (int, optional). The number of samples used for each updates. If None, the whole input data will be
            used.
        get_index_generator (Type[BaseIndexGenerator], optional). A class returning a stateful index generator. Must
            inherit from BaseIndexGenerator. The __next__ method shall return a python object (batch_index) which
            is used for selecting each batch from the output of the _preprocess method during training in this way :
            `x[batch_index], y[batch_index]`. Defaults to NpIndexGenerator.
            If overridden, the generator class must be defined either as part of a package or in a different file
            than the one from which the `execute_experiment` function is called.
        with_batch_norm_parameters (bool). Whether to include the batch norm layer parameters in the fed avg strategy.
            Default to False.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        num_updates: int,
        batch_size: Optional[int],
        get_index_generator: Type[BaseIndexGenerator] = NpIndexGenerator,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        with_batch_norm_parameters: bool = False,
    ):
        """Initialize"""
        super().__init__()

        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._num_updates = num_updates
        self._get_index_generator = get_index_generator
        self._scheduler = scheduler
        self._batch_size = batch_size
        self._with_batch_norm_parameters = with_batch_norm_parameters
        self._index_generator: Optional[BaseIndexGenerator] = None

        if batch_size is None:
            logger.warning("Batch size is none, the whole dataset will be used for each update.")

    @property
    def model(self) -> torch.nn.Module:
        """Model exposed when the user downloads the model

        Returns:
            torch.nn.Module: model
        """
        return self._model

    @abc.abstractmethod
    def _local_train(
        self,
        x: Any,
        y: Any,
    ):
        """Local training loop

        Train the model on num_updates minibatches, using the
        self._index_generator generator to generate the batches.
        WARNING: you must use next(self._index_generator) at each minibatch,
        to ensure that you are using the batches are correct between 2 rounds
        of the federated learning strategy.

        Args:
            x (Any): x as returned by the opener
            y (Any): y as returned by the opener
        """
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
    def _local_predict(self, x: Any) -> Any:
        """Prediction on x

        Args:
            x (Any): x as returned by the opener

        Returns:
            Any: predictions in the format saved then loaded by the opener
            to calculate the metric
        """
        with torch.inference_mode():
            y = self._model(x)
        return y

    def _get_len_from_x(self, x: Any) -> int:
        """Get the length of the dataset from
        x as returned by the opener.

        Default: returns len(x). Override
        if needed.

        Args:
            x (Any): x returned by the opener
            get_X function

        Returns:
            int: Number of samples in the dataset
        """
        return len(x)

    @remote_data
    def train(
        self,
        x: Any,
        y: Any,
        shared_state=None,  # Set to None per default for clarity reason as the decorator will do it if
        # the arg shared_state is not passed.
    ) -> Dict[str, np.ndarray]:
        """Train method of the fed avg strategy implemented with torch. This method will execute the following
        operations:
            * instantiates the provided (or default) batch indexer
            * apply the provided (or default) _processing method to x and y
            * if a shared state is passed, set the parameters of the model to the provided shared state
            * train the model for n_updates
            * compute the weight update

        Args:
            x (Any): Input data.
            y (Any): Input target.
            shared_state (Dict[str, np.ndarray], optional): Dict containing torch parameters that will be set to the
                model. Defaults to None.

        Returns:
            Dict[str, np.ndarray]: weight update (delta between fine-tuned weights and previous weights)
        """

        # Instantiate the index_generator
        if self._index_generator is None:
            self._index_generator = self._get_index_generator(
                n_samples=self._get_len_from_x(x),
                batch_size=self._batch_size,
                num_updates=self._num_updates,
                shuffle=True,
                drop_last=False,
            )
        self._index_generator.reset_counter()

        # The shared states is the average of the difference of the gradient for all nodes
        # Hence we need to add it to the previous local state parameters
        if shared_state is not None:
            weight_manager.increment_parameters(
                model=self._model,
                updates=shared_state.values(),
                with_batch_norm_parameters=self._with_batch_norm_parameters,
            )

        old_parameters = weight_manager.get_parameters(
            model=self._model, with_batch_norm_parameters=self._with_batch_norm_parameters
        )

        # Train mode for torch model
        self._model.train()

        # Train the model
        self._local_train(
            x=x,
            y=y,
        )

        if self._index_generator.counter != self._num_updates:
            raise IndexGeneratorUpdateError(
                "The batch index generator has not been updated properly, it was called"
                f" {self._index_generator.counter} times against {self._num_updates}"
                " expected, please use self._index_generator to generate the batches."
            )

        self._model.eval()

        model_gradient = weight_manager.subtract_parameters(
            parameters=weight_manager.get_parameters(
                model=self._model,
                with_batch_norm_parameters=self._with_batch_norm_parameters,
            ),
            old_parameters=old_parameters,
        )

        # Re set to the previous state
        weight_manager.set_parameters(
            model=self._model,
            parameters=old_parameters,
            with_batch_norm_parameters=self._with_batch_norm_parameters,
        )

        return_dict = {f"grad_{i}": g.cpu().detach().numpy() for i, g in enumerate(model_gradient)}
        return_dict["n_samples"] = self._get_len_from_x(x)
        return return_dict

    @remote_data
    def predict(
        self,
        x: np.ndarray,
        shared_state: Dict[
            str, np.ndarray
        ] = None,  # Set to None per default for clarity reason as the decorator will do it if the arg shared_state
        # is not passed.
    ):
        """Predict method of the fed avg strategy. Executes the following operation :
            * apply user defined (or default) _process method to x
            * if a shared state is given, add it to the model parameters
            * apply the model to the input data (model(x))
            * apply user defined (or default) _postprocess method to the model results

        Args:
            x (np.ndarray): Input data.
            shared_state (Dict[str, np.ndarray], optional): If not None, the shared state will be added to the model
                parameters' before computing the predictions. Defaults to None.

        Returns:
            Any: Model prediction post precessed by the _postprocess class method.
        """
        # Reduce memory consumption as we don't use the model gradients
        with torch.inference_mode():
            # If needed, add the shared state to the model parameters
            if shared_state is not None:
                weight_manager.increment_parameters(
                    model=self._model,
                    updates=shared_state.values(),
                    with_batch_norm_parameters=self._with_batch_norm_parameters,
                )

            self._model.eval()

        predictions = self._local_predict(x)
        return predictions

    def load(self, path: Path):
        """Load the stateful arguments of this class, i.e.:
            * self._model
            * self._optimizer
            * self._scheduler (if provided)
            * self._index_generator
            * torch rng state

        Args:
            path (Path): The path where the class has been saved.

        Returns:
            Class: The class with the loaded elements.
        """
        assert path.is_file(), f'Cannot load the model - does not exist {list(path.parent.glob("*"))}'

        checkpoint = torch.load(path, map_location="cpu")
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self._scheduler is not None:
            self._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self._index_generator = checkpoint["index_generator"]

        torch.set_rng_state(checkpoint["rng_state"])
        return self

    def save(self, path: Path):
        """Saves all the stateful elements of the class to the specified path, i.e.:
            * self._model
            * self._optimizer
            * self._scheduler (if provided)
            * self._index_generator
            * torch rng state

        Args:
            path (Path): A path where to save the class.
        """
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "scheduler_state_dict": self._scheduler.state_dict() if self._scheduler is not None else None,
                "index_generator": self._index_generator,
                "rng_state": torch.get_rng_state(),
            },
            path,
        )
        assert path.is_file(), f'Did not save the model properly {list(path.parent.glob("*"))}'
