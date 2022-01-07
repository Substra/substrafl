import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch

from connectlib import NpIndexGenerator
from connectlib.algorithms import Algo
from connectlib.remote import remote_data

from . import weight_manager

logger = logging.getLogger(__name__)

# TODO/INFO : for the next strategy, all methods and args of this class could be wrapped into a generic TorchAlgo
# class. Every strategy class could inherit from it.


class TorchFedAvgAlgo(Algo):
    """To be inherited. Wrapped the necessary operation so a torch model can be trained in a Federated Learning strategy.
    The child class must at least defines : `model`, `criterion`, `optimizer`, `num_updates` as arguments of the :func:`super().__init__`
    function within the `__init__` method of the child class.
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

    compute_plan = execute_experiment(
        client=network.clients[0],
        algo=my_algo,
        strategy=strategy,
        train_data_nodes=train_linear_nodes,
        test_data_nodes=test_linear_nodes,
        aggregation_node=aggregation_node,
        num_rounds=num_rounds,
        dependencies=algo_deps,
    )
    ```

    It will inherit of the following default arguments : `index_generator = NpIndexGenerator`, `scheduler = None`, `batch_size = None` and `with_batch_norm_parameters = False`
    which can be overwritten as arguments of the :func:`super().__init__` function within the `__init__` method of the child class. E.g.:

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

    It will inherit of the following default methods : `_preprocess`, `_postprocess`, `train`, `predict`, `load`, and `save` which can be overwritten in the child class e.g.:

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

        def _preprocess(self, x: Any, y: Any = None) -> torch.Tensor:
            # convert numpy array to tensor.
            if y is not None:
                return (
                    torch.from_numpy(x).float(),
                    torch.from_numpy(y).float(),
                )
            else:
                return torch.from_numpy(x).float()

    ```

    As development tools, the `train` and `predict` method comes with a default argument : _skip.

    If _skip is set to True, only the function will be executed and not all the code related to connect.

    This allows to quickly debug code and use the defined algorithm as is.

    Attributes:
        model (torch.nn.Modules): A torch model.
        criterion (torch.nn.modules.loss._Loss): A torch criterion (loss).
        optimizer (torch.optim.Optimizer): A torch optimizer linked to the model.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional). A torch scheduler. If None, no scheduler will be used. Defaults to None.
        num_updates (int): The number of times the model will be trained at each step of the strategy (i.e. of the train function).
        batch_size (int, optional). The number of samples used for each updates. If None, the whole input data will be used. Defaults to None.
        get_index_generator (Callable, optional). A function or a class returning a state full index generator. Must expect two arguments: `n_samples` and `batch_size` and returns
            a deterministic generator exposing the __next__() method. This method shall return a python object (batch_index) which will be used for selecting each batch from the output of the _preprocess method
            during training in this way : `x[batch_index], y[batch_index]`.  Defaults to NpIndexGenerator.
        with_batch_norm_parameters (bool). Whether to include the batch norm layer parameters in the fed avg strategy. Default to False.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        num_updates: int,
        get_index_generator: Callable = NpIndexGenerator,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        batch_size: Optional[int] = None,
        with_batch_norm_parameters: bool = False,
        seed: int = 42,
    ):
        """Initialize"""
        super().__init__()
        torch.manual_seed(seed=seed)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_updates = num_updates
        self.get_index_generator = get_index_generator
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.with_batch_norm_parameters = with_batch_norm_parameters
        self.seed = seed
        self.index_generator = None

    # INFO: the two functions below are duplicated from the Algo class to get the specific
    # output type in the documentation.
    def _preprocess(
        self, x: Any, y: Any = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Executed at the beginning of each train and predict method. This loads the data if necessary and returns it as a torch tensor.


        When executing at the beginning of the train method, both x and y are passed.
        When executing at the beginning of the predict method, only x is passed and y is set to None.

        Args:
            x (Any): Data returned per the get_x method of the opener.
            y (Any, optional): Data returned per the get_y method of the opener. Defaults to None.

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: x, y or x as torch tensor.
        """
        if y is not None:
            return x, y

        return x

    def _postprocess(self, y_pred: torch.Tensor) -> Any:
        """Executed at the end of each predict method. As the predict method of the algorithm returns `model(x)`,
        this function transforms the predicted torch.Tensor to the expected format of the input of the `save_predictions`
        function of the opener. The predictions are then loaded and used to calculate the metric.

        Args:
            y_pred (torch.Tensor): The torch tensor of prediction (result of model(x)).

        Returns:
            Any: The predictions adapted to the metric input format.
        """
        return y_pred

    def _safe_preprocess(
        self, x: Any, y: Any = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Checks wether the user preprocessing function returns x, and/or y as a torch tensor.

        Args:
            x (Any): Data returned per the get_x method of the opener.
            y (Any, optional): Data returned per the get_y method of the opener. Defaults to None.

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: x, y or x as torch tensor.
        """
        if y is not None:
            x, y = self._preprocess(x=x, y=y)
            assert isinstance(
                y, torch.Tensor
            ), f"The output (y) of the _preprocess function should be a torch.Tensor but it is {type(y)}. Please overwrite it in your {self.__class__.__name__} class."

            assert isinstance(
                x, torch.Tensor
            ), f"The output (x) of the _preprocess function should be a torch.Tensor but it is {type(x)}. Please overwrite it in your {self.__class__.__name__} class."

            return x, y

        else:
            x = self._preprocess(x)
            assert isinstance(
                x, torch.Tensor
            ), f"The output (x) of the _preprocess function should be a torch.Tensor but it is {type(x)}. Please overwrite it in your {self.__class__.__name__} class."

            return x

    @remote_data
    def train(
        self,
        x: Any,
        y: Any,
        shared_state=None,  # Set to None per default for clarity reason as the decorator will do it if the arg shared_state is not passed.
    ) -> Dict[str, np.ndarray]:
        """Train method of the fed avg strategy implemented with torch. This method will execute the following operations:
            * instantiates the provided (or default) batch indexer
            * apply the provided (or default) _processing method to x and y
            * if a shared state is passed, set the parameters of the model to the provided shared state
            * train the model for n_updates
            * compute the gradients of the training

        Args:
            x (Any): Input data.
            y (Any): Input target.
            shared_state (Dict[str, np.ndarray], optional): Dict containing torch parameters that will be set to the model. Defaults to None.

        Returns:
            Dict[str, np.ndarray]: The gradients of the training.
        """

        # Ensure that we have the proper format
        x, y = self._safe_preprocess(x, y)

        # Instantiate the index_generator
        if self.index_generator is None:
            self.index_generator = self.get_index_generator(
                n_samples=x.shape[0], batch_size=self.batch_size
            )
        # Train mode for torch model
        self.model.train()
        # The shared states is the average of the difference of the gradient for all nodes
        # Hence we need to add it to the previous local state parameters
        if shared_state is not None:
            weight_manager.increment_parameters(
                model=self.model,
                gradients=shared_state.values(),
                with_batch_norm_parameters=self.with_batch_norm_parameters,
            )

        old_parameters = weight_manager.get_parameters(
            model=self.model, with_batch_norm_parameters=self.with_batch_norm_parameters
        )

        # Train the model
        for _ in range(self.num_updates):
            batch_index = next(self.index_generator)
            x_batch, y_batch = x[batch_index], y[batch_index]

            # Forward pass
            y_pred = self.model(x_batch)

            # Compute Loss
            loss = self.criterion(y_pred, y_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

        model_gradient = weight_manager.subtract_parameters(
            parameters=weight_manager.get_parameters(
                model=self.model,
                with_batch_norm_parameters=self.with_batch_norm_parameters,
            ),
            old_parameters=old_parameters,
        )

        # Re set to the previous state
        weight_manager.set_parameters(
            model=self.model,
            parameters=old_parameters,
            with_batch_norm_parameters=self.with_batch_norm_parameters,
        )

        return_dict = {
            f"grad_{i}": g.cpu().detach().numpy() for i, g in enumerate(model_gradient)
        }
        self.gradient_keys = return_dict.keys()
        return_dict["n_samples"] = x.shape[0]
        return return_dict

    @remote_data
    def predict(
        self,
        x: np.ndarray,
        shared_state: Dict[
            str, np.ndarray
        ] = None,  # Set to None per default for clarity reason as the decorator will do it if the arg shared_state is not passed.
    ):
        """Predict method of the fed avg strategy. Executes the following operation :
            * apply user defined (or default) _process method to x
            * if a shared state is given, add it to the model parameters
            * apply the model to the input data (model(x))
            * apply user defined (or default) _postprocess method to the model results

        Args:
            x (np.ndarray): Input data.
            shared_state (Dict[str, np.ndarray], optional): If not None, the shared state will be added to the model parameters' before computing the predictions. Defaults to None.

        Returns:
            Any: Model prediction post precessed by the _postprocess class method.
        """
        # Apply preprocessing methods to adapt the output of the opener.
        x_test = self._safe_preprocess(x=x)

        # Reduce memory consumption as we don't use the model gradients
        with torch.no_grad():
            # If needed, add the shared state to the model parameters
            if shared_state is not None:
                weight_manager.increment_parameters(
                    model=self.model,
                    gradients=shared_state.values(),
                    with_batch_norm_parameters=self.with_batch_norm_parameters,
                )

            self.model.eval()
            y_pred = self.model(x_test)

        # Apply preprocessing methods to adapt the output to the metric.
        return self._postprocess(y_pred=y_pred)

    def load(self, path: Path):
        """Load the stateful arguments of this class, i.e.:
            * self.model
            * self.optimizer
            * self.scheduler (if provided)
            * self.index_generator
            * torch rng state

        Args:
            path (Path): The path where the class has been saved.

        Returns:
            Class: The class with the loaded elements.
        """
        assert (
            path.is_file()
        ), f'Cannot load the model - does not exist {list(path.parent.glob("*"))}'

        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.index_generator = checkpoint["index_generator"]

        torch.set_rng_state(checkpoint["rng_state"])
        return self

    def save(self, path: Path):
        """Saves all the stateful elements of the class to the specified path, i.e.:
            * self.model
            * self.optimizer
            * self.scheduler (if provided)
            * self.index_generator
            * torch rng state

        Args:
            path (Path): A path where to save the class.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict()
                if self.scheduler is not None
                else None,
                "index_generator": self.index_generator,
                "rng_state": torch.get_rng_state(),
            },
            path,
        )
        assert (
            path.is_file()
        ), f'Did not save the model properly {list(path.parent.glob("*"))}'
