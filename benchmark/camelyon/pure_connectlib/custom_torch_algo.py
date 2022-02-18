import logging
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
import torch

from connectlib import NpIndexGenerator
from connectlib.algorithms import Algo
from connectlib.algorithms.pytorch import weight_manager
from connectlib.remote import remote_data

logger = logging.getLogger(__name__)

# TODO/INFO : for the next strategy, all methods and args of this class could be wrapped into a generic
# TorchAlgo class. Every strategy class could inherit from it.


class TorchFedAvgAlgo(Algo):
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

    def local_training(self, x, y):
        raise NotImplementedError

    def local_prediction(self, x):
        raise NotImplementedError

    @remote_data
    def train(
        self,
        x: Any,
        y: Any,
        shared_state=None,  # Set to None per default for clarity reason as the decorator will
        # do it if the arg shared_state is not passed.
    ) -> Dict[str, np.ndarray]:
        """Train method of the fed avg strategy implemented with torch. This method will execute
        the following operations:
            * instantiates the provided (or default) batch indexer
            * apply the provided (or default) _processing method to x and y
            * if a shared state is passed, set the parameters of the model to the provided shared state
            * train the model for n_updates
            * compute the weight update

        Args:
            x (Any): Input data.
            y (Any): Input target.
            shared_state (Dict[str, np.ndarray], optional): Dict containing torch parameters that will
                be set to the model. Defaults to None.

        Returns:
            Dict[str, np.ndarray]: weight update (delta between fine-tuned weights and previous weights)
        """

        # The shared states is the average of the difference of the gradient for all nodes
        # Hence we need to add it to the previous local state parameters
        if shared_state is not None:
            weight_manager.increment_parameters(
                model=self.model,
                updates=shared_state.values(),
                with_batch_norm_parameters=self.with_batch_norm_parameters,
            )

        old_parameters = weight_manager.get_parameters(
            model=self.model, with_batch_norm_parameters=self.with_batch_norm_parameters
        )

        self.local_training(x, y)

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

        return_dict = {f"grad_{i}": g.cpu().detach().numpy() for i, g in enumerate(model_gradient)}
        return_dict["n_samples"] = 1
        return return_dict

    @remote_data
    def predict(
        self,
        x: np.ndarray,
        shared_state: Dict[
            str, np.ndarray
        ] = None,  # Set to None per default for clarity reason as the decorator will do it if the
        # arg shared_state is not passed.
    ):
        """Predict method of the fed avg strategy. Executes the following operation :
            * apply user defined (or default) _process method to x
            * if a shared state is given, add it to the model parameters
            * apply the model to the input data (model(x))
            * apply user defined (or default) _postprocess method to the model results

        Args:
            x (np.ndarray): Input data.
            shared_state (Dict[str, np.ndarray], optional): If not None, the shared state will be added
                to the model parameters' before computing the predictions. Defaults to None.

        Returns:
            Any: Model prediction post precessed by the _postprocess class method.
        """

        # Reduce memory consumption as we don't use the model gradients
        with torch.no_grad():
            # If needed, add the shared state to the model parameters
            if shared_state is not None:
                weight_manager.increment_parameters(
                    model=self.model,
                    updates=shared_state.values(),
                    with_batch_norm_parameters=self.with_batch_norm_parameters,
                )

        return self.local_prediction(x)

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
        assert path.is_file(), f'Cannot load the model - does not exist {list(path.parent.glob("*"))}'

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
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                "index_generator": self.index_generator,
                "rng_state": torch.get_rng_state(),
            },
            path,
        )
        assert path.is_file(), f'Did not save the model properly {list(path.parent.glob("*"))}'
