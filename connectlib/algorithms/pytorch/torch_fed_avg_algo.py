import logging
from typing import Any
from typing import List
from typing import Optional

import torch

from connectlib.algorithms.pytorch import weight_manager
from connectlib.algorithms.pytorch.torch_base_algo import TorchAlgo
from connectlib.index_generator import BaseIndexGenerator
from connectlib.remote import remote_data
from connectlib.schemas import FedAvgAveragedState
from connectlib.schemas import FedAvgSharedState
from connectlib.schemas import StrategyName

logger = logging.getLogger(__name__)


class TorchFedAvgAlgo(TorchAlgo):
    """To be inherited. Wraps the necessary operation so a torch model can be trained in the Federated Averaging strategy.

    The ``train`` method:

        - updates the weights of the model with the aggregated weights,
        - initializes or loads the index generator,
        - calls the :py:func:`~connectlib.algorithms.pytorch.TorchFedAvgAlgo._local_train` method to do the local
          training
        - then gets the weight updates from the models and sends them to the aggregator.

    The ``predict`` method calls the :py:func:`~connectlib.algorithms.pytorch.TorchFedAvgAlgo._local_predict` method to
    generate the predictions.

    The child class must implement the :py:func:`~connectlib.algorithms.pytorch.TorchFedAvgAlgo._local_train` and
    :py:func:`~connectlib.algorithms.pytorch.TorchFedAvgAlgo._local_predict` methods, and can override
    other methods if necessary.

    To add a custom parameter to the ``__init__`` of the class, also add it to the call to ``super().__init__``
    as shown in the example with ``my_custom_extra_parameter``. Only primitive types (str, int, ...) are supported
    for extra parameters.

    Example:

        .. code-block:: python

            class MyAlgo(TorchFedAvgAlgo):
                def __init__(
                    self,
                    my_custom_extra_parameter,
                ):
                    super().__init__(
                        model=perceptron,
                        criterion=torch.nn.MSELoss(),
                        optimizer=optimizer,
                        index_generator=NpIndexGenerator(
                            num_updates=100,
                            batch_size=32,
                        ),
                        my_custom_extra_parameter=my_custom_extra_parameter,
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

    The algo needs to get the number of samples in the dataset from the x sent by the opener.
    By default, it uses len(x). If that is not the proper way of getting the number of samples,
    override the :py:func:`~connectlib.algorithms.pytorch.TorchFedAvgAlgo._get_len_from_x` function

    Example:

        .. code-block:: python

            def _get_len_from_x(self, x):
                return len(x)

    As development tools, the ``train`` and ``predict`` method comes with a default argument : ``_skip``.
    If ``_skip`` is set to ``True``, only the function will be executed and not all the code related to Connect.
    This allows to quickly debug code and use the defined algorithm as is.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        index_generator: BaseIndexGenerator,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        with_batch_norm_parameters: bool = False,
        use_gpu: bool = True,
        *args,
        **kwargs,
    ):
        """
        The ``__init__`` functions is called at each call of the `train()` or `predict()` function
        For round>=2, some attributes will then be overwritten by their previous states in the `load()` function,
        before the `train()` or `predict()` function is ran.

        Args:
            model (torch.nn.modules.module.Module): A torch model.
            criterion (torch.nn.modules.loss._Loss): A torch criterion (loss).
            optimizer (torch.optim.Optimizer): A torch optimizer linked to the model.
            index_generator (BaseIndexGenerator): a stateful index generator.
                Must inherit from BaseIndexGenerator. The __next__ method shall return a python object (batch_index)
                which is used for selecting each batch from the output of the get_X and get_y methods of the opener
                during training in this way: ``x[batch_index], y[batch_index]``.
                If overridden, the generator class must be defined either as part of a package or in a different file
                than the one from which the ``execute_experiment`` function is called.
            scheduler (torch.optim.lr_scheduler._LRScheduler, Optional): A torch scheduler that will be called at every
                batch. If None, no scheduler will be used. Defaults to None.
            with_batch_norm_parameters (bool): Whether to include the batch norm layer parameters in the fed avg
                strategy. Defaults to False.
            use_gpu (bool): Whether to use the GPUs if they are available. Defaults to True.
        """
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            index_generator=index_generator,
            scheduler=scheduler,
            use_gpu=use_gpu,
            *args,
            **kwargs,
        )
        self._with_batch_norm_parameters = with_batch_norm_parameters

    @property
    def strategies(self) -> List[StrategyName]:
        """List of compatible strategies

        Returns:
            typing.List: typing.List[StrategyName]
        """
        return [StrategyName.FEDERATED_AVERAGING]

    @remote_data
    def train(
        self,
        x: Any,
        y: Any,
        shared_state: Optional[FedAvgAveragedState] = None,  # Set to None per default for clarity reason as
        # the decorator will do it if the arg shared_state is not passed.
    ) -> FedAvgSharedState:
        """Train method of the fed avg strategy implemented with torch. This method will execute the following
        operations:

            * instantiates the provided (or default) batch indexer
            * if a shared state is passed, set the parameters of the model to the provided shared state
            * train the model for n_updates
            * compute the weight update

        Args:
            x (typing.Any): Input data.
            y (typing.Any): Input target.
            shared_state (FedAvgAveragedState, Optional): Dict containing torch parameters that
                will be set to the model. Defaults to None.

        Returns:
            FedAvgSharedState: weight update (delta between fine-tuned
            weights and previous weights)
        """
        if shared_state is None:
            # Instantiate the index_generator
            assert self._index_generator.n_samples is None
            self._index_generator.n_samples = self._get_len_from_x(x)
        else:
            assert self._index_generator.n_samples is not None
            # The shared states is the average of the model parameter updates for all organizations
            # Hence we need to add it to the previous local state parameters
            parameter_updates = [torch.from_numpy(x).to(self._device) for x in shared_state.avg_parameters_update]
            weight_manager.increment_parameters(
                model=self._model,
                updates=parameter_updates,
                with_batch_norm_parameters=self._with_batch_norm_parameters,
            )

        self._index_generator.reset_counter()

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

        self._index_generator.check_num_updates()

        self._model.eval()

        parameters_update = weight_manager.subtract_parameters(
            parameters=weight_manager.get_parameters(
                model=self._model,
                with_batch_norm_parameters=self._with_batch_norm_parameters,
            ),
            parameters_to_subtract=old_parameters,
        )

        # Re set to the previous state
        weight_manager.set_parameters(
            model=self._model,
            parameters=old_parameters,
            with_batch_norm_parameters=self._with_batch_norm_parameters,
        )

        return FedAvgSharedState(
            n_samples=self._get_len_from_x(x),
            parameters_update=[p.cpu().detach().numpy() for p in parameters_update],
        )

    def summary(self):
        """Summary of the class to be exposed in the experiment summary file

        Returns:
            dict: a json-serializable dict with the attributes the user wants to store
        """
        summary = super().summary()
        summary.update(
            {
                "with_batch_norm_parameters": str(self._with_batch_norm_parameters),
            }
        )
        return summary
