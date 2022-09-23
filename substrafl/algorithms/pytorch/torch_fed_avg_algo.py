import logging
from typing import Any
from typing import List
from typing import Optional

import torch

from substrafl.algorithms.pytorch import weight_manager
from substrafl.algorithms.pytorch.torch_base_algo import TorchAlgo
from substrafl.index_generator import BaseIndexGenerator
from substrafl.remote import remote_data
from substrafl.schemas import FedAvgAveragedState
from substrafl.schemas import FedAvgSharedState
from substrafl.schemas import StrategyName

logger = logging.getLogger(__name__)


class TorchFedAvgAlgo(TorchAlgo):
    """To be inherited. Wraps the necessary operation so a torch model can be trained in the Federated Averaging
    strategy.

    The ``train`` method:

        - updates the weights of the model with the aggregated weights,
        - initializes or loads the index generator,
        - calls the :py:func:`~substrafl.algorithms.pytorch.TorchFedAvgAlgo._local_train` method to do the local
          training
        - then gets the weight updates from the models and sends them to the aggregator.

    The ``predict`` method generates the predictions.

    The child class can override the :py:func:`~substrafl.algorithms.pytorch.TorchFedAvgAlgo._local_train` and
    :py:func:`~substrafl.algorithms.pytorch.TorchFedAvgAlgo._local_predict` methods, or other methods if necessary.

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
                        dataset=MyDataset,
                        my_custom_extra_parameter=my_custom_extra_parameter,
                    )
                def _local_train(
                    self,
                    train_dataset: torch.utils.data.Dataset,
                ):

                    # Create torch dataloader from the automatically instantiated dataset
                    # ``train_dataset = self._dataset(datasamples=datasamples, is_inference=False)`` is executed prior
                    #  the execution of this function
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

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        index_generator: BaseIndexGenerator,
        dataset: torch.utils.data.Dataset,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        with_batch_norm_parameters: bool = False,
        seed: Optional[int] = None,
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
                which is used for selecting each batch from the output of the get_data method of the opener
                during training in this way: ``x[batch_index], y[batch_index]``.
                If overridden, the generator class must be defined either as part of a package or in a different file
                than the one from which the ``execute_experiment`` function is called.
                This generator is used as stateful ``batch_sampler`` of the data loader created from the given
                ``dataset``
            dataset (torch.utils.data.Dataset): an instantiable dataset class whose ``__init__`` arguments are
                ``x``, ``y`` and ``is_inference``. The torch datasets used for both training and inference will be
                instantiate from it prior to the ``_local_train`` execution and within the ``predict`` method.
                The ``__getitem__`` methods of those generated datasets must return both ``x`` (training data) and y
                (target values) when ``is_inference`` is set to ``False`` and only ``x`` (testing data) when
                ``is_inference`` is set to True.
                This behavior can be changed by re-writing the `_local_train` or `predict` methods.
            scheduler (torch.optim.lr_scheduler._LRScheduler, Optional): A torch scheduler that will be called at every
                batch. If None, no scheduler will be used. Defaults to None.
            with_batch_norm_parameters (bool): Whether to include the batch norm layer parameters in the fed avg
                strategy. Defaults to False.
            seed (typing.Optional[int]): Seed set at the algo initialization on each organization. Defaults to None.
            use_gpu (bool): Whether to use the GPUs if they are available. Defaults to True.
        """
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            index_generator=index_generator,
            dataset=dataset,
            scheduler=scheduler,
            seed=seed,
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
        datasamples: Any,
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
            datasamples (typing.Any): Input data returned by the ``get_data`` method from the opener.
            shared_state (FedAvgAveragedState, Optional): Dict containing torch parameters that
                will be set to the model. Defaults to None.

        Returns:
            FedAvgSharedState: weight update (delta between fine-tuned
            weights and previous weights)
        """

        # Create torch dataset
        train_dataset = self._dataset(datasamples, is_inference=False)

        if shared_state is None:
            # Instantiate the index_generator
            assert self._index_generator.n_samples is None
            self._index_generator.n_samples = len(train_dataset)
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
        self._local_train(train_dataset)

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
            n_samples=len(train_dataset),
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
