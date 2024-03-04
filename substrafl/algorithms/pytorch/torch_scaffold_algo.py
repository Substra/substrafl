import logging
from enum import IntEnum
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional

import torch

from substrafl.algorithms.pytorch import weight_manager
from substrafl.algorithms.pytorch.torch_base_algo import TorchAlgo
from substrafl.exceptions import NumUpdatesValueError
from substrafl.exceptions import ScaffoldLearningRateError
from substrafl.exceptions import TorchScaffoldAlgoParametersUpdateError
from substrafl.index_generator import BaseIndexGenerator
from substrafl.remote import remote_data
from substrafl.strategies.schemas import ScaffoldAveragedStates
from substrafl.strategies.schemas import ScaffoldSharedState
from substrafl.strategies.schemas import StrategyName

logger = logging.getLogger(__name__)


class CUpdateRule(IntEnum):
    """The rule used to update the client control variate

    Values:

        - STABLE (1): The stable rule, I in the Scaffold paper (not implemented)
        - FAST (2): The fast rule, II in the Scaffold paper
    """

    STABLE = 1
    FAST = 2


class TorchScaffoldAlgo(TorchAlgo):
    """To be inherited. Wraps the necessary operation so a torch model can be trained in the Scaffold strategy.

    The ``train`` method:

        - updates the weights of the model with the aggregated weights,
        - initializes or loads the index generator,
        - calls the :py:func:`~substrafl.algorithms.pytorch.torch_scaffold_algo.TorchScaffoldAlgo._local_train` method
          to do the local training
        - then gets the weight updates from the models and sends them to the aggregator.

    The ``predict`` method generates the predictions.

    The child class can override the
    :py:func:`~substrafl.algorithms.pytorch.torch_scaffold_algo.TorchScaffoldAlgo._local_train` and
    :py:func:`~substrafl.algorithms.pytorch.torch_scaffold_algo.TorchScaffoldAlgo._local_predict` methods,
    or other methods if necessary.

    To add a custom parameter to the ``__init__``of the class, also add it to the call to ``super().__init__```
    as shown in the example with ``my_custom_extra_parameter``. Only primitive types (str, int, ...) are supported
    for extra parameters.

    Example:

        .. code-block:: python

            class MyAlgo(TorchScaffoldAlgo):
                def __init__(
                    self,
                    my_custom_extra_parameter,
                ):
                    super().__init__(
                        model=perceptron,
                        criterion=torch.nn.MSELoss(),
                        optimizer=optimizer,
                        num_updates=100,
                        index_generator=NpIndexGenerator(
                            num_updates=10,
                            batch_size=32,
                        ),
                        dataset=MyDataset,
                        my_custom_extra_parameter=my_custom_extra_parameter,
                    )
                def _local_train(
                    self,
                    train_dataset: torch.utils.data.Dataset,
                ):
                    # Create torch dataloader
                    # ``train_dataset = self._dataset(data_from_opener=data_from_opener, is_inference=False)``
                    # is executed prior the execution of this function
                    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=self._index_generator)

                    for x_batch, y_batch in train_data_loader:

                        # Forward pass
                        y_pred = self._model(x_batch)
                        # Compute Loss
                        loss = self._criterion(y_pred, y_batch)
                        self._optimizer.zero_grad()
                        # backward pass: compute the gradients
                        loss.backward()
                        # forward pass: update the weights.
                        self._optimizer.step()

                        # Scaffold specific: to keep between _optimizer.step() and _scheduler.step()
                        # _scheduler and Scaffold strategies are not scientifically validated, it is not
                        # recommended to use one. If one is used, _scheduler.step() must be called after
                        # _scaffold_parameters_update()
                        self._scaffold_parameters_update()
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
        c_update_rule: CUpdateRule = CUpdateRule.FAST,
        seed: Optional[int] = None,
        use_gpu: bool = True,
        *args,
        **kwargs,
    ):
        """The ``__init__`` function is called at each call of the ``train`` or ``predict`` function
        For round>2, some attributes will then be overwritten by their previous states in the `load()` function,
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
            c_update_rule (CUpdateRule): The rule used to update the
                client control variate.
                Defaults to CUpdateRule.FAST.
            seed (typing.Optional[int]): Seed set at the algo initialization on each organization. Defaults to None.
            use_gpu (bool): Whether to use the GPUs if they are available. Defaults to True.
        Raises:
            :ref:`~substrafl.exceptions.NumUpdatesValueError`: If `num_updates` is inferior or equal to zero.
        """
        super().__init__(
            *args,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            index_generator=index_generator,
            dataset=dataset,
            scheduler=scheduler,
            use_gpu=use_gpu,
            seed=seed,
            **kwargs,
        )

        if self._index_generator.num_updates <= 0:
            raise NumUpdatesValueError("Num_updates needs to be superior to 0 for TorchScaffoldAlgo.")

        if not isinstance(self._optimizer, torch.optim.SGD):
            logger.warning("The only optimizer theoretically guaranteed to work with the Scaffold strategy is SGD.")

        self._lr_warnings()

        self._with_batch_norm_parameters = with_batch_norm_parameters
        self._c_update_rule = CUpdateRule(c_update_rule)
        # ci in the paper
        self._client_control_variate: List[torch.Tensor] = None
        # c in the paper
        self._server_control_variate: List[torch.Tensor] = None
        # the lr used by optimizer.step()
        self._current_lr: float = None
        # the delta_variate used in _scaffold_parameters_update()
        self._delta_variate: List[torch.Tensor] = None
        # Private attributes to monitor the use of _scaffold_parameters_update()
        # Must be set to 0 at the beginning of each task execution
        self._scaffold_parameters_update_num_call = 0

    @property
    def strategies(self) -> List[StrategyName]:
        """List of compatible strategies

        Returns:
            typing.List: typing.List[StrategyName]
        """
        return [StrategyName.SCAFFOLD]

    def _lr_warnings(self, learning_rates=None):
        """Scientific warnings about the use of different learning rates during training."""
        # Retrieve all lr defined per layer
        if learning_rates is None:
            learning_rates = set([param_group.get("lr") for param_group in self._optimizer.param_groups])
            learning_rates.discard(0)

        if self._scheduler:
            logger.warning(
                """Scaffold strategies and the use of a scheduler has not been scientifically validated."""
                """\nIf used, self._scheduler.step() must be called after self._scaffold_parameters_update()."""
            )

        if len(learning_rates) == 0:
            # Torch needs a learning rates for each layer to set the optimizer so this case should never happen.
            # Keeping it for consistency
            raise ScaffoldLearningRateError(
                "When using the  Torch Scaffold algo layer, a learning rate must "
                "be passed to for all group of layers in the optimizer and one of them shall be "
                "strictly positive."
            )

        elif len(learning_rates) > 1:
            logger.warning(
                "Different learning rates where found from the optimizer: %s. "
                "The aggregation operation of the Scaffold strategy, requires a unique learning rate. "
                "Hence the smallest one will be used.",
                str(learning_rates),
            )

    def _update_current_lr(self):
        """Update the `self._current_lr` attributes from the optimizer. It will be updated from the scheduler if
        one is used even if this is not recommended as the use of a scheduler with the Scaffold strategy has not
        been scientifically validated.
        Different use cases: https://discuss.pytorch.org/t/get-current-lr-of-optimizer-with-adaptive-lr/24851/6
        """

        if self._scheduler and self._scheduler.get_last_lr():
            learning_rates = set(self._scheduler.get_last_lr())
        else:
            # Retrieve all lr defined per layer
            learning_rates = set([param_group.get("lr") for param_group in self._optimizer.param_groups])

        # If 0 is set as a learning rate for some layers, the weights and biases won't be impacted by the training
        # This behavior is accepted
        learning_rates.discard(0)

        self._lr_warnings(learning_rates=learning_rates)

        self._current_lr = float(min(learning_rates))

    def _scaffold_parameters_update(self):
        """Must be called for each update after the optimizer.step() operation."""
        # Adding control variates on weights times learning rate
        # Scaffold paper's Algo step 10.2 :  yi = last_yi - lr * ( - ci + c) = last_yi + lr * ( ci - c)
        # <=> model = last_model + lr * delta_variate
        self._update_current_lr()
        self._scaffold_parameters_update_num_call += 1
        weight_manager.increment_parameters(
            model=self._model,
            updates=self._delta_variate,
            with_batch_norm_parameters=self._with_batch_norm_parameters,
            updates_multiplier=self._current_lr,
        )

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

            You must use ``next(self._index_generator)`` at each minibatch,
            to ensure that you are using the batches are correct between 2 rounds
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

                    # Scaffold specific function to call between self._optimizer.step() and self._scheduler.step()
                    self._scaffold_parameters_update()

                    if self._scheduler is not None:
                        self._scheduler.step()
        """

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
            # backward pass: compute the gradients
            loss.backward()
            # forward pass: update the weights.
            # if optimizer=SGD, yi = last_yi - lr * grads,
            # with the same lr as _get_current_lr() (used in step 10.2)
            self._optimizer.step()

            # Scaffold specific: to call between self._optimizer.step() and self._scheduler.step()
            self._scaffold_parameters_update()

            if self._scheduler is not None:
                self._scheduler.step()

    @remote_data
    def train(
        self,
        data_from_opener: Any,
        shared_state: Optional[ScaffoldAveragedStates] = None,  # Set to None per default for clarity reason as
        #  the decorator will do it if the arg shared_state is not passed.
    ) -> ScaffoldSharedState:
        """Train method of the Scaffold strategy implemented with torch. This method will execute the following
        operations:

            * instantiates the provided (or default) batch indexer
            * if a shared state is passed, set the parameters of the model to the provided shared state
            * train the model for n_updates
            * compute the weight update and control variate update

        Args:
            data_from_opener (typing.Any): Input data returned by the ``get_data`` method from the opener.
            shared_state (typing.Optional[ScaffoldAveragedStates]): Shared state sent by the aggregate_organization
                (returned by the func strategies.scaffold.avg_shared_states)
                Defaults to None.


        Returns:
            ScaffoldSharedState: the shared states of the Algo
        """

        # Create torch dataset
        train_dataset = self._dataset(data_from_opener, is_inference=False)

        if shared_state is None:  # first round
            # Instantiate the index_generator
            assert self._index_generator.n_samples is None
            self._index_generator.n_samples = len(train_dataset)

            # client_control_variate = zeros matrix with the shape of the model weights
            assert self._client_control_variate is None
            self._client_control_variate = weight_manager.zeros_like_parameters(
                self.model,
                with_batch_norm_parameters=self._with_batch_norm_parameters,
                device=self._device,
            )
            # we initialize the server_control_variate (c in the paper) here so we don't have to do
            # an initialization round
            assert self._server_control_variate is None
            self._server_control_variate = weight_manager.zeros_like_parameters(
                self.model,
                with_batch_norm_parameters=self._with_batch_norm_parameters,
                device=self._device,
            )
        else:  # round>1
            # These should have been loaded by the load() function
            assert self._client_control_variate is not None
            assert self._index_generator.n_samples is not None

            # The shared states is the average of the difference of the parameters_update for all organizations
            # Hence we need to add it to the previous local state parameters
            # Scaffold paper's Algo step 17: model = model + aggregation_lr * parameters_update
            # here shared_state.avg_parameters_update is already aggregation_lr * parameters_update,
            # cf strategies.scaffold.avg_shared_states
            avg_parameters_update = [torch.from_numpy(x).to(self._device) for x in shared_state.avg_parameters_update]
            weight_manager.increment_parameters(
                model=self._model,
                updates=avg_parameters_update,
                with_batch_norm_parameters=self._with_batch_norm_parameters,
            )
            # get the server_control_variate from the aggregator
            self._server_control_variate = [
                torch.from_numpy(t).to(self._device) for t in shared_state.server_control_variate
            ]

        self._index_generator.reset_counter()

        # save original parameters to compute the weight updates and reset the model parameters after
        # the model is trained
        original_parameters = weight_manager.get_parameters(
            model=self._model, with_batch_norm_parameters=self._with_batch_norm_parameters
        )

        # compute delta_variate = ci-c for Scaffold paper's Algo step 10.2 (self._scaffold_parameters_update())
        self._delta_variate = weight_manager.subtract_parameters(
            parameters=self._client_control_variate,
            parameters_to_subtract=self._server_control_variate,
        )

        # Train mode for torch model
        self._model.train()

        # Train the model
        self._local_train(train_dataset)

        self._index_generator.check_num_updates()

        if self._scaffold_parameters_update_num_call != self._index_generator._num_updates:
            raise TorchScaffoldAlgoParametersUpdateError(
                f"`_scaffold_parameters_update` method has been called {self._scaffold_parameters_update_num_call} "
                f"time(s) but num_updates is set to {self._index_generator._num_updates}. Please check within your "
                "`_local_train` function that `_scaffold_parameters_update` is called at each update (each time "
                "self._model(data) is called) after the `self.optimizer.step()` call."
            )

        self._reset_scaffold_parameters_update()

        self._model.eval()

        # Scaffold paper's Algo step 12+13.1: compute parameters_update = (yi-x)
        parameters_update = weight_manager.subtract_parameters(
            parameters=weight_manager.get_parameters(
                model=self._model,
                with_batch_norm_parameters=self._with_batch_norm_parameters,
            ),
            parameters_to_subtract=original_parameters,
        )

        if self._c_update_rule == CUpdateRule.FAST:
            # right_multiplier = -1 / (lr*num_updates)
            right_multiplier = -1.0 / (self._current_lr * self._index_generator.num_updates)
            # Scaffold paper's Algo step 12+13.2: control_variate_update = -c - parameters_update / (lr*num_updates)
            control_variate_update = weight_manager.weighted_sum_parameters(
                parameters_list=[self._server_control_variate, parameters_update],
                coefficient_list=[-1.0, right_multiplier],
            )
        else:
            raise NotImplementedError("rule 1 not implemented")

        # Scaffold paper's Algo step 14: ci = ci + control_variate_update
        self._client_control_variate = weight_manager.add_parameters(
            parameters=self._client_control_variate,
            parameters_to_add=control_variate_update,
        )

        # Re set to the previous state
        weight_manager.set_parameters(
            model=self._model,
            parameters=original_parameters,
            with_batch_norm_parameters=self._with_batch_norm_parameters,
        )

        # Scaffold paper's Algo step 13: return model_weight_update & control_variate_update
        return_dict = ScaffoldSharedState(
            parameters_update=[w.cpu().detach().numpy() for w in parameters_update],
            control_variate_update=[c.cpu().detach().numpy() for c in control_variate_update],
            server_control_variate=[s.cpu().detach().numpy() for s in self._server_control_variate],
            n_samples=len(train_dataset),
        )
        return return_dict

    def _reset_scaffold_parameters_update(self):
        self._scaffold_parameters_update_num_call = 0

    def _get_state_to_save(self) -> dict:
        """Get the local state to save, the only strategy-specific variable
        to save is the ``client_control_variate``.

        Returns:
            dict: checkpoint
        """
        local_state = super()._get_state_to_save()
        local_state.update(
            {
                "client_control_variate": self._client_control_variate,
            }
        )
        return local_state

    def _update_from_checkpoint(self, path: Path) -> dict:
        """Load the local state from the checkpoint.

        Args:
            path (pathlib.Path): path where the checkpoint is saved

        Returns:
            dict: checkpoint
        """
        checkpoint = super()._update_from_checkpoint(path=path)
        self._client_control_variate = checkpoint.pop("client_control_variate")
        return checkpoint

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
