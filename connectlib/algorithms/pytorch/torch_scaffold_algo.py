import abc
import logging
from enum import IntEnum
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import Type

import torch

from connectlib.algorithms.pytorch import weight_manager
from connectlib.algorithms.pytorch.torch_base_algo import TorchAlgo
from connectlib.exceptions import IndexGeneratorUpdateError
from connectlib.exceptions import NumUpdatesValueError
from connectlib.index_generator import BaseIndexGenerator
from connectlib.index_generator import NpIndexGenerator
from connectlib.remote import remote_data
from connectlib.schemas import ScaffoldAveragedStates
from connectlib.schemas import ScaffoldSharedState

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
        - initialises or loads the index generator,
        - calls the :py:func:`~connectlib.algorithms.pytorch.torch_scaffold_algo.TorchScaffoldAlgo._local_train` method
          to do the local training
        - then gets the weight updates from the models and sends them to the aggregator.

    The ``predict`` method calls the
    :py:func:`~connectlib.algorithms.pytorch.torch_scaffold_algo.TorchScaffoldAlgo._local_predict` method to generate
    the predictions.

    The child class must implement the
    :py:func:`~connectlib.algorithms.pytorch.torch_scaffold_algo.TorchScaffoldAlgo._local_train` and
    :py:func:`~connectlib.algorithms.pytorch.torch_scaffold_algo.TorchScaffoldAlgo._local_predict` methods,
    and can override other methods if necessary.

    Example:

        .. code-block:: python

            class MyAlgo(TorchScaffoldAlgo):
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
                    # for each update
                    for batch_index in self._index_generator:
                        # get minibatch
                        x_batch, y_batch = x[batch_index], y[batch_index]
                        # Forward pass
                        y_pred = self._model(x_batch)
                        # Compute Loss
                        loss = self._criterion(y_pred, y_batch)
                        self._optimizer.zero_grad()
                        # backward pass: compute the gradients
                        loss.backward()
                        # forward pass: update the weights.
                        self._optimizer.step()

                        # scaffold specific: to keep between _optimizer.step() and _scheduler.step()
                        self._scaffold_parameters_update()

                        if self._scheduler is not None:
                            self._scheduler.step()

                def _local_predict(self, x: Any) -> Any:
                    with torch.inference_mode():
                        y = self._model(x)
                    return y

    The algo needs to get the number of samples in the dataset from the x sent by the opener.
    By default, it uses len(x). If that is not the proper way of getting the number of samples,
    override the ``_get_len_from_x`` function

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
        num_updates: int,
        batch_size: Optional[int],
        get_index_generator: Type[BaseIndexGenerator] = NpIndexGenerator,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        with_batch_norm_parameters: bool = False,
        c_update_rule: CUpdateRule = CUpdateRule.FAST,
    ):
        """The ``__init__`` function is called at each call of the ``train`` or ``predict`` function
        For round>2, some attributes will then be overwritten by their previous states in the `load()` function,
        before the `train()` or `predict()` function is ran.

        Args:
            model (torch.nn.modules.module.Module): A torch model.
            criterion (torch.nn.modules.loss._Loss): A torch criterion (loss).
            optimizer (torch.optim.Optimizer): A torch optimizer linked to the model.
            scheduler (torch.optim.lr_scheduler._LRScheduler, Optional): A torch scheduler that will be called at every
                batch. If None, no scheduler will be used. Defaults to None.
            num_updates (int): The number of times the model will be trained at each step of the strategy (i.e. of the
                train function).
            batch_size (int, Optional): The number of samples used for each updates. If None, the whole input data will
                be used.
            get_index_generator (Type[BaseIndexGenerator], Optional): A class returning a stateful index generator. Must
                inherit from BaseIndexGenerator. The __next__ method shall return a python object (batch_index) which
                is used for selecting each batch from the output of the _preprocess method during training in this way :
                ``x[batch_index], y[batch_index]``. Defaults to NpIndexGenerator.
                If overridden, the generator class must be defined either as part of a package or in a different file
                than the one from which the ``execute_experiment`` function is called.
            with_batch_norm_parameters (bool): Whether to include the batch norm layer parameters in the fed avg
                strategy. Defaults to False.
            c_update_rule (CUpdateRule): The rule used to update the
                client control variate.
                Defaults to CUpdateRule.FAST.
        Raises:
            :ref:`~connectlib.exceptions.NumUpdatesValueError`: If `num_updates` is inferior or equal to zero.
        """
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            num_updates=num_updates,
            batch_size=batch_size,
            get_index_generator=get_index_generator,
            scheduler=scheduler,
        )
        if self._num_updates <= 0:
            raise NumUpdatesValueError("Num_updates needs to be superior to 0 for TorchScaffoldAlgo.")
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

    def _update_current_lr(self):
        """method to get the current learning rate from the scheduler, or the optimizer if no scheduler.
        If the optimizer has no learning rate, default value is 1.0 (the lr has no effect)
        Different use cases: https://discuss.pytorch.org/t/get-current-lr-of-optimizer-with-adaptive-lr/24851/6

        Returns:
            float: current learning rate
        """
        if self._scheduler and self._scheduler.get_last_lr():
            self._current_lr = self._scheduler.get_last_lr()[0]
        elif self._optimizer.param_groups[0]["lr"]:
            # TODO(sci-review): check if we need to implement the use case where there is different lr per param_group
            self._current_lr = float(self._optimizer.param_groups[0]["lr"])
        elif self._optimizer.defaults["lr"]:
            logger.warning("Could not get the current optimizer learning rate. `optimizer.defaults['lr']` will be used")
            self._current_lr = float(self._optimizer.defaults["lr"])
        else:
            logger.warning(
                "Could not get the optimizer learning rate. The default value of 1.0 will be used when"
                "computing the Scaffold parameters_update"
            )
            self._current_lr = 1.0

    def _scaffold_parameters_update(self):
        # Adding control variates on weights times learning rate
        # Scaffold paper's Algo step 10.2 :  yi = last_yi - lr * ( - ci + c) = last_yi + lr * ( ci - c)
        # <=> model = last_model + lr * delta_variate
        self._update_current_lr()
        weight_manager.increment_parameters(
            model=self._model,
            updates=self._delta_variate,
            with_batch_norm_parameters=self._with_batch_norm_parameters,
            updates_multiplier=self._current_lr,
        )

    @abc.abstractmethod
    def _local_train(
        self,
        x: Any,
        y: Any,
    ):
        """Local train method, the user must override it, this function
        contains the local training loop with the data pre-processing.

        Train the model on ``num_updates`` minibatches, using
        ``self._index_generator`` to generate the batches.

        Args:
            x (typing.Any): x as returned by the opener
            y (typing.Any): y as returned by the opener

        Important:

            You must use ``next(self._index_generator)`` at each minibatch,
            to ensure that you are using the batches are correct between 2 rounds
            of the federated learning strategy.

        Important:

            Call the function ``self._scaffold_parameters_update()`` between the
            optimizer and scheduler update, see the example.

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

                    # SCAFFOLD specific function, to call between the
                    # optimizer and scheduler update
                    self._scaffold_parameters_update()

                    if self._scheduler is not None:
                        self._scheduler.step()
        """
        # for each update
        for batch_index in self._index_generator:
            # get minibatch
            x_batch, y_batch = x[batch_index], y[batch_index]
            # Forward pass
            y_pred = self._model(x_batch)
            # Compute Loss
            loss = self._criterion(y_pred, y_batch)
            self._optimizer.zero_grad()
            # backward pass: compute the gradients
            loss.backward()
            # forward pass: update the weights.
            # TODO(sci-review): check that we execute Scaffold paper's Algo step 10.1:
            # if optimizer=SGD, yi = last_yi - lr * grads,
            # with the same lr as _get_current_lr() (used in step 10.2)
            self._optimizer.step()

            # scaffold specific: to keep between _optimizer.step() and _scheduler.step()
            self._scaffold_parameters_update()

            if self._scheduler is not None:
                self._scheduler.step()

    @remote_data
    def train(
        self,
        x: Any,
        y: Any,
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
            x (typing.Any): Input data.
            y (typing.Any): Input target.
            shared_state (typing.Optional[ScaffoldAveragedStates]): Shared state sent by the aggregate_node
                (returned by the func strategies.scaffold.avg_shared_states)
                Defaults to None.


        Returns:
            ScaffoldSharedState: the shared states of the Algo
        """

        if shared_state is None:  # first round
            # Instantiate the index_generator
            assert self._index_generator is None
            self._index_generator = self._get_index_generator(
                n_samples=self._get_len_from_x(x),
                batch_size=self._batch_size,
                num_updates=self._num_updates,
                shuffle=True,
                drop_last=False,
            )

            # client_control_variate = zeros matrix with the shape of the model weights
            assert self._client_control_variate is None
            self._client_control_variate = weight_manager.zeros_like_parameters(
                self.model, with_batch_norm_parameters=self._with_batch_norm_parameters
            )
            # we initialize the server_control_variate (c in the paper) here so we don't have to do
            # an initialization round
            assert self._server_control_variate is None
            self._server_control_variate = weight_manager.zeros_like_parameters(
                self.model, with_batch_norm_parameters=self._with_batch_norm_parameters
            )
        else:  # round>1
            # The shared states is the average of the difference of the parameters_update for all nodes
            # Hence we need to add it to the previous local state parameters
            # Scaffold paper's Algo step 17: model = model + aggregation_lr * parameters_update
            # here shared_state.avg_parameters_update is already aggregation_lr * parameters_update,
            # cf strategies.scaffold.avg_shared_states
            weight_manager.increment_parameters(
                model=self._model,
                updates=shared_state.avg_parameters_update,
                with_batch_norm_parameters=self._with_batch_norm_parameters,
            )
            # get the server_control_variate from the aggregator
            self._server_control_variate = [torch.from_numpy(t) for t in shared_state.server_control_variate]

            # These should have been loaded by the load() function
            assert self._client_control_variate is not None
            assert self._index_generator is not None

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
            # TODO(sci-review): for now we take the lr from the latest optimizer.step(), be sure this is the right one
            right_multiplier = -1.0 / (self._current_lr * self._num_updates)
            # Scaffold paper's Algo step 12+13.2: control_variate_update = -c - parameters_update / (lr*num_updates)
            control_variate_update = weight_manager.weighted_sum_parameters(
                parameters_list=[self._server_control_variate, parameters_update],
                coefficient_list=[-1.0, right_multiplier],
            )
        else:
            # TODO(sci-review): implement rule 1 ? and add tests
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
            n_samples=self._get_len_from_x(x),
        )
        return return_dict

    @remote_data
    def predict(
        self,
        x: Any,
        shared_state: ScaffoldAveragedStates,
    ):
        """Predict method of the scaffold strategy. Executes the following operation:

            * If a shared state is given, add it to the model parameters
            * Apply the :py:func:`~connectlib.algorithms.pytorch.torch_scaffold_algo.TorchScaffoldAlgo._local_predict`
            * Return the predictions

        Args:
            x (typing.Any): Input data.
            shared_state (ScaffoldAveragedStates): The shared state is added
                to the model parameters before computing the predictions.

        Returns:
            typing.Any: Model prediction post precessed by the _postprocess class method.
        """
        # Reduce memory consumption as we don't use the model parameters_update
        with torch.inference_mode():
            # Add the shared state to the model parameters
            weight_manager.increment_parameters(
                model=self._model,
                updates=shared_state.avg_parameters_update,
                with_batch_norm_parameters=self._with_batch_norm_parameters,
            )

            self._model.eval()

        predictions = self._local_predict(x)
        return predictions

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
