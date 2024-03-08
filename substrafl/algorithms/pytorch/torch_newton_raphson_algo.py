import math
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch

from substrafl.algorithms.pytorch import weight_manager
from substrafl.algorithms.pytorch.torch_base_algo import TorchAlgo
from substrafl.exceptions import CriterionReductionError
from substrafl.exceptions import NegativeHessianMatrixError
from substrafl.index_generator import NpIndexGenerator
from substrafl.remote import remote_data
from substrafl.strategies.schemas import NewtonRaphsonAveragedStates
from substrafl.strategies.schemas import NewtonRaphsonSharedState
from substrafl.strategies.schemas import StrategyName


class TorchNewtonRaphsonAlgo(TorchAlgo):
    """To be inherited. Wraps the necessary operation so a torch model can be trained in the Newton-Raphson strategy.

    The ``train`` method:

        - updates the weights of the model with the calculated weight updates
        - creates and initializes the index generator with the given batch size
        - calls the
          :py:func:`~substrafl.algorithms.pytorch.torch_newton_raphson_algo.TorchNewtonRaphsonAlgo._local_train`
          method to compute the local gradients and Hessian and sends them to the aggregator.
        - a L2 regularization can be applied to the loss by settings ``l2_coeff`` different to zero (default value). L2
          regularization adds numerical stability when inverting the hessian.

    The child class can overwrite
    :py:func:`~substrafl.algorithms.pytorch.torch_newton_raphson_algo.TorchNewtonRaphsonAlgo._local_train`
    and :py:func:`~substrafl.algorithms.pytorch.torch_newton_raphson_algo.TorchNewtonRaphsonAlgo._local_predict`,
    or other methods if necessary.

    To add a custom parameter to the ``__init__`` of the class, also add it to the call to ``super().__init__``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.modules.loss._Loss,
        batch_size: Optional[int],
        dataset: torch.utils.data.Dataset,
        l2_coeff: float = 0,
        with_batch_norm_parameters: bool = False,
        seed: Optional[int] = None,
        use_gpu: bool = True,
        *args,
        **kwargs,
    ):
        """The ``__init__`` function is called at each call of the ``train`` or ``predict`` function.

        For ``round>=2``, some attributes will then be overwritten by their previous states in the
        :py:func:`~substrafl.algorithms.pytorch.torch_base_algo.TorchAlgo.load_local_state` function, before the
        ``train`` or ``predict`` function is ran.

        ``TorchNewtonRaphsonAlgo`` computes its :py:class:`~substrafl.strategies.schemas.NewtonRaphsonSharedState`
        (gradients and Hessian matrix) on all the samples of the dataset. Data might be split into mini-batches to
        prevent loading too much data at once.

        Args:
            model (torch.nn.modules.module.Module): A torch model.
            criterion (torch.nn.modules.loss._Loss): A torch criterion (loss).
            batch_size (int): The size of the batch. If set to None it will be set to the number of samples in the
                dataset. Note that dividing the data to batches is done only to avoid the memory issues. The weights
                are updated only at the end of the epoch.
            dataset (torch.utils.data.Dataset): an instantiable dataset class whose ``__init__`` arguments are
                ``x``, ``y`` and ``is_inference``. The torch datasets used for both training and inference will be
                instantiate from it prior to the ``_local_train`` execution and within the ``predict`` method.
                The ``__getitem__`` methods of those generated datasets must return both ``x`` (training data) and y
                (target values) when ``is_inference`` is set to ``False`` and only ``x`` (testing data) when
                ``is_inference`` is set to True.
                This behavior can be changed by re-writing the `_local_train` or `predict` methods.
            l2_coeff (float): L2 regularization coefficient. The larger l2_coeff is, the better the stability of the
                hessian matrix will be, however the convergence might be slower. Defaults to 0.
            with_batch_norm_parameters (bool): Whether to include the batch norm layer parameters in the Newton-Raphson
                strategy. Defaults to False.
            seed (typing.Optional[int]): Seed set at the algo initialization on each organization. Defaults to None.
            use_gpu (bool): Whether to use the GPUs if they are available. Defaults to True.
        """
        assert "optimizer" not in kwargs, "Newton Raphson strategy does not uses optimizers"

        super().__init__(
            *args,
            model=model,
            criterion=criterion,
            optimizer=None,
            index_generator=None,
            dataset=dataset,
            use_gpu=use_gpu,
            seed=seed,
            **kwargs,
        )
        self._with_batch_norm_parameters = with_batch_norm_parameters
        self._l2_coeff = l2_coeff
        self._batch_size = batch_size

        if self._criterion.reduction != "mean":
            raise CriterionReductionError(
                "The criterion reduction must be set to 'mean' to use the Newton-Raphson strategy"
            )

        # initialized and used only in the train method
        self._final_gradients = None
        self._final_hessian = None
        self._n_samples_done = None

    @property
    def strategies(self) -> List[StrategyName]:
        """List of compatible strategies

        Returns:
            typing.List: typing.List[StrategyName]
        """
        return [StrategyName.NEWTON_RAPHSON]

    def _l2_reg(self) -> torch.Tensor:
        """Compute the l2 regularization regarding the model parameters.

        Returns:
            torch.Tensor: the updated loss with the l2 regularization included.
        """
        # L2 regularization
        l2_reg = 0
        for param in self.model.parameters():
            l2_reg += self._l2_coeff * torch.sum(param**2) / 2
        return l2_reg

    def _initialize_gradients_and_hessian(self):
        """Initializes the gradients and hessian matrices"""
        number_of_trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)

        n_samples_done = 0

        final_gradients = [torch.zeros_like(p).numpy() for p in self._model.parameters()]
        final_hessian = np.zeros([number_of_trainable_params, number_of_trainable_params])

        return final_gradients, final_hessian, n_samples_done

    def _update_gradients_and_hessian(self, loss: torch.Tensor, current_batch_size: int):
        """Updates the gradients and hessian matrices.

        Args:
            loss (torch.Tensor): the loss to compute the gradients and hessian from.
            current_batch_size (int): The length of the batch used to compute the given loss.
        """

        gradients, hessian = self._compute_gradients_and_hessian(loss)

        self._n_samples_done += current_batch_size

        batch_coefficient = current_batch_size / self._index_generator.n_samples

        self._final_hessian += hessian.cpu().detach().numpy() * batch_coefficient
        self._final_gradients = [
            sum(final_grad, grad.cpu().detach().numpy() * batch_coefficient)
            for final_grad, grad in zip(self._final_gradients, gradients)
        ]

    def _instantiate_index_generator(self, n_samples):
        if self._batch_size is None:
            # If batch_size is None, it is set to the number of samples in the dataset by the index generator
            num_updates = 1
        else:
            num_updates = math.ceil(float(n_samples) / self._batch_size)

        index_generator = NpIndexGenerator(batch_size=self._batch_size, num_updates=num_updates, drop_last=False)
        index_generator.n_samples = n_samples
        return index_generator

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
            of the Newton Raphson strategy.

        Important:

            Call the function ``self._update_gradients_and_hessian(loss, current_batch_size)`` after computing the loss
            and the current_batch_size.

        Example:

            .. code-block:: python

                # As the parameters of the model don't change during the loop, the l2 regularization is constant and
                # can be calculated only once for all the batches.
                l2_reg = self._l2_reg()

                # Create torch dataloader
                train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=self._index_generator)

                for x_batch, y_batch in train_data_loader:

                    # Forward pass
                    y_pred = self._model(x_batch)

                    # Compute Loss
                    loss = self._criterion(y_pred, y_batch)

                    # L2 regularization
                    loss += l2_reg

                    current_batch_size = len(x_batch)

                    # NEWTON RAPHSON specific function, to call after computing the loss and the current_batch_size.

                    self._update_gradients_and_hessian(loss, current_batch_size)
        """

        # As the parameters of the model don't change during the loop, the l2 regularization is constant and can be
        # calculated only once for all the batches.
        l2_reg = self._l2_reg()

        # Create torch dataloader
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=self._index_generator)

        for x_batch, y_batch in train_data_loader:
            x_batch = x_batch.to(self._device)
            y_batch = y_batch.to(self._device)

            # Forward pass
            y_pred = self._model(x_batch)

            # Compute Loss
            loss = self._criterion(y_pred, y_batch)

            # L2 regularization
            loss += l2_reg

            current_batch_size = len(x_batch)

            self._update_gradients_and_hessian(loss, current_batch_size)

    def _local_predict(self, predict_dataset: torch.utils.data.Dataset) -> torch.Tensor:
        """Execute the following operations:

            * Create the torch dataloader using the batch size given at the ``__init__`` of the class
            * Set the model to `eval` mode
            * Return the predictions

        Args:
            predict_dataset (torch.utils.data.Dataset): predict_dataset build from the x returned by the opener.

        Returns:
            torch.Tensor: The computed predictions.
        """
        dataloader_batchsize = min(self._batch_size, len(predict_dataset)) if self._batch_size else len(predict_dataset)
        predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=dataloader_batchsize)

        self._model.eval()

        predictions = torch.Tensor([]).to(self._device)
        with torch.no_grad():
            for x in predict_loader:
                x = x.to(self._device)
                predictions = torch.cat((predictions, self._model(x)), 0)

        predictions = predictions.cpu().detach()

        return predictions

    @remote_data
    def train(
        self,
        data_from_opener: Any,
        # Set shared_state to None per default for clarity reason as
        # the decorator will do it if the arg shared_state is not passed.
        shared_state: Optional[NewtonRaphsonAveragedStates] = None,
    ) -> NewtonRaphsonSharedState:
        """Train method of the Newton Raphson strategy implemented with torch. This method will execute the following
        operations:

            * creates and initializes the index generator
            * if a shared state is passed, set the weights of the model to the provided shared state weights
            * initializes hessians and gradient
            * calls the
                :py:func:`~substrafl.algorithms.pytorch.torch_newton_raphson_algo.TorchNewtonRaphsonAlgo._local_train`
                method to compute the local gradients and Hessian and sends them to the aggregator.
            * a L2 regularization can be applied to the loss by settings `l2_coeff` different to zero (default value)

        Args:
            data_from_opener (typing.Any): Input data returned by the ``get_data`` method from the opener.
            shared_state (NewtonRaphsonAveragedStates, Optional): Dict containing torch parameters that
                will be set to the model. Defaults to None.

        Returns:
            NewtonRaphsonSharedState: local gradients, local Hessian and the number of samples they were computed from.

        Raises:
            NegativeHessianMatrixError: Hessian matrix must be positive semi-definite to correspond to a convex problem.
        """

        # Create torch dataset
        train_dataset = self._dataset(data_from_opener, is_inference=False)

        if shared_state is None:
            # Instantiate the index_generator
            n_samples = len(train_dataset)

            self._index_generator = self._instantiate_index_generator(n_samples)

        else:
            assert self._index_generator.n_samples is not None

            # The shared states are the model parameter updates.
            # Hence we need to add it to the previous local state parameters.
            # unflatten_parameters_update = self._unflatten_tensor(
            #     shared_state.parameters_update, [p for p in self.model.parameters()]
            # )
            parameter_updates = [torch.from_numpy(x).to(self._device) for x in shared_state.parameters_update]
            weight_manager.increment_parameters(
                model=self._model,
                updates=parameter_updates,
                with_batch_norm_parameters=self._with_batch_norm_parameters,
            )
        self._index_generator.reset_counter()

        # Train mode for torch model
        self._model.train()

        self._final_gradients, self._final_hessian, self._n_samples_done = self._initialize_gradients_and_hessian()

        self._local_train(train_dataset)

        # Newton Raphson strategy must go through all the samples before each next update.
        assert self._index_generator.n_samples == self._n_samples_done

        eigenvalues = np.linalg.eig(self._final_hessian)[0].real
        if not (eigenvalues >= 0).all():
            raise NegativeHessianMatrixError(
                "Hessian matrix is not positive semi-definite, either the problem is not convex or due to numerical"
                " instability. It is advised to try to increase the l2_coeff. "
                f"Calculated eigenvalues are {eigenvalues.tolist()} and considered l2_coeff is {self._l2_coeff}"
            )
        self._index_generator.check_num_updates()

        return NewtonRaphsonSharedState(
            n_samples=self._index_generator.n_samples,
            hessian=self._final_hessian,
            gradients=self._final_gradients,
        )

    def _jacobian(self, tensor_y: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        """Compute the Jacobian for each  the given tensor_y regarding the model parameters.

        Args:
            tensor_y (torch.Tensor): _description_
            create_graph (bool, optional): Create the graph to compute higher order derivative. Defaults to False.

        Returns:
            torch.Tensor: Jacobian.
        """
        jacobian = []
        flat_y = torch.cat([t.reshape(-1) for t in tensor_y])
        for y in flat_y:
            for param in self._model.parameters():
                if param.requires_grad:
                    (gradient,) = torch.autograd.grad(y, param, retain_graph=True, create_graph=create_graph)
                    jacobian.append(gradient)

        return jacobian

    def _hessian_shape(self, second_order_derivative: List[torch.Tensor]) -> torch.Tensor:
        """Reshape from the second order derivative to obtain the Hessian matrix.

        Args:
            second_order_derivative (List[torch.Tensor]): second order derivative of a tensor regarding the
            registered parameters

        Returns:
            torch.Tensor: Hessian matrix
        """
        hessian = torch.cat([t.reshape(-1) for t in second_order_derivative])
        return hessian.reshape(self._final_hessian.shape)

    def _compute_gradients_and_hessian(self, loss: torch.Tensor) -> Tuple[torch.Tensor]:
        """The compute_gradients_and_hessian function compute the gradients and the Hessian matrix of the parameters
        regarding the given loss, and outputs them.

        Note that the hessian outputted by pytorch is numerically symmetrized by averaging it with its transpose.

        Args:
            loss (torch.Tensor): the loss to compute the gradients and Hessian on.

        Returns:
            torch.Tensor: the computed gradients of the parameters regarding the loss, flattened into a 1d torch
            Tensor.
            torch.Tensor: the computed Hessian matrix of the parameters regarding the loss.
        """

        gradients = self._jacobian(loss[None], create_graph=True)
        second_order_derivative = self._jacobian(gradients)

        hessian = self._hessian_shape(second_order_derivative)

        hessian = 0.5 * hessian + 0.5 * hessian.T  # ensure the hessian is symmetric

        return gradients, hessian

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
