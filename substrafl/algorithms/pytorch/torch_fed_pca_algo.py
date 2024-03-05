import math
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional

import torch

from substrafl.algorithms.pytorch import weight_manager
from substrafl.algorithms.pytorch.torch_base_algo import TorchAlgo
from substrafl.index_generator import NpIndexGenerator
from substrafl.remote import remote_data
from substrafl.strategies.schemas import FedPCAAveragedState
from substrafl.strategies.schemas import FedPCASharedState
from substrafl.strategies.schemas import StrategyName


class TorchLinearModel(torch.nn.Module):
    """Define linear model to encapsulate eigenvectors.

    Args:
        in_features (int): dimension of input vectors
        out_features (int): dimension to keep as part of dimensionality
            reduction
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eigen_vectors = torch.nn.Linear(self.in_features, self.out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs dimensionality reduction.

        Args:
            x (torch.Tensor): inputs to map to reduced dim space.
                shape of x is (B, D_IN)

        Returns:
            torch.Tensor: reduced dim vectors
        """
        eigen_vectors = self.eigen_vectors.weight.data
        projection = x @ eigen_vectors.T
        return projection


class TorchFedPCAAlgo(TorchAlgo):
    """To be inherited. Wraps the necessary operation so a torch model can be trained in the Federated PCA
    strategy.

    The ``train`` method:

        - computes the local mean during the first round
        - computes the covariance matrix during the second round
        - computes the eigen vectors regarding the shared covariance matrix for all next rounds

    The ``predict`` method generates the eigen vectors.

    To add a custom parameter to the ``__init__`` of the class, also add it to the call to ``super().__init__``
    as shown in the example with ``my_custom_extra_parameter``. Only primitive types (str, int, ...) are supported
    for extra parameters.

    Example:

        .. code-block:: python

            class MyAlgo(TorchFedPCAAlgo):
                def __init__(
                    self,
                    my_custom_extra_parameter,
                ):
                    super().__init__(
                        in_features=10,
                        out_features=2,
                        batch_size=16,
                        dataset=my_dataset,
                        seed=seed,
                        my_custom_extra_parameter=my_custom_extra_parameter,
                    )
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        in_features: int,
        out_features: int,
        batch_size: Optional[int] = None,
        seed: int = 1,
        use_gpu: bool = True,
        *args,
        **kwargs,
    ):
        """The ``__init__`` function is called at each call of the `train()` or `predict()` function
        Some attributes will then be overwritten by their previous states in the `load()` function,
        before the `train()` or `predict()` function is run.

        Args:
            dataset (torch.utils.data.Dataset): input data on which to perform PCA
            in_features (int): input data dimensionality
            out_features (int): dimension to keep after PCA
            batch_size (Optional[int]): mini-batch size
            seed (int): random generator seed. The seed is mandatory. Default to 1.
            use_gpu (bool): whether to use GPU or not. Default to True.
        """
        self.in_features = in_features
        self.out_features = out_features
        self._batch_size = batch_size
        self.local_mean = None
        self.local_covmat = None
        self._seed = seed

        torch.manual_seed(self._seed)

        super().__init__(
            *args,
            model=TorchLinearModel(
                self.in_features,
                self.out_features,
            ),
            criterion=None,
            index_generator=None,
            dataset=dataset,
            seed=self._seed,
            use_gpu=use_gpu,
            **kwargs,
        )

    @property
    def strategies(self) -> List[StrategyName]:
        """List of compatible strategies.

        Returns:
            typing.List: typing.List[StrategyName]
        """
        return [StrategyName.FEDERATED_PCA]

    @property
    def eigen_vectors(self) -> torch.Tensor:
        """Current computed eigen vectors.

        Returns:
            torch.Tensor: eigen vectors
        """
        return self._model.eigen_vectors.weight.data

    def transform(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Project the input vector on the eigen vector subspace.
        Input tensor shape must be of shape (N, in_features). The returned tensor
        will be of shape (N, out_features).

        Args:
            input_tensor (torch.Tensor): input tensor to compute the dimension reduction on.

        Returns:
            torch.Tensor: projected tensor on the computed eigen vector dimension.
        """
        self._model.eval()

        return self._model(input_tensor)

    def _instantiate_index_generator(self, n_samples: int):
        """Create a generator for batches data points indices.

        Args:
            n_samples (int): the desired batch size.

        Returns:
            NpIndexGenerator: the index generator.
        """
        if self._batch_size is None:
            # If batch_size is None, it is set to the number of samples in the dataset by the index generator
            num_updates = 1
        else:
            num_updates = math.ceil(n_samples / self._batch_size)

        index_generator = NpIndexGenerator(batch_size=self._batch_size, num_updates=num_updates, drop_last=False)
        index_generator.n_samples = n_samples
        return index_generator

    def _compute_local_mean(self, train_data_loader: torch.utils.data.DataLoader):
        dataset_size = len(train_data_loader.dataset)
        self.local_mean = torch.zeros((self.in_features,)).to(self._device)

        for x_batch, _ in train_data_loader:
            x_batch = x_batch.to(self._device)
            self.local_mean += torch.sum(x_batch, dim=(0,))

        self.local_mean /= dataset_size
        return self.local_mean.cpu().numpy()

    def _compute_local_covmat(self, averaged_mean: torch.Tensor, train_data_loader: torch.utils.data.DataLoader):
        # Starting local covariate matrix computation
        self.local_covmat = torch.zeros((self.in_features, self.in_features)).to(self._device)

        for x_batch, _ in train_data_loader:
            x_batch = x_batch.to(self._device)
            # Centering input vectors
            repeated_means = averaged_mean.repeat(x_batch.shape[0], 1).to(self._device)
            x_batch -= repeated_means
            # Updating covariance matrix
            self.local_covmat += torch.matmul(x_batch.T, x_batch)
        # Initializing the weights for the subspace iteration
        initialization = torch.normal(
            torch.zeros((self.out_features, self.in_features)),
            torch.ones((self.out_features, self.in_features)),
            generator=torch.Generator().manual_seed(self._seed),
        ).to(self._device)

        new_parameters = torch.matmul(initialization.to(self._device), self.local_covmat).cpu().numpy()
        return new_parameters

    @remote_data
    def train(
        self,
        data_from_opener: Any,
        shared_state: Optional[FedPCAAveragedState] = None,  # Set to None per default for clarity reason as
        # the decorator will do it if the arg shared_state is not passed.
    ) -> FedPCASharedState:
        """Train the local model for one round of the federated algorithm.

        Important:
            This functions behaves differently depending on the round of the federated
            algorithm for PCA. In the first round, the mean vector is computed. In the
            second round, the covariance matrix is computed. The computation of the
            eigenvectors starts from round 3. A sufficient number of rounds is necessary
            for the method to produce accurate eigenvectors. This can be monitored
            through mean square reconstruction error which should reach a global
            minimum when the algorithm has converged.

        Args:
            data_from_opener (Any): input data
            shared_state (Optional[FedPCAAveragedState]): incoming FedPCAAveragedState
              obtained at the previous round of the federated algorithm (after
              aggregation). It contains the federatively learnt eigenvectors.
              Defaults to None.

        Returns:
            FedPCASharedState: updated model and parameters shared for aggregation.
        """
        # Create torch dataset
        train_dataset = self._dataset(data_from_opener, is_inference=False)

        if shared_state is None:
            # Instantiate the index_generator
            n_samples = len(train_dataset)
            self._index_generator = self._instantiate_index_generator(n_samples)

        else:
            assert self._index_generator.n_samples is not None

        self._index_generator.reset_counter()

        # Create torch dataloader
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=self._index_generator)

        if self.local_mean is None:
            # In round 1:
            #   - Computation of the local mean
            parameters_update = self._compute_local_mean(
                train_data_loader=train_data_loader,
            )

        elif self.local_covmat is None:
            # In round 2:
            #   - Computation of the local covariance matrix
            #   - Initialization of the weights for the subspace iteration method
            averaged_mean = torch.Tensor(shared_state.avg_parameters_update[0])

            parameters_update = self._compute_local_covmat(
                averaged_mean=averaged_mean,
                train_data_loader=train_data_loader,
            )

        else:
            averaged_parameters = torch.Tensor(shared_state.avg_parameters_update[0])

            weight_manager.set_parameters(
                model=self._model,
                parameters=[averaged_parameters],
                with_batch_norm_parameters=False,
            )

            parameters_update = torch.matmul(averaged_parameters.to(self._device), self.local_covmat).cpu().numpy()

        return FedPCASharedState(n_samples=len(train_dataset), parameters_update=[parameters_update])

    def predict(self, data_from_opener: Any, shared_state: Any = None) -> torch.Tensor:
        """Execute the following operations:

            * Create the test torch dataset.
            * Execute the reduction dimension of the test dataset, and save them as
              predictions on the prediction path.

        Args:
            data_from_opener (typing.Any): Input data
            shared_state (typing.Any): Latest train task shared state (output of the train method)

        Returns:
            torch.Tensor: The computed predictions.
        """

        # Create torch dataset
        predict_dataset = self._dataset(data_from_opener, is_inference=True)

        dataloader_batchsize = min(self._batch_size, len(predict_dataset)) if self._batch_size else len(predict_dataset)
        predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=dataloader_batchsize)

        predictions = torch.Tensor([])
        with torch.no_grad():
            for x in predict_loader:
                x = x.to(self._device)
                predictions = torch.cat((predictions, self.transform(x)), 0)

        predictions = predictions.cpu().detach()

        return predictions

    def _get_state_to_save(self) -> dict:
        """Create the algo checkpoint: a dictionary saved with ``torch.save`` using the
        parent class.

        The parent class saves the TorchLinearModel containing the eigenvectors. In this
        algorithm, we additionally need to save the sample mean and covariance matrix
        as well as the round index as the self.train function behaves differently based
        on this round index.

        Returns:
            dict: checkpoint to save
        """
        checkpoint = super()._get_state_to_save()
        checkpoint.update(
            {
                "mean": self.local_mean,
                "covariance_matrix": self.local_covmat,
            }
        )
        return checkpoint

    def _update_from_checkpoint(self, path: Path) -> dict:
        """Load the checkpoint and update the internal state from it.

        Pop the values from the checkpoint so that we can ensure that it is empty at the
        end, i.e. all the values have been used. For the specific case of PCA computation,
        the sample mean, covariance matrix and federated algorithm round index also need
        to be popped.

        Args:
            path (pathlib.Path): path where the checkpoint is saved

        Returns:
            dict: checkpoint
        """
        checkpoint = super()._update_from_checkpoint(path)
        self.local_mean = checkpoint.pop("mean")
        self.local_covmat = checkpoint.pop("covariance_matrix")
        return checkpoint
