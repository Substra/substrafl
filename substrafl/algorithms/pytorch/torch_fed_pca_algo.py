import logging
import math
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional

import torch

from substrafl import NpIndexGenerator
from substrafl.algorithms.pytorch import weight_manager
from substrafl.algorithms.pytorch.torch_base_algo import TorchAlgo
from substrafl.remote import remote_data
from substrafl.schemas import FedAvgAveragedState
from substrafl.schemas import FedAvgSharedState
from substrafl.schemas import StrategyName

logger = logging.getLogger(__name__)


class TorchLinearModel(torch.nn.Module):
    """Define linear model to encapsulate eigenvectors.

    Args:
        in_features (int): dimension of input vectors
        out_features (int): number of dimensions to keep as part of dimensionsality
            reduction
        device (str): working device, cuda or cpu
    """

    def __init__(self, in_features: int, out_features: int, device: str):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.eigen_vectors = torch.nn.Linear(self.in_features, self.out_features, bias=False).to(self.device)
        self.eigen_vectors.weight.data.to(self.device)

    def forward(self, x):
        """Perform dimensionality reduction.

        Args:
            x (torch.Tensor): inputs to map to reduced dim space.
                shape of x is (B, D_IN)

        Returns:
            torch.Tensor: reduced dim vectors
        """
        eigen_vectors = self.eigen_vectors.weight.data
        coefs = x @ eigen_vectors.T
        return coefs @ eigen_vectors


class TorchFedPCAAlgo(TorchAlgo):
    """TorchFedPCAAlgo class, inheriting from TorchAlgo and designed to perform Principal Component Analysis (PCA).

    Args:
        dataset (torch.utils.data.Dataset): input data on which to perform PCA
        in_features (int): input data dimensionality
        out_features (int): number of dimensions to keep after PCA
        batch_size (Optional[int]): mini-batch size
        seed (int): random generator seed. Default to 1.
        use_gpu (bool): whether to use GPU or not
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
        self.in_features = in_features
        self.out_features = out_features
        self._batch_size = batch_size
        self.local_mean = None
        self.local_covmat = None
        self.round_counter = 0
        self._seed = seed
        torch.manual_seed(self._seed)

        super().__init__(
            model=TorchLinearModel(
                self.in_features,
                self.out_features,
                device=self._get_torch_device(use_gpu),
            ),
            criterion=None,
            optimizer=None,
            index_generator=None,
            dataset=dataset,
            scheduler=None,
            seed=self._seed,
            use_gpu=use_gpu,
            *args,
            **kwargs,
        )

    def _local_predict(self, predict_dataset: torch.utils.data.Dataset, predictions_path: Path):
        """Map inputs to lower dimensional space.

        Args:
            predict_dataset (torch.utils.data.Dataset): predict_dataset build from the x returned by the opener.
            predictions_path (os.PathLike): path where to save predictions.

        Important:
            This function shall not be used before the second round to Fed PCA algo is
            completed. Before that, the covariance matrix is not built.

        """
        if self.round_counter <= 2:
            logger.warning(f"Evaluation ignored at round zero and one for {self.name} (pre-processing rounds).")
        else:
            super(TorchFedPCAAlgo, self)._local_predict(predict_dataset, predictions_path)

    @property
    def strategies(self) -> List[StrategyName]:
        """List of compatible strategies.

        Returns:
            typing.List: typing.List[StrategyName]
        """
        return [StrategyName.FEDERATED_PCA]

    def _instantiate_index_generator(self, n_samples):
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

    @remote_data
    def train(
        self,
        datasamples: Any,
        shared_state: Optional[FedAvgAveragedState] = None,  # Set to None per default for clarity reason as
        # the decorator will do it if the arg shared_state is not passed.
    ) -> FedAvgSharedState:
        """Train the local model for one round of the federated algorithm.

        Args:
            datasamples (Any): input data
            shared_state (Optional[FedAvgAveragedState]): incoming FedAvgAveragedState
            obtained at the previous round of the federated algorihtm (after
            aggregation). It contains the federatively learnt eigenvectors.
            Defaults to None.

        Important:
            this functions behaves differently depending on the round of the federated
            algorithm for PCA. In the first round, the mean vector is computed. In the
            second round, the covariance matrix is computed. The computation of the
            eigenvectors starts from round 3. Sufficiently many rounds are necessary
            for the method to produce accurate eigenvectors. This can be monitored
            through mean square reconstruction error which should reach a global
            minimum when the algorithm has converged.

        Returns:
            FedAvgSharedState: updated model and parameters shared for aggregation.
        """
        # Create torch dataset
        train_dataset = self._dataset(datasamples, is_inference=False)

        if shared_state is None:
            # Instantiate the index_generator
            n_samples = len(train_dataset)
            self._index_generator = self._instantiate_index_generator(n_samples)

        else:
            assert self._index_generator.n_samples is not None

        self._index_generator.reset_counter()

        # Create torch dataloader
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=self._index_generator)

        if shared_state is not None:
            # The shared states are the model parameters.
            # Hence we need to assign them to the previous local state parameters.
            weight_manager.set_parameters(
                model=self._model,
                parameters=[torch.Tensor(shared_state.avg_parameters_update[0])],
                with_batch_norm_parameters=False,
            )

        old_parameters = weight_manager.get_parameters(
            model=self._model,
            with_batch_norm_parameters=False,
        )

        if self.local_mean is None:
            # In round 0
            new_parameters = old_parameters[0].cpu().numpy()
            self.local_mean = torch.zeros((self._model.in_features,)).to(self._device)
            self.local_n = 0
            for x_batch, _ in train_data_loader:
                x_batch = x_batch.to(self._device)
                self.local_mean += torch.sum(x_batch, dim=(0,))
                self.local_n += x_batch.shape[0]

            self.local_mean /= self.local_n
            # Using the model parameters as a container for local_mean to be aggregated
            new_parameters[0] = self.local_mean.cpu().numpy()
            self.round_counter += 1
            return FedAvgSharedState(n_samples=self.local_n, parameters_update=[new_parameters])
        elif self.local_covmat is None:
            # In round 1 we are:
            #   - Computing the local covariance matrix
            #   - Initializing the weights for the subspace iteration method
            #      and storing them in old_parameters

            # Replacing the local mean by the aggregated one
            self.local_mean = old_parameters[0][0]
            # Fill new parameters with an arbitrary numpy array of correct shape
            # Starting local covariate matrix computation
            self.local_covmat = torch.zeros((self._model.in_features, self._model.in_features)).to(self._device)
            self.local_n = 0
            for x_batch, _ in train_data_loader:
                x_batch = x_batch.to(self._device)
                # Centering input vectors
                rep_means = self.local_mean.repeat(x_batch.shape[0], 1).to(self._device)
                x_batch -= rep_means
                self.local_n += x_batch.shape[0]
                # Updating cov matrix
                self.local_covmat += torch.matmul(x_batch.T, x_batch)

            # Initializing the weights for the subspace iteration
            old_parameters[0] = torch.normal(
                torch.zeros(old_parameters[0].shape),
                torch.ones(old_parameters[0].shape),
                generator=torch.Generator().manual_seed(self._seed),
            ).to(self._device)

        new_parameters = torch.matmul(old_parameters[0].to(self._device), self.local_covmat).cpu().numpy()

        # Assigning orthonormalized parameters
        weight_manager.set_parameters(
            model=self._model,
            parameters=old_parameters,
            with_batch_norm_parameters=False,
        )

        self.round_counter += 1
        return FedAvgSharedState(n_samples=len(train_dataset), parameters_update=[new_parameters])

    def _get_state_to_save(self) -> dict:
        """Create the algo checkpoint: a dictionary saved with ``torch.save`` using the
        parent class.

        The parent class saves the TorchLinearModel containing the eigenvectors. In this
        algorithm, we additionnally need to save the sample mean and covariance matrix
        as well as the round index as the self.train function behaves differently based
        on this round index.

        Returns:
            dict: checkpoint to save
        """
        checkpoint = super(TorchFedPCAAlgo, self)._get_state_to_save()
        checkpoint.update(
            {
                "mean": self.local_mean,
                "cov": self.local_covmat,
                "round_counter": self.round_counter,
            }
        )
        return checkpoint

    def _update_from_checkpoint(self, path: Path) -> dict:
        """Load the checkpoint and update the internal state from it.

        Pop the values from the checkpoint so that we can ensure that it is empty at the
        end, ie all the values have been used. For the specific case of PCA computation,
        the sample mean, covariance matrix and federated algorithm round index also need
        to be popped.

        Args:
            path (pathlib.Path): path where the checkpoint is saved

        Returns:
            dict: checkpoint
        """
        checkpoint = super(TorchFedPCAAlgo, self)._update_from_checkpoint(path)
        self.local_mean = checkpoint.pop("mean")
        self.local_covmat = checkpoint.pop("cov")
        self.round_counter = checkpoint.pop("round_counter")
        return checkpoint