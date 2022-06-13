from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional
from typing import TypeVar

from connectlib.algorithms.algo import Algo
from connectlib.organizations.aggregation_organization import AggregationOrganization
from connectlib.organizations.test_data_organization import TestDataOrganization
from connectlib.organizations.train_data_organization import TrainDataOrganization
from connectlib.schemas import StrategyName

SharedState = TypeVar("SharedState")


class Strategy(ABC):
    """Base strategy to be inherited from connectlib strategies."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @property
    @abstractmethod
    def name(self) -> StrategyName:
        """The name of the strategy

        Returns:
            StrategyName: Name of the strategy
        """
        raise NotImplementedError

    @abstractmethod
    def perform_round(
        self,
        algo: Algo,
        train_data_organizations: List[TrainDataOrganization],
        aggregation_organization: Optional[AggregationOrganization],
        round_idx: int,
    ):
        """Perform one round of the strategy

        Args:
            algo (Algo): algo with the code to execute on the organization
            train_data_organizations (typing.List[TrainDataOrganization]): list of the train organizations
            aggregation_organization (typing.Optional[AggregationOrganization]): aggregation organization, necessary for
                centralized strategy, unused otherwise
            round_idx (int): index of the round
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        test_data_organizations: List[TestDataOrganization],
        train_data_organizations: List[TrainDataOrganization],
        round_idx: int,
    ):
        """Predict function of the strategy: evaluate the model.
        Gets the model for a train organization and evaluate it on the
        test organizations.

        Args:
            test_data_organizations (typing.List[TestDataOrganization]): list of organizations on which to evaluate
            train_data_organizations (typing.List[TrainDataOrganization]): list of organizations on which the model has
            been trained round_idx (int): index of the round
        """
        raise NotImplementedError
