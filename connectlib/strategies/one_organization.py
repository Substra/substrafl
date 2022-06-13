import logging
from typing import List
from typing import Optional

from connectlib.algorithms import Algo
from connectlib.organizations import AggregationOrganization
from connectlib.organizations import TestDataOrganization
from connectlib.organizations import TrainDataOrganization
from connectlib.organizations.references.local_state import LocalStateRef
from connectlib.schemas import StrategyName
from connectlib.strategies.strategy import Strategy

logger = logging.getLogger(__name__)


class OneOrganization(Strategy):
    """One Organization strategy.

    One Organization is not a real federated strategy and it is rather used for testing as it is faster than other
    'real' strategies. The training and prediction are performed on a single Organization. However, the number of
    passes to that Organization (num_rounds) is still defined to test the actual federated setting.
    In OneOrganization strategy a single client ``TrainDataOrganization`` and ``TestDataOrganization`` performs
    all the model execution.
    """

    def __init__(self):
        super(OneOrganization, self).__init__()

        # State
        self.local_state: Optional[LocalStateRef] = None

    @property
    def name(self) -> StrategyName:
        """The name of the strategy

        Returns:
            StrategyName: Name of the strategy
        """
        return StrategyName.ONE_ORGANIZATION

    def perform_round(
        self,
        algo: Algo,
        train_data_organizations: List[TrainDataOrganization],
        round_idx: int,
        aggregation_organization: Optional[AggregationOrganization] = None,
    ):
        """One round of the OneOrganization strategy: perform a local update (train on n mini-batches) of the models on a given
        data organization

        Args:
            algo (Algo): User defined algorithm: describes the model train and predict
            train_data_organizations (List[TrainDataOrganization]): List of the organizations on which to perform local
                updates aggregation_organization (AggregationOrganization): Should be None otherwise it will be ignored
            round_idx (int): Round number, it starts at 1.
        """
        if aggregation_organization is not None:
            logger.info("Aggregation organizations are ignored for decentralized strategies.")

        n_train_data_organizations = len(train_data_organizations)
        if n_train_data_organizations != 1:
            raise ValueError(
                "One organization strategy can only be used with one train_data_organization"
                f" but {n_train_data_organizations} were passed."
            )

        # define composite tuples (do not submit yet)
        # for each composite tuple give description of Algo instead of a key for an algo
        next_local_state, _ = train_data_organizations[0].update_states(
            algo.train(  # type: ignore
                train_data_organizations[0].data_sample_keys,
                shared_state=None,
                _algo_name=f"Training with {algo.__class__.__name__}",
            ),
            local_state=self.local_state,
            round_idx=round_idx,
        )

        # keep the states in a list: one/organization
        self.local_state = next_local_state

    def predict(
        self,
        test_data_organizations: List[TestDataOrganization],
        train_data_organizations: List[TrainDataOrganization],
        round_idx: int,
    ):
        if len(train_data_organizations) != 1:
            raise ValueError(
                "One organization strategy can only be used with one train_data_organization but"
                f" {len(train_data_organizations)} were passed."
            )

        for test_organization in test_data_organizations:

            if train_data_organizations[0].organization_id != test_organization.organization_id:
                raise NotImplementedError("Cannot test on a organization we did not train on for now.")
            # Init state for testtuple
            test_organization.update_states(traintuple_id=self.local_state.key, round_idx=round_idx)
