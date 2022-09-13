import logging
from typing import List
from typing import Optional

from substrafl.algorithms import Algo
from substrafl.nodes import AggregationNode
from substrafl.nodes import TestDataNode
from substrafl.nodes import TrainDataNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.schemas import StrategyName
from substrafl.strategies.strategy import Strategy

logger = logging.getLogger(__name__)


class SingleOrganization(Strategy):
    """Single organization strategy.

    Single organization is not a real federated strategy and it is rather used for testing as it is faster than other
    'real' strategies. The training and prediction are performed on a single Node. However, the number of
    passes to that Node (num_rounds) is still defined to test the actual federated setting.
    In SingleOrganization strategy a single client ``TrainDataNode`` and ``TestDataNode`` performs
    all the model execution.
    """

    def __init__(self):
        super(SingleOrganization, self).__init__()

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
        train_data_nodes: List[TrainDataNode],
        round_idx: int,
        clean_models: bool,
        aggregation_node: Optional[AggregationNode] = None,
    ):
        """One round of the SingleOrganization strategy: perform a local update (train on n mini-batches) of the models
        on a given data node

        Args:
            algo (Algo): User defined algorithm: describes the model train and predict
            train_data_nodes (List[TrainDataNode]): List of the nodes on which to perform local
                updates, there should be exactly one item in the list.
            aggregation_node (AggregationNode): Should be None otherwise it will be ignored
            round_idx (int): Round number, it starts at 1.
            clean_models (bool): Clean the intermediary models of this round on the Substra platform.
                Set it to False if you want to download or re-use intermediary models. This causes the disk
                space to fill quickly so should be set to True unless needed.
        """
        if aggregation_node is not None:
            logger.info("Aggregation nodes are ignored for decentralized strategies.")

        n_train_data_nodes = len(train_data_nodes)
        if n_train_data_nodes != 1:
            raise ValueError(
                "One organization strategy can only be used with one train_data_node"
                f" but {n_train_data_nodes} were passed."
            )

        # define composite tuples (do not submit yet)
        # for each composite tuple give description of Algo instead of a key for an algo
        next_local_state, _ = train_data_nodes[0].update_states(
            algo.train(  # type: ignore
                train_data_nodes[0].data_sample_keys,
                shared_state=None,
                _algo_name=f"Training with {algo.__class__.__name__}",
            ),
            local_state=self.local_state,
            round_idx=round_idx,
            authorized_ids=[train_data_nodes[0].organization_id],
            clean_models=clean_models,
        )

        # keep the states in a list: one/organization
        self.local_state = next_local_state

    def predict(
        self,
        algo: Algo,
        test_data_nodes: List[TestDataNode],
        train_data_nodes: List[TrainDataNode],
        round_idx: int,
    ):
        if len(train_data_nodes) != 1:
            raise ValueError(
                "Single organization strategy can only be used with one train_data_node but"
                f" {len(train_data_nodes)} were passed."
            )

        for test_data_node in test_data_nodes:

            if train_data_nodes[0].organization_id != test_data_node.organization_id:
                raise NotImplementedError("Cannot test on a organization we did not train on for now.")
            # Init state for testtuple
            test_data_node.update_states(
                traintuple_id=self.local_state.key,
                operation=algo.predict(
                    data_samples=test_data_node.test_data_sample_keys,
                    shared_state=None,
                    _algo_name=f"Testing with {algo.__class__.__name__}",
                ),
                round_idx=round_idx,
            )  # Init state for testtuple
