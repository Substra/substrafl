import logging
from typing import List
from typing import Optional

from connectlib.algorithms import Algo
from connectlib.nodes import AggregationNode
from connectlib.nodes import TestDataNode
from connectlib.nodes import TrainDataNode
from connectlib.nodes.references.local_state import LocalStateRef
from connectlib.schemas import StrategyName
from connectlib.strategies.strategy import Strategy

logger = logging.getLogger(__name__)


class OneNode(Strategy):
    """One Node strategy.

    One Node is not a real federated strategy and it is rather used for testing as it is faster than other 'real'
    strategies. The training and prediction are performed on a single Node. However, the number of passes to
    that Node (num_rounds) is still defined to test the actual federated setting.
    In OneNode strategy a single client ``TrainDataNode`` and ``TestDataNode`` performs all the model
    execution.
    """

    def __init__(self):
        super(OneNode, self).__init__()

        # State
        self.local_state: Optional[LocalStateRef] = None

    @property
    def name(self) -> StrategyName:
        """The name of the strategy

        Returns:
            StrategyName: Name of the strategy
        """
        return StrategyName.ONE_NODE

    def perform_round(
        self,
        algo: Algo,
        train_data_nodes: List[TrainDataNode],
        round_idx: int,
        aggregation_node: Optional[AggregationNode] = None,
    ):
        """One round of the OneNode strategy: perform a local update (train on n mini-batches) of the models on a given
        data node

        Args:
            algo (Algo): User defined algorithm: describes the model train and predict
            train_data_nodes (List[TrainDataNode]): List of the nodes on which to perform local updates
            aggregation_node (AggregationNode): Should be None otherwise it will be ignored
            round_idx (int): Round number, it starts by zero.
        """
        if aggregation_node is not None:
            logger.info("Aggregation nodes are ignored for decentralized strategies.")

        n_train_data_nodes = len(train_data_nodes)
        if n_train_data_nodes != 1:
            raise ValueError(
                f"One node strategy can only be used with one train_data_node but {n_train_data_nodes} were passed."
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
        )

        # keep the states in a list: one/node
        self.local_state = next_local_state

    def predict(
        self,
        test_data_nodes: List[TestDataNode],
        train_data_nodes: List[TrainDataNode],
        round_idx: int,
    ):
        if len(train_data_nodes) != 1:
            raise ValueError(
                f"One node strategy can only be used with one train_data_node but {len(train_data_nodes)} were passed."
            )

        for test_node in test_data_nodes:

            if train_data_nodes[0].node_id != test_node.node_id:
                raise NotImplementedError("Cannot test on a node we did not train on for now.")
            # Init state for testtuple
            test_node.update_states(traintuple_id=self.local_state.key, round_idx=round_idx)
