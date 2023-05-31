import logging
from typing import List
from typing import Optional

from substrafl.algorithms import Algo
from substrafl.nodes import AggregationNode
from substrafl.nodes import TestDataNode
from substrafl.nodes import TrainDataNode
from substrafl.nodes.references.local_state import LocalStateRef
from substrafl.strategies.schemas import StrategyName
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

    def __init__(self, algo: Algo):
        """
        Args:
            algo (Algo): The algorithm your strategy will execute (i.e. train and test on all the specified nodes)
        """
        super().__init__(algo=algo)

        # State
        self.local_state: Optional[LocalStateRef] = None

    @property
    def name(self) -> StrategyName:
        """The name of the strategy

        Returns:
            StrategyName: Name of the strategy
        """
        return StrategyName.SINGLE_ORGANIZATION

    def initialization_round(
        self,
        *,
        train_data_nodes: List[TrainDataNode],
        clean_models: bool,
        round_idx: Optional[int] = 0,
        additional_orgs_permissions: Optional[set] = None,
    ):
        """Call the initialize function of the algo on each train node.

        Args:
            train_data_nodes (typing.List[TrainDataNode]): list of the train organizations
            clean_models (bool): Clean the intermediary models of this round on the Substra platform.
                Set it to False if you want to download or re-use intermediary models. This causes the disk
                space to fill quickly so should be set to True unless needed.
            round_idx (typing.Optional[int]): index of the round. Defaults to 0.
            additional_orgs_permissions (typing.Optional[set]): Additional permissions to give to the model outputs
                after training, in order to test the model on an other organization. Default to None
        """
        n_train_data_nodes = len(train_data_nodes)
        if n_train_data_nodes != 1:
            raise ValueError(
                "One organization strategy can only be used with one train_data_node"
                f" but {n_train_data_nodes} were passed."
            )

        next_local_state = train_data_nodes[0].init_states(
            operation=self.algo.initialize(
                _algo_name=f"Initializing with {self.algo.__class__.__name__}",
            ),
            round_idx=round_idx,
            authorized_ids=set([train_data_nodes[0].organization_id]) | additional_orgs_permissions,
            clean_models=clean_models,
        )

        self.local_state = next_local_state

    def perform_round(
        self,
        *,
        train_data_nodes: List[TrainDataNode],
        round_idx: int,
        clean_models: bool,
        aggregation_node: Optional[AggregationNode] = None,
        additional_orgs_permissions: Optional[set] = None,
    ):
        """One round of the SingleOrganization strategy: perform a local update (train on n mini-batches) of the models
        on a given data node

        Args:
            train_data_nodes (List[TrainDataNode]): List of the nodes on which to perform local
                updates, there should be exactly one item in the list.
            aggregation_node (AggregationNode): Should be None otherwise it will be ignored
            round_idx (int): Round number, it starts at 0.
            clean_models (bool): Clean the intermediary models of this round on the Substra platform.
                Set it to False if you want to download or re-use intermediary models. This causes the disk
                space to fill quickly so should be set to True unless needed.
            additional_orgs_permissions (typing.Optional[set]): Additional permissions to give to the model outputs
                after training, in order to test the model on an other organization.
        """

        if aggregation_node is not None:
            logger.info("Aggregation nodes are ignored for decentralized strategies.")

        n_train_data_nodes = len(train_data_nodes)
        if n_train_data_nodes != 1:
            raise ValueError(
                "One organization strategy can only be used with one train_data_node"
                f" but {n_train_data_nodes} were passed."
            )

        # define train tasks (do not submit yet)
        # for each train task give description of Algo instead of a key for an algo
        next_local_state, _ = train_data_nodes[0].update_states(
            operation=self.algo.train(
                train_data_nodes[0].data_sample_keys,
                shared_state=None,
                _algo_name=f"Training with {self.algo.__class__.__name__}",
            ),
            local_state=self.local_state,
            round_idx=round_idx,
            authorized_ids=set([train_data_nodes[0].organization_id]) | additional_orgs_permissions or set(),
            clean_models=clean_models,
        )

        # keep the states in a list: one/organization
        self.local_state = next_local_state

    def perform_predict(
        self,
        test_data_nodes: List[TestDataNode],
        train_data_nodes: List[TrainDataNode],
        round_idx: int,
    ):
        """Perform prediction on test_data_nodes.

        Args:
            test_data_nodes (List[TestDataNode]): test data nodes to perform the prediction from the algo on.
            train_data_nodes (List[TrainDataNode]): train data nodes the model has been trained
                on.
            round_idx (int): round index.
        """
        if len(train_data_nodes) != 1:
            raise ValueError(
                "Single organization strategy can only be used with one train_data_node but"
                f" {len(train_data_nodes)} were passed."
            )

        for test_data_node in test_data_nodes:
            # Init state for testtask
            test_data_node.update_states(
                traintask_id=self.local_state.key,
                operation=self.algo.predict(
                    data_samples=test_data_node.test_data_sample_keys,
                    _algo_name=f"Predicting with {self.algo.__class__.__name__}",
                ),
                round_idx=round_idx,
            )  # Init state for testtask
