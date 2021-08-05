import random

import numpy as np

from typing import Optional, List

from connectlib.algorithms import Algo
from connectlib.algorithms.algo import Weights
from connectlib.nodes import TrainDataNode, AggregationNode, TestDataNode
from connectlib.strategies.strategy import Strategy
from connectlib.remote import remote, remote_data


class FedAVG(Strategy):
    """[summary]

    Args:
        Strategy ([type]): [description]
    """

    def __init__(self, num_rounds: int, num_updates: int, batch_size: int):
        self.num_rounds = num_rounds
        self.num_updates = num_updates
        self.batch_size = batch_size
        seed = 42
        super(FedAVG, self).__init__(
            num_rounds=num_rounds, num_updates=num_updates, batch_size=batch_size, seed=seed
        )

        # States
        self.local_states = None
        self.avg_shared_state = None

    @remote
    def avg_shared_states(self, shared_states: List[Weights]) -> Weights:
        # get keys
        keys = shared_states[0].keys()

        # average weights
        averaged_states = {}
        for key in keys:
            states = np.stack([state[key] for state in shared_states])
            averaged_states[key] = np.mean(states, axis=0)

        return averaged_states

    @remote_data
    def data_indexer(
        self,
        x,
        y,
        num_rounds: int,
        num_updates: int,
        batch_size: int,
        seed: Optional[int] = 42,
        drop_last: Optional[bool] = None,
        shared_state: Optional = None,
    ):
        """Remotly (on the node) generates the batch_indices from the given data

        Args:
            x (sequence): x data, only for consistency purposes, not used
                in the function
            y (sequence): y data, the length of this data
                is used to calculate the indices
            num_rounds (int): number of rounds
            num_updates (int): number of batches in each round
            batch_size (int): number of data points in each batch
            drop_last (bool): if set to True it will ignore the last, not full batch
                if set to False it will make the last batch smaller
            shared_state: used only for consistency purposes

            return (dictionary): the structure is:
                {"minibatch_indices": minibatch_indices,
                 "index": 0}
                where `minibatch_indices` is a list of rounds of list
                of updates of `batch_size`, and each batch is filled
                with integer indices.

            The indices of the data are shuffled. The data is used to fill the batches until
            a full batch cannot be made. The next batch will be shorter if `drop_last` is False.
            The data is reshuffled and used again to fill next batches until `num_rounds` of
            `num_updates` of batches is satisfied.

            for example:
                if the size of the data is 7, num_rounds = 3, num_updates=2 and batch_size=3,
                and drop_last set to False,
                the resulting minibatch_indices list will be:
                1. first list of rounds of 2 lists
                2. lists of updates with 2 lists of sizes
                   either 3 or 1 (because drop_last is set to False)
                therefore the minibatch_indices list will look as follows:
                [[[id, id, id], [id, id, id]],
                 [[id], [id2, id2, id2]],
                 [[id2, id2, id2], [id2]]]
                 where id are indices and id2 are reshufled indices

                if, in the same case drop_last is set to True,
                the resulting minibatch_indices will be:
                [[[id, id, id], [id, id, id]],
                 [[id2, id2, id2], [id2, id2, id2]],
                 [[id3, id3, id3], [id3, id3, id3]]]
                where id2 and id3 are indices drawn from the second and third pass on the data
                and one data point is not used on each of the pass

        """
        data_len = len(y)

        # if batch_size is larger than data size and drop_last is True we raise an error
        if batch_size > data_len and drop_last:
            raise ValueError(
                "batch_size cannot be larger "
                "than length of the data "
                "if drop_last is set to True"
            )
        random.seed(seed)
        indices = list(range(data_len))
        minibatch_indices = list()
        random.shuffle(indices)
        idx = 0
        for _ in range(num_rounds):
            round_indices = list()
            for _ in range(num_updates):
                if drop_last and idx + batch_size > data_len:
                    # Not enough samples to do a full batch, we shuffle
                    # and go from the beginning again
                    idx = 0
                    random.shuffle(indices)
                    round_indices.append(indices[idx : idx + batch_size])
                    idx = idx + batch_size
                else:
                    round_indices.append(indices[idx : idx + batch_size])
                    if idx + batch_size >= data_len:
                        # we had just enough for one batch, we shuffle
                        # and go from the beginning again
                        idx = 0
                        random.shuffle(indices)
                    else:
                        idx = idx + batch_size

            minibatch_indices.append(round_indices)

        return {"minibatch_indices": minibatch_indices, "index": 0}

    def initialize(self, train_data_nodes: List[TrainDataNode]):
        """[summary]

        Args:
            train_data_nodes (List[TrainDataNode]): [description]
        """
        self.local_states = list()
        for node in train_data_nodes:
            next_local_state, _ = node.compute(
                self.data_indexer(
                    node.data_sample_keys,
                    shared_state=None,
                    num_rounds=self.num_rounds,
                    num_updates=self.num_updates,
                    batch_size=self.batch_size,
                )
            )
            self.local_states.append(next_local_state)

    def perform_round(
        self,
        algo: Algo,
        train_data_nodes: List[TrainDataNode],
        aggregation_node: AggregationNode,
    ):
        """One round of the Federated Averaging strategy:
        - if they exist, set the model weights to the aggregated weights on each train data nodes
        - perform a local update (train on n minibatches) of the models on each train data nodes
        - aggregate the model gradients

        Args:
            algo (Algo): User defined algorithm: describes the model train and predict
            train_data_nodes (List[TrainDataNode]): List of the nodes on which to perform local updates
            aggregation_node (AggregationNode): Node without data, used to perform operations on the shared states of the models
        """
        next_local_states = []
        states_to_aggregate = []
        for i, node in enumerate(train_data_nodes):
            previous_local_state = (
                self.local_states[i] if self.local_states is not None else None
            )

            # define composite tuples (do not submit yet)
            # for each composite tuple give description of Algo instead of a key for an algo
            next_local_state, next_shared_state = node.compute(
                algo.train(  # type: ignore
                    node.data_sample_keys,
                    shared_state=self.avg_shared_state,
                    num_updates=self.num_updates,
                ),
                local_state=previous_local_state,
            )
            # keep the states in a list: one/node
            next_local_states.append(next_local_state)
            states_to_aggregate.append(next_shared_state)

        avg_shared_state = aggregation_node.compute(
            self.avg_shared_states(shared_states=states_to_aggregate)  # type: ignore
        )

        self.local_states = next_local_states
        self.avg_shared_state = avg_shared_state

    def predict(
        self,
        algo: Algo,
        test_data_nodes: List[TestDataNode],
        train_data_nodes: List[TrainDataNode],
    ):
        # TODO: REMOVE WHEN HACK IS FIXED
        traintuple_ids = []
        for i, node in enumerate(train_data_nodes):
            previous_local_state = (
                self.local_states[i] if self.local_states is not None else None
            )

            assert previous_local_state is not None

            traintuple_id_ref, _ = node.compute(
                # here we could also use algo.train or whatever method marked as @remote_data
                # in the algo
                # because fake_traintuple is true so the method name and the method
                # are not used
                algo.predict(  # type: ignore
                    [node.data_sample_keys[0]],
                    shared_state=self.avg_shared_state,
                    fake_traintuple=True,
                ),
                local_state=previous_local_state,
            )
            traintuple_ids.append(traintuple_id_ref.key)

        for i, node in enumerate(test_data_nodes):
            node.compute(traintuple_ids[i])  # compute testtuple
