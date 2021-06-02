import uuid
import substra

from typing import List, Optional

from .strategy import Strategy
from .aggregators import AvgAggregator
from .aggregators.substra_utils import add_aggregator
from ..specs.node import NodeSpec


class FedAVG(Strategy):
    def __init__(self, num_rounds: int, num_updates: int):
        self.num_rounds = num_rounds
        self.num_updates = num_updates

        self.aggregator_key = None

    def register_aggregator(
        self, client: substra.Client, permisions: substra.sdk.schemas.Permissions
    ) -> str:
        return add_aggregator(client, AvgAggregator, permisions)

    def perform_round(
        self,
        client: substra.Client,
        algo_key: str,
        agg_key: Optional[str],
        node_specs: List[NodeSpec],
    ):
        node_ids = [node_spec.node_id for node_spec in node_specs]

        permissions = substra.sdk.schemas.Permissions(public=False, authorized_ids=node_ids)

        if self.aggregator_key is None:
            self.aggregator_key = self.register_aggregator(client, permisions=permissions)

        composite_traintuple = [
            {
                "algo_key": algo_key,
                "data_manager_key": node_spec.data_manager_key,
                "train_data_sample_keys": node_spec.data_sample_keys,
                "in_head_model_id": None,
                "in_trunk_model_id": agg_key,
                "out_trunk_model_permissions": permissions,
                "tag": node_spec.objective_name,
                "composite_traintuple_id": uuid.uuid4().hex,
            }
            for node_spec in node_specs
        ]

        previous_composite_ids = [ct["composite_traintuple_id"] for ct in composite_traintuple]

        aggregatetuple = {
            "algo_key": self.aggregator_key,
            "worker": node_ids[0],
            "in_models_ids": previous_composite_ids,
            "tag": node_specs[0].objective_name,
            "aggregatetuple_id": uuid.uuid4().hex,
        }

        return composite_traintuple, aggregatetuple

    # def perform_round(self, ...):
    #     results = []
    #     for node in node_specs:
    #         result = node.execute(algo.perform_round())
    #         results.append(result)
    #
    #     avg_result = aggregator_node.execute(self.aggregate_states(results))
    #
    #     return avg_result

    def run(
        self,
        client: substra.Client,
        algo_key: str,
        init_agg_key: str,
        node_specs: List[NodeSpec],
    ):
        for round in range(self.num_rounds):
            # TODO: add a way of selecting data samples inside node_specs
            # TODO: either pass indices or create new NodeSpec instances
            self.perform_round(client, algo_key, init_agg_key, node_specs=node_specs)


#
# class FedAVG:
#     ...
#
#     def perform_round(self, hospitals: List):
#
#         results = []
#         for hospital in hospitals:
#             result = perform_update(hospital)
#             results.append(result)
#
#         agg = aggregate(results)
#
#
# class RucheFedAVG:
#     ...
#
#     def perform_round(self):
#         updates = []
#         for i, client in enumerate(clients):
#             tokens_list = client.data_indexer.generate_tokens(self._num_updates)
#             client_update = client.compute_updates(tokens_list=tokens_list)
#
#             updates.append(client_update)
#             if verbose > 0:
#                 print(" - Client {} emitted updates".format(i))
#
#         # Transmit updates to server
#         agg_update = server.aggregate_updates(updates=updates)


class FedAvg:
    def perform_round(self, train_):
