import uuid
import substra

from typing import List

from .strategy import Strategy
from .aggregators import AvgAggregator
from .aggregators.substra_utils import add_aggregator
from ..orchestrator import NodeSpec


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
        self, client: substra.Client, algo_key: str, agg_key: str, node_specs: List[NodeSpec]
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
