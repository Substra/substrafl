import datetime
import substra

from typing import List

from connectlib.algorithms import RegisteredAlgo
from connectlib.strategies import FedAVG

from .algorithms.substra_utils import add_algo
from .specs.node import NodeSpec
from .strategies.aggregators.init_aggregator import InitializationAggregator
from .strategies.aggregators.substra_utils import add_aggregator


class Orchestrator:
    def __init__(self, algo: RegisteredAlgo, strategy: FedAVG, num_rounds: int):
        self.algo = algo
        self.strategy = strategy
        self.num_rounds = num_rounds

    def register_init_aggregator(
        self, client: substra.Client, permisions: substra.sdk.schemas.Permissions
    ) -> str:
        return add_aggregator(client, InitializationAggregator, permisions)

    def run(self, client: substra.Client, node_specs: List[NodeSpec]):
        node_ids = [node_spec.node_id for node_spec in node_specs]

        permissions = substra.sdk.schemas.Permissions(public=False, authorized_ids=node_ids)

        algo_key = add_algo(client, self.algo, permissions)

        composite_traintuples, aggregatetuple = self.strategy.perform_round(
            client=client, algo_key=algo_key, agg_key=None, node_specs=node_specs
        )

        compute_plan = client.add_compute_plan(
            {
                "composite_traintuples": composite_traintuples,
                "aggregatetuples": [aggregatetuple],
                "tag": str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
            }
        )

        # TODO: wait on compute_plan

        return compute_plan
