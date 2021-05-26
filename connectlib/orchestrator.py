import substra

from typing import List
from dataclasses import dataclass

from connectlib.algorithms import RegisteredAlgo
from connectlib.strategies import FedAVG

from .algorithms.substra_utils import add_algo


@dataclass
class NodeSpec:
    node_id: str
    data_manager_key: str
    data_sample_keys: List[str]
    objective_name: str


class Orchestrator:
    def __init__(self, algo: RegisteredAlgo, strategy: FedAVG, num_rounds: int):
        self.algo = algo
        self.strategy = strategy
        self.num_rounds = num_rounds

    def run(self, client: substra.Client, node_specs: List[NodeSpec]):
        node_ids = [node_spec.node_id for node_spec in node_specs]

        permissions = substra.sdk.schemas.Permissions(public=False, authorized_ids=node_ids)

        algo_key = add_algo(client, self.algo, permissions)

        agg_key = None
        for round in range(self.num_rounds):
            agg_key = self.strategy.perform_round(
                client=client, algo_key=algo_key, agg_key=agg_key, node_specs=node_specs
            )
