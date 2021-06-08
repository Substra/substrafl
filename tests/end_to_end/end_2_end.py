import substra

import numpy as np

from pathlib import Path
from loguru import logger

from connectlib.algorithms import Algo
from connectlib.remote import remote_data
from connectlib.orchestrator import Orchestrator
from connectlib.strategies import FedAVG
from connectlib.nodes import TrainDataNode, AggregationNode

ASSETS_DIR = Path(__file__).parent / "test_assets"
DEFAULT_PERMISSIONS = substra.sdk.schemas.Permissions(public=True, authorized_ids=list())


def register_dataset(client: substra.Client, asset_dir: Path):
    # Add the dataset
    # This is the data opener that for nows returns directly a mdf
    logger.info("Adding dataset")
    dataset_key = client.add_dataset(
        substra.sdk.schemas.DatasetSpec(
            name="opener - MDF",
            data_opener=asset_dir / "opener" / "opener.py",
            type="opener - MDF",
            description=asset_dir / "opener" / "description.md",
            permissions=DEFAULT_PERMISSIONS,
        )
    )

    # Add the data samples : one data sample with the MDF inside
    logger.info("Adding data sample")
    data_sample_key = client.add_data_sample(
        substra.sdk.schemas.DataSampleSpec(
            path=asset_dir / "datasample",
            test_only=False,
            data_manager_keys=[dataset_key],
        )
    )

    return dataset_key, data_sample_key


class MyAlgo(Algo):
    def delayed_init(self, seed: int, *args, **kwargs):
        self._shared_state = {"test": np.random.randn(8, 16)}

    @remote_data
    def train(self, x: np.array, y: np.array, num_updates: int, shared_state):
        return self._shared_state

    @remote_data
    def predict(self, x: np.array, shared_state):
        return np.random.randint(0, 2, size=(len(x), 1))

    def load(self, path: Path):
        pass

    def save(self, path: Path):
        assert path.parent.exists()
        with path.open("w") as f:
            f.write("test")


org1_client = substra.Client(debug=True)
org2_client = substra.Client(debug=True)

org1_dataset_key, org1_data_sample_key = register_dataset(org1_client, ASSETS_DIR)
org2_dataset_key, org2_data_sample_key = register_dataset(org2_client, ASSETS_DIR)
###########################

train_data_nodes = [
    TrainDataNode("0", org1_dataset_key, [org1_data_sample_key]),
    TrainDataNode("1", org2_dataset_key, [org2_data_sample_key]),
]

aggregation_node = AggregationNode("0")

my_algo = MyAlgo()
strategy = FedAVG(num_updates=1)

orchestrator = Orchestrator(my_algo, strategy, num_rounds=1)
orchestrator.run(org1_client, train_data_nodes, aggregation_node)
