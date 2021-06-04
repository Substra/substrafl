import datetime
import substra

from pathlib import Path
from loguru import logger

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


def test_end_2_end():
    org1_client = substra.Client(debug=True)
    org2_client = substra.Client(debug=True)

    org1_dataset_key, org1_data_sample_key = register_dataset(org1_client, ASSETS_DIR)
    org2_dataset_key, org2_data_sample_key = register_dataset(org2_client, ASSETS_DIR)

    from algo.algo import MyAlgo
    from connectlib.strategies import FedAVG
    from connectlib.nodes import TrainDataNode, AggregationNode
    from connectlib.nodes.register import register_algo

    train_data_nodes = [
        TrainDataNode("0", org1_dataset_key, [org1_data_sample_key], "fake_objective"),
        TrainDataNode("1", org1_dataset_key, [org1_data_sample_key], "fake_objective"),
    ]

    aggregation_node = AggregationNode("0")

    my_algo = MyAlgo()
    strategy = FedAVG(1, 1)

    algo_ptr = register_algo(org1_client, my_algo, permisions=DEFAULT_PERMISSIONS)

    strategy.perform_round(
        algo=algo_ptr,
        train_data_nodes=train_data_nodes,
        aggregation_node=aggregation_node,
        local_states=None,
        shared_state=None,
    )

    permissions = substra.sdk.schemas.Permissions(public=False, authorized_ids=["0", "1"])

    composite_traintuples = []
    for node in train_data_nodes:
        node.set_permissions(permissions)
        composite_traintuples += node.tuples

    aggregation_node.register_operations(org1_client, permissions)

    print(composite_traintuples, aggregation_node.tuples)

    compute_plan = org1_client.add_compute_plan(
        {
            "composite_traintuples": composite_traintuples,
            "aggregatetuples": aggregation_node.tuples,
            "tag": str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
        }
    )
