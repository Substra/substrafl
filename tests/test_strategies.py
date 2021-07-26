import json
import numpy as np
from pathlib import Path
from loguru import logger
import os
import zipfile

from connectlib.algorithms import Algo
from connectlib.nodes import TrainDataNode, AggregationNode, TestDataNode
from connectlib.orchestrator import Orchestrator
from connectlib.remote import remote_data
from connectlib.strategies import FedAVG

import substra

ASSETS_DIR = Path(__file__).parent / "end_to_end" / "test_assets"
DEFAULT_PERMISSIONS = substra.sdk.schemas.Permissions(public=True, authorized_ids=list())
LOCAL_WORKER_PATH = Path.cwd() / "local-worker"


def register_dataset(client: substra.Client, asset_dir: Path, partner_name: str):
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
            # this is important in the debug mode to separate the partners
            metadata={substra.DEBUG_OWNER: partner_name},
        )
    )

    # Add the data samples : one data sample with the MDF inside
    logger.info("Adding data sample")
    data_sample_key = client.add_data_sample(
        substra.sdk.schemas.DataSampleSpec(
            path=asset_dir / f"datasample_{partner_name}",
            test_only=False,
            data_manager_keys=[dataset_key],
        )
    )

    return dataset_key, data_sample_key


def make_objective(asset_dir: Path):
    # objective_key is a key to the metric your registered
    # objective = asset_factory.create_objective(dataset=org1_client.get_dataset(org1_dataset_key))
    # data['test_data_sample_keys'] = load_data_samples_keys(data_samples)
    zip_objective(asset_dir=asset_dir)

    objective = {
        "name": "avg strategy end-to-end",
        "description": ASSETS_DIR / "opener" / "description.md",
        "metrics_name": "accuracy",
        "metrics": ASSETS_DIR / "objective" / "metrics.zip",
        "permissions": DEFAULT_PERMISSIONS,  # {"public": False, "authorized_ids": []},
    }

    return objective


def zip_objective(asset_dir: Path):
    # Create necessary archive to register the operation on substra
    # the previous metric.zip will be overwritten
    operation_dir = asset_dir / "objective"
    archive_path = operation_dir / "metrics.zip"
    with zipfile.ZipFile(archive_path, "w") as z:
        for filepath in operation_dir.glob("*[!.zip]"):
            print(f"zipped in {filepath}")
            z.write(filepath, arcname=os.path.basename(filepath))


def test_fed_avg():
    # makes sure that federated average strategy leads to the averaging output of the models from both partners.
    # The data for the two partners consists of only 0s or 1s respectively. The train() returns the data.
    # predict() returns the data, score returned by AccuracyMetric (in the objective) is the mean of all the y_pred
    # passed to it. The tests asserts if the score is 0.5
    class MyAlgo(Algo):
        # this class must be within the test, otherwise the Docker will not find it correctly (ie because of the way
        # pytest calls it)
        def delayed_init(self, seed: int, *args, **kwargs):
            pass

        @remote_data
        def train(self, x: np.array, y: np.array, num_updates: int, shared_state):
            return dict(test=x)

        @remote_data
        def predict(self, x: np.array, shared_state):
            return shared_state["test"]

        def load(self, path: Path):
            return self

        def save(self, path: Path):
            assert path.parent.exists()
            with path.open("w") as f:
                f.write("test")

    org_client = substra.Client(debug=True)
    partners = ["0", "1"]

    # generate the data for partner "0" and "1"
    for partner_name in partners:
        # TODO: remove datasample dirs at the end of the tests
        path_data = ASSETS_DIR / f"datasample_{partner_name}"
        if not path_data.is_dir():
            path_data.mkdir()
        # all the data will be either 0s or 1s depending on the partner
        new_data = np.ones([8, 16]) * int(partner_name)
        np.save(path_data / "data.npy", new_data)

    org1_dataset_key, org1_data_sample_key = register_dataset(
        org_client, ASSETS_DIR, partners[0]
    )
    org2_dataset_key, org2_data_sample_key = register_dataset(
        org_client, ASSETS_DIR, partners[1]
    )

    train_data_nodes = [
        TrainDataNode(partners[0], org1_dataset_key, [org1_data_sample_key]),
        TrainDataNode(partners[1], org2_dataset_key, [org2_data_sample_key]),
    ]

    OBJECTIVE = make_objective(org_client, ASSETS_DIR)
    org1_objective_key = org_client.add_objective(
        {
            "name": OBJECTIVE["name"],
            "description": str(OBJECTIVE["description"]),
            "metrics_name": OBJECTIVE["metrics_name"],
            "metrics": str(OBJECTIVE["metrics"]),
            "test_data_sample_keys": [org1_data_sample_key],
            "test_data_manager_key": org1_dataset_key,
            "permissions": OBJECTIVE["permissions"],
        },
    )

    OBJECTIVE = make_objective(org_client, ASSETS_DIR)
    org2_objective_key = org_client.add_objective(
        {
            "name": OBJECTIVE["name"],
            "description": str(OBJECTIVE["description"]),
            "metrics_name": OBJECTIVE["metrics_name"],
            "metrics": str(OBJECTIVE["metrics"]),
            "test_data_sample_keys": [org2_data_sample_key],
            "test_data_manager_key": org2_dataset_key,
            "permissions": OBJECTIVE["permissions"],
        },
    )

    test_data_nodes = [
        TestDataNode(
            partners[0],
            org1_dataset_key,
            [org1_data_sample_key],
            objective_key=org1_objective_key,
        ),
        TestDataNode(
            partners[1],
            org2_dataset_key,
            [org2_data_sample_key],
            objective_key=org2_objective_key,
        ),
    ]

    aggregation_node = AggregationNode(partners[0])
    my_algo0 = MyAlgo()
    strategy = FedAVG(num_updates=2)

    orchestrator = Orchestrator(my_algo0, strategy, num_rounds=1)
    orchestrator.run(
        org_client, train_data_nodes, aggregation_node, test_data_nodes=test_data_nodes
    )

    # read the results from saved performances
    path = LOCAL_WORKER_PATH / "performances"
    dirs = [d for d in path.iterdir() if d.is_dir()]
    newest = max(dirs, key=os.path.getctime)
    score_file = newest / "performance.json"
    score = json.loads(score_file.read_bytes())["all"]

    # assert that the calculated score matches the expected score
    assert score == 0.5
