import json
import os
import tarfile
from pathlib import Path
from typing import List
from typing import Optional

import substra
import yaml
from substra import BackendType
from substra.sdk import DEBUG_OWNER
from substra.sdk.schemas import DataSampleSpec
from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import MetricSpec
from substra.sdk.schemas import Permissions

from connectlib.nodes import AggregationNode
from connectlib.nodes import TestDataNode
from connectlib.nodes import TrainDataNode

CURRENT_DIRECTORY = Path(__file__).parent

ASSETS_DIRECTORY = CURRENT_DIRECTORY / "assets"

PUBLIC_PERMISSIONS = Permissions(public=True, authorized_ids=[])

CONNECT_CONFIG_FOLDER = Path(__file__).parents[1].resolve() / "connect_conf"


def instantiate_clients(mode: str = "subprocess", n_centers: Optional[int] = 2, conf: Optional[dict] = None):
    """Create substra client according to passed args

    Args:
        mode (str, optional): Specify a backend type. Either subprocess, docker or remote. Defaults to "subprocess".
        n_centers (int, optional): Only subprocess and docker. Number of clients to create. Defaults to 2.
        conf (dict, optional): Only remote: connect configuration. Defaults to None.

    Returns:
        _type_: _description_
    """
    if mode == "remote":
        clients = []
        for node in conf:
            client = substra.Client(debug=False, url=node.get("url"))
            client.login(username=node.get("username"), password=node.get("password"))
            clients.append(client)

    else:
        os.environ["DEBUG_SPAWNER"] = mode
        clients = [substra.Client(debug=True)] * n_centers

    return clients


def get_clients(mode: str = "subprocess", credentials: os.PathLike = "remote.yaml", n_centers: int = 2):
    # Load Configuration
    conf = yaml.full_load((CONNECT_CONFIG_FOLDER / credentials).read_text())
    clients = instantiate_clients(conf=conf, mode=mode, n_centers=n_centers)
    return clients


def load_assets_keys(assets_keys_path, mode):
    if mode == "remote" and (CONNECT_CONFIG_FOLDER / assets_keys_path).exists():
        keys = json.loads((CONNECT_CONFIG_FOLDER / assets_keys_path).read_text())
    else:
        keys = {}
    return keys


def save_assets_keys(assets_keys_path, assets_keys):
    (CONNECT_CONFIG_FOLDER / assets_keys_path).write_text(json.dumps(assets_keys, indent=4, sort_keys=True))


def get_train_data_nodes(
    clients: List[substra.Client], train_folders: List[Path], assets_keys: dict
) -> List[TrainDataNode]:
    """Generate a connectlib train data nodes for each client.
    Each client will be associated to one node where the training data are the one in his index wise
    associates train_folder.

    Args:
        clients (List[substra.Client]): List of substra clients.
        train_folders (List[Path]): List of associated train data folders.

    Returns:
        List[TrainDataNode]: Registered train data nodes for connectlib.
    """

    dataset = DatasetSpec(
        name="CameLyon",
        type="None",
        data_opener=ASSETS_DIRECTORY / "opener.py",
        description=ASSETS_DIRECTORY / "description.md",
        permissions=PUBLIC_PERMISSIONS,
        logs_permission=PUBLIC_PERMISSIONS,
    )

    train_data_nodes = []

    for k, train_folder in enumerate(train_folders):
        client = clients[k]
        nodes = client.list_node()
        if nodes:
            msp_id = [c.id for c in nodes if c.is_current][0]
        else:
            msp_id = str(k)

        assets_keys.setdefault(msp_id, {})

        dataset.metadata = {DEBUG_OWNER: msp_id}

        dataset_key = assets_keys.get(msp_id).get("dataset_key") or client.add_dataset(dataset)

        data_sample = DataSampleSpec(
            data_manager_keys=[dataset_key],
            test_only=False,
            path=train_folder
            if client.backend_mode != BackendType.DEPLOYED
            else Path("/var/substra/servermedias/train"),
        )

        data_sample_key = assets_keys.get(msp_id).get("train_data_sample_key") or client.add_data_sample(
            data_sample,
            local=client.backend_mode != BackendType.DEPLOYED,
        )

        assets_keys[msp_id].update(
            {
                "train_data_sample_key": data_sample_key,
                "dataset_key": dataset_key,
            }
        )

        train_data_nodes.append(
            TrainDataNode(
                node_id=msp_id,
                data_manager_key=dataset_key,
                data_sample_keys=[data_sample_key],
            )
        )

    return train_data_nodes


def register_metric(client: substra.Client) -> str:
    """Register a metric default metric to connect.

    Args:
        client (substra.Client): Substra client to register the metric.

    Returns:
        str: Substra returned key of the registered metric.
    """

    metric_archive_path = ASSETS_DIRECTORY / "metric.tar.gz"

    with tarfile.open(metric_archive_path, "w:gz") as tar:
        tar.add(ASSETS_DIRECTORY / "Dockerfile", arcname="Dockerfile")
        tar.add(ASSETS_DIRECTORY / "metric.py", arcname="metrics.py")

    metric_spec = MetricSpec(
        name="ROC",
        description=ASSETS_DIRECTORY / "description.md",
        file=metric_archive_path,
        permissions=PUBLIC_PERMISSIONS,
    )

    metric_key = client.add_metric(metric_spec)

    return metric_key


def get_test_data_nodes(clients: List[substra.Client], test_folders: List[Path], assets_keys: dict) -> TestDataNode:
    """Generate a test data node for the data within the passed folder with the client.
    The associated metric only returns the float(y_pred) where y_pred is the results of the
    predict method of the used algorithm.

    Args:
        client (substra.Client): Substra client to register the asset with.
        test_folder (Path): Folder where the test data are stored.

    Returns:
        TestDataNode: Connectlib test data.
    """
    # only one metric is needed as permissions are public
    metric_key = assets_keys.get("metric") or register_metric(clients[0])
    assets_keys.update(
        {
            "metric_key": metric_key,
        }
    )

    dataset = DatasetSpec(
        name="CameLyon",
        type="None",
        data_opener=ASSETS_DIRECTORY / "opener.py",
        description=ASSETS_DIRECTORY / "description.md",
        permissions=PUBLIC_PERMISSIONS,
        logs_permission=PUBLIC_PERMISSIONS,
    )

    test_data_nodes = []

    for k, test_folder in enumerate(test_folders):

        client = clients[k]
        nodes = client.list_node()
        if nodes:
            msp_id = [c.id for c in nodes if c.is_current][0]
        else:
            msp_id = str(k)

        dataset_key = assets_keys.get(msp_id).get("dataset_key") or client.add_dataset(dataset)

        data_sample = DataSampleSpec(
            data_manager_keys=[dataset_key],
            test_only=True,
            path=test_folder if client.backend_mode != BackendType.DEPLOYED else Path("/var/substra/servermedias/test"),
        )

        data_sample_key = assets_keys.get(msp_id).get("test_data_sample_key") or client.add_data_sample(
            data_sample,
            local=client.backend_mode != BackendType.DEPLOYED,
        )

        test_data_node = TestDataNode(
            node_id=msp_id,  # The one we want as everything is public
            data_manager_key=dataset_key,
            test_data_sample_keys=[data_sample_key],
            metric_keys=[metric_key],
        )

        assets_keys[msp_id].update(
            {
                "test_data_sample_key": data_sample_key,
                "dataset_key": dataset_key,
            }
        )

        test_data_nodes.append(test_data_node)

    return test_data_nodes


def get_aggregation_node(client) -> AggregationNode:
    """Returns a connectlib aggregation node.

    Returns:
        AggregationNode: Connectlib aggregation node.
    """
    nodes = client.list_node()
    if nodes:
        msp_id = [c.id for c in nodes if c.is_current][0]
    else:
        msp_id = "0"
    return AggregationNode(msp_id)
