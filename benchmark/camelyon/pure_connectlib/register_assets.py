import tarfile
from pathlib import Path
from typing import List

import substra
from substra.sdk import DEBUG_OWNER
from substra.sdk.schemas import DataSampleSpec
from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import MetricSpec
from substra.sdk.schemas import Permissions

from connectlib.nodes import AggregationNode
from connectlib.nodes import TestDataNode
from connectlib.nodes import TrainDataNode

current_dir = Path(__file__).parent

assets_directory = current_dir / "assets"


PUBLIC_PERMISSIONS = Permissions(public=True, authorized_ids=[])


def get_train_data_nodes(clients: List[substra.Client], trains_folders: List[Path]) -> List[TrainDataNode]:
    """Generate a connectlib train data nodes for each client.
    Each client will be associated to one node where the training data are the one in his index wise
    associates train_folder.

    Args:
        clients (List[substra.Client]): List of substra clients.
        trains_folders (List[Path]): List of associated train data folders.

    Returns:
        List[TrainDataNode]: Registered train data nodes for connectlib.
    """

    dataset = DatasetSpec(
        name="CameLyon",
        type="None",
        data_opener=assets_directory / "opener.py",
        description=assets_directory / "description.md",
        permissions=PUBLIC_PERMISSIONS,
        logs_permission=PUBLIC_PERMISSIONS,
    )

    train_data_nodes = []

    for k, train_folder in enumerate(trains_folders):
        dataset.metadata = {DEBUG_OWNER: str(k)}
        client = clients[k]

        dataset_key = client.add_dataset(dataset)

        data_sample = DataSampleSpec(
            data_manager_keys=[dataset_key],
            test_only=False,
            path=train_folder,
        )
        data_sample_key = client.add_data_sample(
            data_sample,
            local=True,
        )

        train_data_nodes.append(
            TrainDataNode(
                node_id=str(k),
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

    metric_archive_path = assets_directory / "metric.tar.gz"

    with tarfile.open(metric_archive_path, "w:gz") as tar:
        tar.add(assets_directory / "Dockerfile", arcname="Dockerfile")
        tar.add(assets_directory / "metric.py", arcname="metrics.py")

    metric_spec = MetricSpec(
        name="ROC",
        description=assets_directory / "description.md",
        file=metric_archive_path,
        permissions=PUBLIC_PERMISSIONS,
    )

    metric_key = client.add_metric(metric_spec)

    return metric_key


def get_test_data_node(client: substra.Client, test_folder: Path) -> TestDataNode:
    """Generate a test data node for the data within the passed folder with the client.
    The associated metric only returns the float(y_pred) where y_pred is the results of the
    predict method of the used algorithm.

    Args:
        client (substra.Client): Substra client to register the asset with.
        test_folder (Path): Folder where the test data are stored.

    Returns:
        TestDataNode: Connectlib test data.
    """
    metric_key = register_metric(client)

    dataset = DatasetSpec(
        name="CameLyon",
        type="None",
        data_opener=assets_directory / "opener.py",
        description=assets_directory / "description.md",
        permissions=PUBLIC_PERMISSIONS,
        logs_permission=PUBLIC_PERMISSIONS,
    )
    dataset_key = client.add_dataset(dataset)

    data_sample = DataSampleSpec(
        data_manager_keys=[dataset_key],
        test_only=True,
        path=test_folder,
    )

    data_sample_key = client.add_data_sample(
        data_sample,
        local=True,
    )

    test_data_node = TestDataNode(
        node_id="0",  # The one we want as everything is public
        data_manager_key=dataset_key,
        test_data_sample_keys=[data_sample_key],
        metric_keys=[metric_key],
    )

    return test_data_node


def get_aggregation_node() -> AggregationNode:
    """Returns a connectlib aggregation node.

    Returns:
        AggregationNode: Connectlib aggregation node.
    """

    return AggregationNode("0")
