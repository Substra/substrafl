import json
import os
from copy import deepcopy
from pathlib import Path
from typing import List
from typing import Optional

import substra
import yaml
from substra.sdk.schemas import DataSampleSpec
from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import Permissions
from tqdm import tqdm

from substrafl.nodes import AggregationNode
from substrafl.nodes import TestDataNode
from substrafl.nodes import TrainDataNode
from substrafl.schemas import TaskType

CURRENT_DIRECTORY = Path(__file__).parent

ASSETS_DIRECTORY = CURRENT_DIRECTORY / "assets"

PUBLIC_PERMISSIONS = Permissions(public=True, authorized_ids=[])

SUBSTRA_CONFIG_FOLDER = Path(__file__).parents[1].resolve() / "substra_conf"

DEFAULT_DATASET = DatasetSpec(
    name="Camelyon",
    type="None",
    data_opener=ASSETS_DIRECTORY / "opener.py",
    description=ASSETS_DIRECTORY / "description.md",
    permissions=PUBLIC_PERMISSIONS,
    logs_permission=PUBLIC_PERMISSIONS,
)


def instantiate_clients(
    mode: substra.BackendType = substra.BackendType.LOCAL_SUBPROCESS,
    n_centers: Optional[int] = 2,
    conf: Optional[List[dict]] = None,
) -> List[substra.Client]:
    """Create substra client according to passed args

    Args:
        mode (substra.BackendType): Specify a backend type. Either subprocess, docker or remote. Defaults to
        "subprocess".
        n_centers (int, optional): Only subprocess and docker. Number of clients to create. Defaults to 2.
        conf (dict, optional): Only remote: Substra configuration. Defaults to None.

    Returns:
        _type_: _description_
    """
    if mode == substra.BackendType.REMOTE:
        clients = []
        for organization in conf:
            client = substra.Client(
                backend_type=mode,
                url=organization.get("url"),
                username=organization.get("username"),
                password=organization.get("password"),
                insecure=organization.get("insecure", False),
            )
            clients.append(client)
    else:
        clients = [substra.Client(backend_type=mode) for _ in range(n_centers)]

    return clients


def get_clients(
    mode: substra.BackendType, credentials: os.PathLike = "remote.yaml", n_centers: int = 2
) -> List[substra.Client]:
    # Load Configuration
    conf = yaml.full_load((SUBSTRA_CONFIG_FOLDER / credentials).read_text())
    clients = instantiate_clients(conf=conf, mode=mode, n_centers=n_centers)
    return clients


def load_asset_keys(asset_keys_path, mode: substra.BackendType):
    if mode == substra.BackendType.REMOTE and (SUBSTRA_CONFIG_FOLDER / asset_keys_path).exists():
        keys = json.loads((SUBSTRA_CONFIG_FOLDER / asset_keys_path).read_text())
    else:
        keys = {}
    return keys


def save_asset_keys(asset_keys_path, asset_keys):
    (SUBSTRA_CONFIG_FOLDER / asset_keys_path).write_text(json.dumps(asset_keys, indent=4, sort_keys=True))


def add_duplicated_dataset(
    client: substra.Client,
    nb_data_sample: int,
    data_sample_folder: os.PathLike,
    asset_keys: dict,
    msp_id: str,
    kind: str = TaskType.TRAIN.value,
) -> dict:
    """Update asset_keys.msp_id.{kind}_data_sample_keys so there is exactly `nb_data_sample` keys
    by adding the `data_sample_folder` as a data sample with the provided `client` as many times as it's need
    or by selecting the first `nb_data_sample` of the given data samples keys.

    Args:
        client (substra.Client): Substra client to use to add the data samples
        nb_data_sample (int): Number of data sample keys to be returned in the asset_keys dict.
        data_sample_folder (os.PathLike): Folder where the data sample data are stored.
        asset_keys (dict): already registered assets within Substra. It needs to be formatted as followed:
            .. code-block:: json

                {
                    <msp_id>: {
                        "dataset_key": "b8d754f0-40a5-4976-ae16-8dd4eca35ffc",
                        "data_sample_keys": ["1238452c-a1dd-47ef-84a8-410c0841693a"],
                        "train_data_sample_keys": ["38071944-c974-4b3b-a671-aa4835a0ae62"]
                    },
                    <msp_id>: {
                        "dataset_key": "fa8e9bf7-5084-4b59-b089-a459495a08be",
                        "data_sample_keys": ["73715d69-9447-4270-9d3f-d0b17bb88a87"],
                        "train_data_sample_keys": ["766d2029-f90b-440e-8b39-2389ab04041d"]
                    },
                    ...
                }
        msp_id (str): asset_keys key where to find the registered assets for the given client
        kind (str, optional): Kind of data sample to add, either train or test.  Defaults to
        TaskType.TRAIN.value.

    Returns:
        dict: The updated asset_keys.
    """

    asset_keys.setdefault(msp_id, {})

    dataset = deepcopy(DEFAULT_DATASET)

    dataset_key = asset_keys.get(msp_id).get("dataset_key") or client.add_dataset(dataset)

    data_sample = DataSampleSpec(
        data_manager_keys=[dataset_key],
        path=data_sample_folder,
    )

    data_sample_keys = asset_keys.get(msp_id).get(f"{kind}_data_sample_keys") or []
    nb_data_sample_to_add = nb_data_sample - len(data_sample_keys)

    if nb_data_sample_to_add < 0:
        data_sample_keys = data_sample_keys[:nb_data_sample]

    for _ in tqdm(range(nb_data_sample_to_add), desc=f"Client {msp_id}: adding {kind} data samples"):
        data_sample_key = client.add_data_sample(data_sample, local=True)
        data_sample_keys.append(data_sample_key)

    asset_keys[msp_id].update(
        {
            f"{kind}_data_sample_keys": data_sample_keys,
            "dataset_key": dataset_key,
        }
    )

    return asset_keys


def get_train_data_nodes(
    clients: List[substra.Client], train_folder: Path, asset_keys: dict, nb_data_sample: int
) -> List[TrainDataNode]:
    """Generate a substrafl train data node for each client.
    Each client will be associated to one node where the training data are the one in his index wise
    associates train_folder.

    Args:
        clients (List[substra.Client]): List of substra clients.
        train_folder (Path): Unique train data sample to be replicated and used.
        asset_keys (dict): Already registered asset to be reused. If an asset is defined in this dict,
            it will be reused.
        nb_data_sample (int): The number of time the train data folder will be used as a datasample.
            If train data sample keys are already present in the assets keys, the first nb_data_sample will
            be reused and new ones will be added if needed so the number of datasamples used always is nb_data_sample

    Returns:
        List[TrainDataNode]: Registered train data nodes for substrafl.
    """

    train_data_nodes = []

    for client in clients:
        msp_id = client.organization_info().organization_id
        asset_keys = add_duplicated_dataset(
            client=client,
            nb_data_sample=nb_data_sample,
            data_sample_folder=train_folder,
            asset_keys=asset_keys,
            msp_id=msp_id,
            kind=TaskType.TRAIN.value,
        )

        train_data_nodes.append(
            TrainDataNode(
                organization_id=msp_id,
                data_manager_key=asset_keys.get(msp_id)["dataset_key"],
                data_sample_keys=asset_keys.get(msp_id)["train_data_sample_keys"],
            )
        )

    return train_data_nodes


def get_test_data_nodes(
    clients: List[substra.Client], test_folder: Path, asset_keys: dict, nb_data_sample
) -> List[TestDataNode]:
    """Generate a test data node for the data within the passed folder with the client.
    The associated metric only returns the float(y_pred) where y_pred is the results of the
    predict method of the used function.

    Args:
        client (substra.Client): Substra client to register the asset with.
        test_folder (Path): Folder where the test data are stored.
        nb_data_sample (int): The number of time the test data folder will be added as a datasample.
            If a test data sample keys is present in the assets keys, new datasamples will be added so the length of the
            data sample keys list matches the nb_data_sample value.

    Returns:
        TestDataNode: Substrafl test data.
    """

    test_data_nodes = []

    for client in clients:
        msp_id = client.organization_info().organization_id
        asset_keys = add_duplicated_dataset(
            client=client,
            data_sample_folder=test_folder,
            nb_data_sample=nb_data_sample,
            kind="test",
            msp_id=msp_id,
            asset_keys=asset_keys,
        )

        test_data_nodes.append(
            TestDataNode(
                organization_id=msp_id,
                data_manager_key=asset_keys.get(msp_id)["dataset_key"],
                data_sample_keys=asset_keys.get(msp_id)["test_data_sample_keys"],
            )
        )

    return test_data_nodes


def get_aggregation_node(client: substra.Client) -> AggregationNode:
    """Returns a substrafl aggregation node.

    Returns:
        AggregationNode: Substrafl aggregation node.
    """
    return AggregationNode(organization_id=client.organization_info().organization_id)
