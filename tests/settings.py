"""Global settings for all tests environment."""
from pathlib import Path
from typing import List

import substra
import yaml
from pydantic import BaseModel, root_validator, validator

CURRENT_DIR = Path(__file__).parent

DEFAULT_LOCAL_NETWORK_CONFIGURATION_FILE = CURRENT_DIR / "connect_conf" / "local.yaml"
DEFAULT_REMOTE_NETWORK_CONFIGURATION_FILE = CURRENT_DIR / "connect_conf" / "remote.yaml"
NIGHTLY_REMOTE_NETWORK_CONFIGURATION_FILE = (
    CURRENT_DIR / "connect_conf" / "nightly.yaml"
)

MIN_NODES = 2


class NodeCfg(BaseModel):
    """Information needed to configure a Connect client to interact with a node.

    Args:
        url (str): URL of the Connect node.
        msp_id (str): Node id. Passed as a permission when assets are added to Connect,
            it ensures access to the associated node.
        username (str): A user define username to login to the Connect platform. This username will
            be the one used to access Connect frontend.
        password (str): The password to login to the node.
    """

    url: str
    msp_id: str
    username: str
    password: str


class LocalConfiguration(BaseModel):
    """Configuration for local tests (debug mode). As in local mode, client must be duplicated
    from one another, the only information is the number of nodes to create.

    Args:
        n_local_nodes (int): the number of node you want to run the tests on.
    """

    n_local_nodes: int

    @validator("n_local_nodes")
    def minimal_number_of_nodes(cls, v):
        assert v >= MIN_NODES, "Not enough nodes specified in configuration). "
        f"Found {v}, at least {MIN_NODES} are required."
        return v


class RemoteConfiguration(BaseModel):
    """Configuration for the remote tests i.e. a list of nodes configuration.

    Args:
        nodes (List[NodeCfg]): A list of Connect nodes configuration.
    """

    nodes: List[NodeCfg]

    @validator("nodes")
    def minimal_number_of_nodes(cls, v):
        assert len(v) >= MIN_NODES, "Not enough nodes defined in your configuration. "
        f"Found {len(v)}, at least {MIN_NODES} is/are required."
        return v


class Network(BaseModel):
    """:term:`Connect` network elements required for testing.
    `msp_ids` and `clients` must be passed in the same order.

    Args:
        msp_ids (List[str]): Ids for each node of your network. These will be
            used as input permissions during the test.
        clients (List[substra.Client]): Substra clients managing the interaction with the :term:`Connect` platform.
        n_nodes (int): The number of nodes in the network.
    """

    msp_ids: List[str]
    clients: List[substra.sdk.client.Client]
    # Arbitrary type is used because substra Client is not pydantic compatible for now

    class Config:
        arbitrary_types_allowed = True

    @property
    def n_nodes(self) -> int:
        return len(self.msp_ids)

    @root_validator
    def consistent_clients_msp_ids(cls, values):
        """Msp_id is the id of a node. We need to associate one with each client to allow access to the pushed assets.
        msp_ids and clients must be of the same length"""
        l_msp_ids = len(values.get("msp_ids"))
        l_clients = len(values.get("clients"))
        assert (
            l_msp_ids == l_clients
        ), f"`msp_ids` and `clients` must be of same length. Its are respectively {l_msp_ids} and {l_clients}."

        return values


def local_network():
    """Instantiates a local connect network from the user defined configuration file.
    As the configuration is static and immutable, it is loaded only once from the disk.

    Returns:
        Network: A local network (debug mode) where all connect clients are duplicated from one an other.
    """
    cfg = yaml.full_load(DEFAULT_LOCAL_NETWORK_CONFIGURATION_FILE.read_text())

    cfg = LocalConfiguration(n_local_nodes=cfg.get("n_local_nodes"))
    n_nodes = cfg.n_local_nodes

    # In debug mode, clients must be duplicated from one another
    clients = [substra.Client(debug=True)] * n_nodes

    # This shall be used as DEBUG_OWNER parameters during assets creation
    msp_ids = [f"MyOrgMSP{k}" for k in range(n_nodes)]

    return Network(clients=clients, msp_ids=msp_ids)


def remote_network(is_nightly: bool = False):
    """Instantiates a remote connect network from the user defined configuration file.
    As the configuration is static and immutable, it is loaded only once from the disk.

    Args:
        is_nightly (bool): Set to True the substra network has to be configured to the connect-test.

    Returns:
        Network: A remote network.
    """
    cfg_file = (
        NIGHTLY_REMOTE_NETWORK_CONFIGURATION_FILE
        if is_nightly
        else DEFAULT_REMOTE_NETWORK_CONFIGURATION_FILE
    )

    cfg = yaml.full_load(cfg_file.read_text())
    cfg = RemoteConfiguration(
        nodes=[
            NodeCfg(
                url=node_cfg.get("url"),
                msp_id=node_cfg.get("msp_id"),
                username=node_cfg.get("username"),
                password=node_cfg.get("password"),
            )
            for node_cfg in cfg
        ]
    )

    clients = []
    msp_ids = []
    for node in cfg.nodes:
        client = substra.Client(debug=False, url=node.url)
        client.login(username=node.username, password=node.password)
        clients.append(client)
        msp_ids.append(node.msp_id)

    return Network(clients=clients, msp_ids=msp_ids)
