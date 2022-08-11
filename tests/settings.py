"""Global settings for all tests environment."""
from pathlib import Path
from typing import List

import substra
import yaml
from pydantic import BaseModel
from pydantic import root_validator
from pydantic import validator

CURRENT_DIR = Path(__file__).parent

DEFAULT_LOCAL_NETWORK_CONFIGURATION_FILE = CURRENT_DIR / "substra_conf" / "local.yaml"
DEFAULT_REMOTE_NETWORK_CONFIGURATION_FILE = CURRENT_DIR / "substra_conf" / "remote.yaml"
CI_REMOTE_NETWORK_CONFIGURATION_FILE = CURRENT_DIR / "substra_conf" / "ci.yaml"

MIN_ORGANIZATIONS = 2


class OrganizationCfg(BaseModel):
    """Information needed to configure a Substra client to interact with a organization.

    Args:
        url (str): URL of the Substra organization.
        username (str): A user define username to login to the Substra platform. This username will
            be the one used to access Substra frontend.
        password (str): The password to login to the organization.
    """

    url: str
    username: str
    password: str


class LocalConfiguration(BaseModel):
    """Configuration for local tests (debug mode). As in local mode, client must be duplicated
    from one another, the only information is the number of organizations to create.

    Args:
        n_local_organizations (int): the number of organization you want to run the tests on.
    """

    n_local_organizations: int

    @validator("n_local_organizations")
    def minimal_number_of_organizations(cls, v):  # noqa: N805
        assert (
            v >= MIN_ORGANIZATIONS
        ), f"Not enough organizations specified in configuration. Found {v}, at least {MIN_ORGANIZATIONS} are required."
        return v


class RemoteConfiguration(BaseModel):
    """Configuration for the remote tests i.e. a list of organizations configuration.

    Args:
        organizations (List[OrganizationCfg]): A list of Substra organizations configuration.
    """

    organizations: List[OrganizationCfg]

    @validator("organizations")
    def minimal_number_of_organizations(cls, v):  # noqa: N805
        assert len(v) >= MIN_ORGANIZATIONS, (
            "Not enough organizations defined in your configuration. "
            f"Found {len(v)}, at least {MIN_ORGANIZATIONS} is/are required."
        )
        return v


class Network(BaseModel):
    """:term:`Substra` network elements required for testing.
    `msp_ids` and `clients` must be passed in the same order.

    Args:
        msp_ids (List[str]): Ids for each organization of your network. These will be
            used as input permissions during the test.
        clients (List[substra.Client]): Substra clients managing the interaction with the :term:`Substra` platform.
        n_organizations (int): The number of organizations in the network.
    """

    msp_ids: List[str]
    clients: List[substra.sdk.client.Client]
    # Arbitrary type is used because substra Client is not pydantic compatible for now

    class Config:
        arbitrary_types_allowed = True

    @property
    def n_organizations(self) -> int:
        return len(self.msp_ids)

    @root_validator
    def consistent_clients_msp_ids(cls, values):  # noqa: N805
        """Msp_id is the id of a organization. We need to associate one with each client to allow access to the pushed assets.
        msp_ids and clients must be of the same length"""
        l_msp_ids = len(values.get("msp_ids"))
        l_clients = len(values.get("clients"))
        assert (
            l_msp_ids == l_clients
        ), f"`msp_ids` and `clients` must be of same length. Its are respectively {l_msp_ids} and {l_clients}."

        return values


def local_network():
    """Instantiates a local Substra network from the user defined configuration file.
    As the configuration is static and immutable, it is loaded only once from the disk.

    Returns:
        Network: A local network (debug mode) where all Substra clients are duplicated from one an other.
    """
    cfg = yaml.full_load(DEFAULT_LOCAL_NETWORK_CONFIGURATION_FILE.read_text())

    cfg = LocalConfiguration(n_local_organizations=cfg.get("n_local_organizations"))
    n_organizations = cfg.n_local_organizations

    # In debug mode, clients must be duplicated from one another
    clients = [substra.Client(debug=True)] * n_organizations

    # This shall be used as DEBUG_OWNER parameters during assets creation
    msp_ids = [f"MyOrgMSP{k}" for k in range(n_organizations)]

    return Network(clients=clients, msp_ids=msp_ids)


def remote_network(is_ci: bool = False):
    """Instantiates a remote Substra network from the user defined configuration file.
    As the configuration is static and immutable, it is loaded only once from the disk.

    Args:
        is_ci (bool): Set to True the substra network has to be configured to the substra-test.

    Returns:
        Network: A remote network.
    """
    cfg_file = CI_REMOTE_NETWORK_CONFIGURATION_FILE if is_ci else DEFAULT_REMOTE_NETWORK_CONFIGURATION_FILE

    cfg = yaml.full_load(cfg_file.read_text())
    cfg = RemoteConfiguration(
        organizations=[
            OrganizationCfg(
                url=organization_cfg.get("url"),
                username=organization_cfg.get("username"),
                password=organization_cfg.get("password"),
            )
            for organization_cfg in cfg
        ]
    )

    clients = []
    msp_ids = []
    for organization in cfg.organizations:
        client = substra.Client(debug=False, url=organization.url)
        client.login(username=organization.username, password=organization.password)
        clients.append(client)
        msp_ids += [n.id for n in client.list_organization() if n.is_current]

    return Network(clients=clients, msp_ids=msp_ids)
