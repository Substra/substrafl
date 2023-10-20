"""Global settings for all tests environment."""
import functools
from pathlib import Path
from typing import List
from typing import Optional

import substra
import yaml
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator

CURRENT_DIR = Path(__file__).parent

DEFAULT_REMOTE_NETWORK_CONFIGURATION_FILE = CURRENT_DIR / "substra_conf" / "remote.yaml"
CI_REMOTE_NETWORK_CONFIGURATION_FILE = CURRENT_DIR / "substra_conf" / "ci.yaml"

MIN_ORGANIZATIONS = 2

FUTURE_TIMEOUT = 3600
FUTURE_POLLING_PERIOD = 1


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


class Configuration(BaseModel):
    """Configuration for the remote tests i.e. a list of organizations configuration.

    Args:
        organizations (List[OrganizationCfg]): A list of Substra organizations configuration.
    """

    organizations: List[OrganizationCfg]

    @field_validator("organizations")
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

    clients: List[substra.sdk.client.Client]
    msp_ids: Optional[List[str]] = None

    # Arbitrary type is used because substra Client is not pydantic compatible
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def n_organizations(self) -> int:
        return len(self.clients)

    def __init__(self, **data):
        super().__init__(**data)
        self.msp_ids = [client.organization_info().organization_id for client in data["clients"]]


def network(backend_type: substra.BackendType, is_ci: bool = False):
    cfg_file = DEFAULT_REMOTE_NETWORK_CONFIGURATION_FILE
    if is_ci:
        cfg_file = CI_REMOTE_NETWORK_CONFIGURATION_FILE

    cfg = yaml.full_load(cfg_file.read_text())
    cfg = Configuration(
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
    for organization in cfg.organizations:
        if backend_type == substra.BackendType.REMOTE:
            client = substra.Client(backend_type=substra.BackendType.REMOTE, url=organization.url)
        else:
            client = substra.Client(backend_type=backend_type)
        client.login(username=organization.username, password=organization.password)
        client._wait = functools.partial(client._wait, timeout=FUTURE_TIMEOUT, polling_period=FUTURE_POLLING_PERIOD)
        clients.append(client)

    return Network(clients=clients)
