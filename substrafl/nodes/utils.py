from typing import Any
from typing import List

import substra
import substratools


def preload_data(
    client: substra.Client,
    data_manager_key: str,
    data_sample_keys: List[str],
) -> Any:
    """Get the opener from the client using its key, and apply
    the method `get_data` to the datasamples in order to retrieve them.

    Args:
        client(substra.Client): A substra client to interact with the Substra platform, in order to retrieve the
            registered data.
        data_manager_key(str): key of the registered opener.
        data_sample_keys(List[str]): keys of the registered datasamples paths.

    Returns:
        Any: output of the opener's `get_data` method applied on the corresponding datasamples paths.
    """
    dataset_info = client.get_dataset(data_manager_key)

    opener_interface = substratools.utils.load_interface_from_module(
        "opener",
        interface_class=substratools.Opener,
        interface_signature=None,
        path=dataset_info.opener.storage_address,
    )

    data_sample_paths = [client.get_data_sample(dsk).path for dsk in data_sample_keys]

    return opener_interface.get_data(data_sample_paths)
