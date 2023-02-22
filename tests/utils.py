import pickle
import time

from substra.sdk.models import ComputePlanStatus
from substra.sdk.models import Status

from substrafl.nodes.node import OutputIdentifiers

FUTURE_TIMEOUT = 3600
FUTURE_POLLING_PERIOD = 1


_get_methods = {
    "Task": "get_task",
    "ComputePlan": "get_compute_plan",
}


class TError(Exception):
    """Substra Test Error."""

    pass


class FutureTimeoutError(TError):
    """Future execution timed out."""

    pass


class FutureFailureError(TError):
    """Future execution failed."""

    pass


def wait(client, asset, timeout=FUTURE_TIMEOUT, raises=True):
    try:
        m = _get_methods[asset.__class__.__name__]
    except KeyError:
        raise KeyError("Future not supported")
    getter = getattr(client, m)

    key = asset.key

    tstart = time.time()
    while asset.status not in [
        Status.done.value,
        Status.failed.value,
        Status.canceled.value,
        ComputePlanStatus.done.value,
        ComputePlanStatus.failed.value,
        ComputePlanStatus.canceled.value,
    ]:
        if time.time() - tstart > timeout:
            raise FutureTimeoutError(f"Future timeout on {asset}")

        time.sleep(FUTURE_POLLING_PERIOD)
        asset = getter(key)

    if raises and asset.status in (Status.failed.value, ComputePlanStatus.failed.value):
        raise FutureFailureError(f"Future execution failed on {asset}")

    if raises and asset.status in (
        Status.canceled.value,
        ComputePlanStatus.canceled.value,
    ):
        raise FutureFailureError(f"Future execution canceled on {asset}")

    return asset


def download_composite_models_by_rank(network, session_dir, my_algo, compute_plan, rank: int):
    # Retrieve local train task key
    train_tasks = network.clients[0].list_task(
        filters={
            "compute_plan_key": [compute_plan.key],
            "rank": [rank],
        }
    )
    local_models = list()
    for task in train_tasks:
        client = None
        if task.worker == network.msp_ids[0]:
            client = network.clients[0]
        elif task.worker == network.msp_ids[1]:
            client = network.clients[1]

        for identifier, output in task.outputs.items():
            if identifier != OutputIdentifiers.local:
                continue
            model_path = client.download_model(output.value.key, session_dir)
            model = my_algo.load(model_path)
            # Move the torch model to CPU
            model.model.to("cpu")
            local_models.append(model)
    return local_models


def download_aggregate_model_by_rank(network, session_dir, compute_plan, rank: int):
    aggregate_tasks = network.clients[0].list_task(filters={"compute_plan_key": [compute_plan.key], "rank": [rank]})
    aggregate_tasks = [t for t in aggregate_tasks if t.tag == "aggregate"]
    assert len(aggregate_tasks) == 1
    model_path = network.clients[0].download_model_from_task(
        aggregate_tasks[0].key, identifier=OutputIdentifiers.model, folder=session_dir
    )
    aggregate_model = pickle.loads(model_path.read_bytes())

    return aggregate_model
