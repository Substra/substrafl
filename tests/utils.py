import time

from substra.sdk.models import ComputePlanStatus
from substra.sdk.models import ModelType
from substra.sdk.models import Status

FUTURE_TIMEOUT = 3600
FUTURE_POLLING_PERIOD = 1


_get_methods = {
    "Traintuple": "get_traintuple",
    "Testtuple": "get_testtuple",
    "Aggregatetuple": "get_aggregatetuple",
    "CompositeTraintuple": "get_composite_traintuple",
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
    # Retrieve composite train tuple key
    train_tasks = network.clients[0].list_composite_traintuple(
        filters=[
            f"composite_traintuple:compute_plan_key:{compute_plan.key}",
            f"composite_traintuple:rank:{rank}",
        ]
    )
    local_models = list()
    for task in train_tasks:
        for model in task.composite.models:
            client = None
            if task.worker == network.msp_ids[0]:
                client = network.clients[0]
            elif task.worker == network.msp_ids[1]:
                client = network.clients[1]
            client.download_model(model.key, session_dir)
            model_path = session_dir / f"model_{model.key}"
            if model.category == ModelType.head:
                model = my_algo.load(model_path)
                # Move the torch model to CPU
                model.model.to("cpu")
                local_models.append(model)
    return local_models
