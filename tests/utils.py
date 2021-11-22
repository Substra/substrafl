import time

from substra.sdk.models import ComputePlanStatus, Status

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
        assert False, "Future not supported"
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
