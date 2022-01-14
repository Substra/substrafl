from typing import List

# TODO: fix black/isort conflicts
from connectlib.remote.methods import AggregateOperation  # isort:skip
from connectlib.remote.methods import DataOperation, remote, remote_data  # isort:skip


class RemoteClass:
    # test Class for testing if the remote_data and remote_data_func work correctly
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @remote_data
    def remote_data_func(self, x: int, y: int, shared_state: int, extra_arg: int = 0) -> int:
        return x + y + shared_state + extra_arg

    @remote
    def remote_func(self, shared_states: List[int], extra_arg: int = 0) -> int:
        return sum(shared_states) + extra_arg


# TODO: test fake_traintuple
# TODO: test RemoteDataMethod


def test_remote():
    """Test that the remote decorator works properly with the RemoteStruct"""
    my_remote_class = RemoteClass(50, 20, a=42, b=3)
    aggregate_op = my_remote_class.remote_func(shared_states=None)

    # Check that the output of a function decorated with remote is not the result
    # but an AggregateOperation object
    assert isinstance(aggregate_op, AggregateOperation)

    # Reconstruct the RemoteClass from the RemoteStruct
    remote_cls_parameters = aggregate_op.remote_struct.cls_parameters
    new_remote_class = aggregate_op.remote_struct.cls(*remote_cls_parameters["args"], **remote_cls_parameters["kwargs"])

    # Check that the args are still there
    assert new_remote_class.args == (50, 20)
    assert new_remote_class.kwargs == {"a": 42, "b": 3}

    # Execute the function itself
    result = new_remote_class.remote_func(_skip=True, shared_states=[4, 5])
    assert result == 9


def test_remote_skip():
    """Test that the _skip parameter works as expected"""
    my_remote_class = RemoteClass(50, 20, a=42, b=3)
    result = my_remote_class.remote_func(_skip=True, shared_states=[4, 5])
    assert result == 9


def test_remote_data():
    """Test that the remote_data decorator works properly with the RemoteStruct"""
    my_remote_class = RemoteClass(50, 20, a=42, b=3)
    data_op = my_remote_class.remote_data_func(data_samples=["fake_path", "fake_path_2"], shared_state=None)
    # Check that the output of a function decorated with remote_data is not the result
    # but a DataOperation object
    assert isinstance(data_op, DataOperation)

    # Reconstruct the RemoteClass from the RemoteStruct
    remote_cls_parameters = data_op.remote_struct.cls_parameters
    new_remote_class = data_op.remote_struct.cls(*remote_cls_parameters["args"], **remote_cls_parameters["kwargs"])

    # Check that the args are still there
    assert new_remote_class.args == (50, 20)
    assert new_remote_class.kwargs == {"a": 42, "b": 3}

    # Execute the function itself
    result = new_remote_class.remote_data_func(x=4, y=5, _skip=True, shared_state=4)
    assert result == 13


def test_remote_data_get_method_from_remote_struct():
    """Test that the remote_data decorator works properly with the RemoteStruct
    when getting the name of the method from the RemoteStruct"""
    my_remote_class = RemoteClass()
    data_op = my_remote_class.remote_data_func(data_samples=["fake_path", "fake_path_2"], shared_state=None)

    assert isinstance(data_op, DataOperation)

    new_remote_class = data_op.remote_struct.cls(data_op.remote_struct.cls_parameters)
    new_remote_data_func = getattr(
        new_remote_class,
        data_op.remote_struct.remote_cls_parameters["kwargs"]["method_name"],
    )

    result = new_remote_data_func(x=4, y=5, _skip=True, shared_state=4)

    assert result == 13


def test_remote_data_extra_arg():
    """Test that the remote_data decorator works properly with the RemoteStruct and that an
    extra argument in the function is saved in the RemoteStruct"""
    my_remote_class = RemoteClass()
    data_op = my_remote_class.remote_data_func(
        data_samples=["fake_path", "fake_path_2"], shared_state=None, extra_arg=100
    )
    assert isinstance(data_op, DataOperation)

    new_remote_class = data_op.remote_struct.cls(data_op.remote_struct.cls_parameters)

    result = new_remote_class.remote_data_func(
        x=4,
        y=5,
        _skip=True,
        shared_state=4,
        **data_op.remote_struct.remote_cls_parameters["kwargs"]["method_parameters"],  # extra arg is there
    )

    assert result == 113


def test_security_remote_data():
    # TODO: make sure that the function with this decorator has only access
    # to the given dataset and not that of ofther nodes
    # (same for remote decorator - that it does not have access to any data)
    pass
