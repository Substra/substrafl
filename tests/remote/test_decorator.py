from typing import List

from substrafl.remote.decorators import remote
from substrafl.remote.decorators import remote_data
from substrafl.remote.operations import AggregateOperation
from substrafl.remote.operations import DataOperation

# TODO: these are actually integration tests between the decorator, the RemoteStruct and the Remote methods


class RemoteClass:
    # test Class for testing if the remote_data and train work correctly
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @remote
    def aggregate(self, shared_states: List[int], extra_arg: int = 0) -> int:
        return sum(shared_states) + extra_arg

    @remote_data
    def train(self, x: int, y: int, shared_state: int, extra_arg: int = 0) -> int:
        return x + y + shared_state + extra_arg


def test_remote_data():
    """Test that the remote_data decorator works properly with the RemoteStruct"""
    my_remote_class = RemoteClass(50, 20, a=42, b=3)
    data_op = my_remote_class.train(data_samples=["fake_path", "fake_path_2"], shared_state=None)
    # Check that the output of a function decorated with remote_data is not the result
    # but a DataOperation object
    assert isinstance(data_op, DataOperation)

    # Reconstruct the RemoteClass from the RemoteStruct
    remote_struct = data_op.remote_struct
    new_class = remote_struct.get_instance()

    # Check that the args are still there
    assert new_class.args == (50, 20)
    assert new_class.kwargs == {"a": 42, "b": 3}

    # Execute the function itself
    result = new_class.train(x=4, y=5, _skip=True, shared_state=4)
    assert result == 13


def test_remote_data_get_method_from_remote_struct():
    """Test that the remote_data decorator works properly with the RemoteStruct
    when getting the name of the method from the RemoteStruct"""
    my_remote_class = RemoteClass()
    data_op = my_remote_class.train(data_samples=["fake_path", "fake_path_2"], shared_state=None)

    assert isinstance(data_op, DataOperation)

    new_remote_class = data_op.remote_struct.get_remote_instance()
    result = new_remote_class.train(
        X=4,
        y=5,
        trunk_model=4,
        head_model=None,
        rank=0,
    )

    assert result[1] == 13


def test_remote_data_extra_arg():
    """Test that the remote_data decorator works properly with the RemoteStruct and that an
    extra argument in the function is saved in the RemoteStruct"""
    my_remote_class = RemoteClass()
    data_op = my_remote_class.train(data_samples=["fake_path", "fake_path_2"], shared_state=None, extra_arg=100)
    assert isinstance(data_op, DataOperation)

    new_remote_class = data_op.remote_struct.get_remote_instance()
    result = new_remote_class.train(
        X=4,
        y=5,
        trunk_model=4,
        head_model=None,
        rank=0,
    )

    assert result[1] == 113


def test_remote():
    """Test that the remote decorator works properly with the RemoteStruct"""
    my_remote_class = RemoteClass(50, 20, a=42, b=3)
    aggregate_op = my_remote_class.aggregate(shared_states=None)

    # Check that the output of a function decorated with remote is not the result
    # but an AggregateOperation object
    assert isinstance(aggregate_op, AggregateOperation)

    # Reconstruct the RemoteClass from the RemoteStruct
    new_remote_class = aggregate_op.remote_struct.get_instance()

    # Check that the args are still there
    assert new_remote_class.args == (50, 20)
    assert new_remote_class.kwargs == {"a": 42, "b": 3}

    # Execute the function itself
    result = new_remote_class.aggregate(_skip=True, shared_states=[4, 5])
    assert result == 9


def test_remote_skip():
    """Test that the _skip parameter works as expected"""
    my_remote_class = RemoteClass(50, 20, a=42, b=3)
    result = my_remote_class.aggregate(_skip=True, shared_states=[4, 5])
    assert result == 9
