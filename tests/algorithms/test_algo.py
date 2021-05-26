import json
import pytest
import cloudpickle

from connectlib.algorithms import register


def test_register():
    @register
    class TestClass:
        def __init__(self, a, b, c):
            self.a = a
            self.b = b
            self.c = c

    registered_instance = TestClass(1, 2, 3)

    with registered_instance.parameters_path.open("r") as f:
        parameters = json.load(f)

    with registered_instance.cloudpickle_path.open("rb") as f:
        cls = cloudpickle.load(f)

    instance = cls(*parameters["args"], **parameters["kwargs"])

    assert instance.a == 1
    assert instance.b == 2
    assert instance.c == 3


def test_register_wrong_parameters():
    class A:
        pass

    @register
    class TestClass:
        def __init__(self, a):
            self.a = a

    with pytest.raises(TypeError):
        TestClass(A())
