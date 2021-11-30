import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import substratools

from connectlib.remote.serializers import PickleSerializer, Serializer


class RemoteDataMethod(substratools.CompositeAlgo):
    # TODO: docstring, tests
    def __init__(
        self,
        instance,
        method_name: str,
        method_parameters: Dict,
        fake_traintuple: bool = False,
        shared_state_serializer: Type[Serializer] = PickleSerializer,
    ):
        self.instance = instance
        self.instance.delayed_init(instance.seed, *instance.args, **instance.kwargs)

        self.method_name = method_name
        self.fake_traintuple = fake_traintuple
        self.method_parameters = method_parameters

        self.shared_state_serializer = shared_state_serializer

    def train(
        self,
        X: Any,
        y: Any,
        head_model: Any,  # instance of algo
        trunk_model: Any,  # shared state
        rank: int,
    ) -> Tuple:
        if not self.fake_traintuple:
            # head_model should be None only at initialization
            if head_model is not None:
                instance = head_model
            else:
                instance = self.instance

            method_to_call = getattr(instance, self.method_name)
            next_shared_state = method_to_call(
                x=X, y=y, shared_state=trunk_model, _skip=True, **self.method_parameters
            )

            return instance, next_shared_state
        else:
            return head_model, trunk_model

    def predict(self, X: Any, head_model: Any, trunk_model: Any):
        assert (
            head_model is not None
        ), "head model is None. Possibly you did not train() before running predict()"
        instance = head_model

        method_to_call = getattr(instance, self.method_name)
        predictions = method_to_call(
            x=X, shared_state=trunk_model, _skip=True, **self.method_parameters
        )

        return predictions

    def load_trunk_model(self, path: str):
        return self.shared_state_serializer.load(Path(path))

    def save_trunk_model(self, model, path: str):
        self.shared_state_serializer.save(model, Path(path))

    def load_head_model(self, path: str):
        return self.instance.load(Path(path))

    def save_head_model(self, model, path: str):
        model.save(Path(path))


class RemoteMethod(substratools.AggregateAlgo):
    # TODO: docstring, tests
    def __init__(
        self,
        instance,
        method_name: str,
        method_parameters: Dict,
        shared_state_serializer: Type[Serializer] = PickleSerializer,
    ):
        self.instance = instance
        self.instance.delayed_init(instance.seed, *instance.args, **instance.kwargs)

        self.method_name = method_name
        self.method_parameters = method_parameters

        self.shared_state_serializer = shared_state_serializer

    def aggregate(self, models, rank):
        method_to_call = getattr(self.instance, self.method_name)
        next_shared_state = method_to_call(
            shared_states=models, _skip=True, **self.method_parameters
        )

        return next_shared_state

    def predict(self, X, model):
        """This predict method is required by substratools"""
        return

    def load_model(self, path: str):
        return self.shared_state_serializer.load(Path(path))

    def save_model(self, model, path: str):
        self.shared_state_serializer.save(model, Path(path))


class RemoteStruct:
    """Represents a submittable substra object.
    E.g.: An algorithm, a dataset, an objective

    Args:
        cls (Type): The remote struct type (e.g. Algorithm, dataset)
        cls_parameters (str): The class parameters serialized into json string.
            E.g.: use `json.dumps({"args": [], "kwargs": kwargs})`
        remote_cls_name (str): The name of the class used remotely
        remote_cls_parameters (str): The remote class parameters serialized into json string.
            E.g.: use `json.dumps({"args": [], "kwargs": kwargs})`
    """

    def __init__(
        self,
        cls: Type,
        cls_parameters: str,
        remote_cls_name: str,
        remote_cls_parameters: str,
    ):
        self.cls = cls
        self.cls_parameters = cls_parameters
        self.remote_cls_name = remote_cls_name
        self.remote_cls_parameters = remote_cls_parameters

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RemoteStruct):
            return NotImplemented
        return (
            self.cls == other.cls
            and self.cls_parameters == other.cls_parameters
            and self.remote_cls_name == other.remote_cls_name
            and self.remote_cls_parameters == other.remote_cls_parameters
        )

    def __hash__(self):
        return hash(
            (
                self.cls,
                self.cls_parameters,
                self.remote_cls_name,
                self.remote_cls_parameters,
            )
        )


@dataclass
class DataOperation:
    remote_struct: RemoteStruct
    data_samples: List[str]
    shared_state: Any


@dataclass
class AggregateOperation:
    remote_struct: RemoteStruct
    shared_states: Optional[List]


def remote_data(method: Callable):
    """Decorator for a remote function containing a data_samples argument (e.g the Algo.train function)
    With this decorator, when the function is called, it is not executed but it returns a DataOperation object containing
    all the informations needed to execute it later (see connectlib.remote.methods.DataOperation).
    - The decorated function definition should have at least a shared_state argument
    - If the decorated function is called without a `_skip=True` argument, the arguments required are the ones in
    remote_method_inner, and it should have at least a `datasamples` argument
    - If the decorated function is called with a `_skip=True` argument, it should have the arguments of its original definition
    - The decorated function should be within a class
    - The init of the class must be
    def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
    - self.args and self.kwargs will be given to the init, any other init argument is ignored (not saved in the RemoteStruct)

    """

    def remote_method_inner(
        self,
        data_samples: Optional[List[str]] = None,
        shared_state: Any = None,
        _skip: bool = False,
        fake_traintuple: bool = False,
        **method_parameters
    ) -> DataOperation:
        """
        Args:
            data_samples (List[str]): The data samples paths. Defaults to None.
            shared_state (Any): a shared state, could be a SharedStateRef object or anything else. Defaults to None.
            _skip (bool, optional): if True, calls the decorated function. Defaults to False.
            fake_traintuple (bool, optional): if True, the decorated function won't be executed (see RemoteDataMethod). Defaults to False.

        Returns:
            DataOperation: [description]
        """
        if _skip:
            return method(self=self, shared_state=shared_state, **method_parameters)

        assert data_samples is not None

        assert "x" not in method_parameters.keys()
        assert "y" not in method_parameters.keys()

        cls = self.__class__
        cls_parameters = json.dumps({"args": self.args, "kwargs": self.kwargs})

        kwargs = {
            "method_name": method.__name__,
            "method_parameters": method_parameters,
            "fake_traintuple": fake_traintuple,
        }
        remote_cls_parameters = json.dumps({"args": [], "kwargs": kwargs})

        return DataOperation(
            RemoteStruct(
                cls, cls_parameters, "RemoteDataMethod", remote_cls_parameters
            ),
            data_samples,
            shared_state,
        )

    return remote_method_inner


def remote(method: Callable):
    # TODO: add docstring
    def remote_method_inner(
        self,
        shared_states: Optional[List] = None,
        _skip: bool = False,
        **method_parameters
    ) -> AggregateOperation:
        if _skip:
            return method(self=self, shared_states=shared_states, **method_parameters)

        cls = self.__class__
        cls_parameters = json.dumps({"args": self.args, "kwargs": self.kwargs})

        kwargs = {
            "method_name": method.__name__,
            "method_parameters": method_parameters,
        }
        remote_cls_parameters = json.dumps({"args": [], "kwargs": kwargs})

        return AggregateOperation(
            RemoteStruct(cls, cls_parameters, "RemoteMethod", remote_cls_parameters),
            shared_states,
        )

    return remote_method_inner
