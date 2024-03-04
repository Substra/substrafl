"""
Decorators to wrap functions
so that they are executed on the remote organizations.
"""

from functools import wraps
from typing import Any
from typing import Callable
from typing import List
from typing import Optional

from substrafl.remote.operations import RemoteDataOperation
from substrafl.remote.operations import RemoteOperation
from substrafl.remote.remote_struct import RemoteStruct
from substrafl.remote.substratools_methods import RemoteMethod


def remote_data(method: Callable):
    """Decorator for a remote function containing a ``data_samples`` argument (e.g the ``Algo.train`` function)
    With this decorator, when the function is called, it is not executed but it returns a ``RemoteDataOperation``
    object containing all the informations needed to execute it later
    (see ``substrafl.remote.operations.RemoteDataOperation``).

        - The decorated function definition should have at least a shared_state argument
        -   If the decorated function is called without a ``_skip=True`` argument, the arguments required
            are the ones in ``remote_method_inner``, and it should have at least a ``data_samples`` argument
        -   If the decorated function is called with a ``_skip=True`` argument, it should have the arguments of its
            original definition
        - The decorated function should be within a class
        - The ``__init__`` of the class must be

            .. code-block:: python

                def __init__(self, *args, **kwargs):
                    self.args = args
                    self.kwargs = kwargs

        -   ``self.args`` and ``self.kwargs`` will be given to the init, any other init argument is ignored
            (not saved in the RemoteStruct)

    Args:
        method (Callable): Method to wrap so that it is executed on the remote server
    """

    @wraps(method)
    def remote_method_inner(
        self,
        data_samples: Optional[List[str]] = None,
        shared_state: Any = None,
        *,
        _skip: bool = False,
        _algo_name: Optional[str] = None,
        **method_parameters,
    ) -> RemoteDataOperation:
        """
        Args:
            data_samples (List[str]): The data samples paths. Defaults to None.
            shared_state (typing.Any): a shared state, could be a SharedStateRef object or anything else. Defaults to
                None.
            _skip (bool, Optional): if True, calls the decorated function. Defaults to False.
            _algo_name(str, Optional): opportunity to set a custom algo name.
                Defaults to None.

        Returns:
            RemoteDataOperation: resulting RemoteDataOperation
        """
        if _skip:
            return method(self=self, shared_state=shared_state, **method_parameters)

        assert data_samples is not None

        assert "data_from_opener" not in method_parameters.keys()

        return RemoteDataOperation(
            RemoteStruct(
                cls=self.__class__,
                cls_args=self.args,
                cls_kwargs=self.kwargs,
                method_name=method.__name__,
                method_parameters=method_parameters,
                algo_name=_algo_name,
                remote_cls=RemoteMethod,
            ),
            data_samples,
            shared_state,
        )

    return remote_method_inner


def remote(method: Callable):
    """Decorator for a remote function.
    With this decorator, when the function is called, it is not executed but it returns a ``RemoteOperation``
    object containing all the informations needed to execute it later (see
    ``substrafl.remote.operations.RemoteOperation``).

        - The decorated function definition should have at least a shared_state argument
        -   If the decorated function is called without a ``_skip=True`` argument, the arguments required
            are the ones in ``remote_method_inner``
        -   If the decorated function is called with a ``_skip=True`` argument, it should have the arguments of its
            original definition
        - The decorated function should be within a class
        - The ``__init__`` of the class must be

            .. code-block:: python

                def __init__(self, *args, **kwargs):
                    self.args = args
                    self.kwargs = kwargs

        -   ``self.args`` and ``self.kwargs`` will be given to the init, any other init argument is ignored
            (not saved in the RemoteStruct)

    Args:
        method (Callable): Method to wrap so that it is executed on the remote server
    """

    @wraps(method)
    def remote_method_inner(
        self,
        shared_states: Optional[List] = None,
        *,
        _skip: bool = False,
        _algo_name: Optional[str] = None,
        **method_parameters,
    ) -> RemoteOperation:
        if _skip:
            return method(self=self, shared_states=shared_states, **method_parameters)

        return RemoteOperation(
            RemoteStruct(
                cls=self.__class__,
                cls_args=self.args,
                cls_kwargs=self.kwargs,
                method_name=method.__name__,
                method_parameters=method_parameters,
                algo_name=_algo_name,
                remote_cls=RemoteMethod,
            ),
            shared_states,
        )

    return remote_method_inner
