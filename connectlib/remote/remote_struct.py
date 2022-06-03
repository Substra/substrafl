import inspect
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union

import cloudpickle

from connectlib.remote.connect_tools_methods import RemoteDataMethod
from connectlib.remote.connect_tools_methods import RemoteMethod


class RemoteStruct:
    """Contains the wrapped user code and the necessary functions
    to transform it into a Connect asset to execute on the platform.

    Args:
        cls (Type): The remote struct type (e.g. Algorithm, dataset)
        cls_parameters (str): The class parameters serialized into json string.
            E.g.: use ``json.dumps({"args": [], "kwargs": kwargs})``
        remote_cls (str): The name of the class used remotely
        remote_cls_parameters (str): The remote class parameters serialized into json string.
            E.g.: use ``json.dumps({"args": [], "kwargs": kwargs})``
        algo_name(str, Optional): opportunity to set a custom algo name.
            If None, set to "{method_name}_{class_name}"
    """

    def __init__(
        self,
        cls: Type,
        cls_args: list,
        cls_kwargs: dict,
        remote_cls: Union[Type[RemoteDataMethod], Type[RemoteMethod]],
        method_name: str,
        method_parameters: dict,
        algo_name: Optional[str],
    ):
        """
        Args:
            cls (Type): Locally defined class
            cls_args (list): Arguments (args) to instantiate the class
            cls_kwargs (dict): Arguments (kwargs) to instantiate the class
            remote_cls (Union[Type[RemoteDataMethod], Type[RemoteMethod]]): Remote class to create from the user code
            method_name (str): Name of the method from the local class to execute
            method_parameters (dict): Parameters to pass to the method
            algo_name(str, Optional): opportunity to set a custom algo name.
                If None, set to "{method_name}_{class_name}"
        """
        self._cls = cls
        self._cls_args = cls_args
        self._cls_kwargs = cls_kwargs
        self._remote_cls = remote_cls
        self._method_name = method_name
        self._method_parameters = method_parameters
        self._algo_name = algo_name or (self._method_name + "_" + self._cls.__name__)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RemoteStruct):
            return NotImplemented
        return (
            self._cls == other._cls
            and self._cls_args == other._cls_args
            and self._cls_kwargs == other._cls_kwargs
            and self._remote_cls == other._remote_cls
            and self._method_name == other._method_name
            and self._method_parameters == other._method_parameters
        )

    def __hash__(self):
        return hash(
            (
                self._cls,
                frozenset(self._cls_args),
                frozenset(self._cls_kwargs),
                self._remote_cls,
                self._method_name,
                frozenset(self._method_parameters),
            )
        )

    @property
    def algo_name(self):
        return self._algo_name

    @classmethod
    def load(cls, src: Path) -> "RemoteStruct":
        """Load the remote struct from the src
        directory.

        Args:
            src (pathlib.Path): Path to the directory where the remote struct has been saved.
        """
        cls_cloudpickle_path = src / "cls_cloudpickle"
        with cls_cloudpickle_path.open("rb") as f:
            instance = cloudpickle.load(f)
        return instance

    def save(self, dest: Path):
        """Save the instance to the dest directory using
        cloudpickle.

        Args:
            dest (pathlib.Path): directory where to save the remote struct
        """
        cloudpickle_path = dest / "cls_cloudpickle"
        with cloudpickle_path.open("wb") as f:
            cloudpickle.dump(self, f)

    def get_instance(self) -> Any:
        """Get the class instance.

        Returns:
            typing.Any: Instance of the saved class
        """
        return self._cls(*self._cls_args, **self._cls_kwargs)

    def get_remote_instance(self) -> Union[RemoteMethod, RemoteDataMethod]:
        """Get the remote class (ie Connect algo) instance.

        Returns:
            typing.Union[RemoteMethod, RemoteDataMethod]: instance of the remote Connect class
        """
        return self._remote_cls(
            self.get_instance(),
            method_name=self._method_name,
            method_parameters=self._method_parameters,
        )

    def get_cls_file_path(self) -> Path:
        """Get the path to the file where the cls attribute is defined.

        Returns:
            pathlib.Path: path to the file where the cls is defined.
        """
        try:
            algo_file_path = Path(inspect.getfile(self._cls)).resolve().parent
        except TypeError:
            # In a notebook, we get the TypeError: <class '__main__.MyAlgo'> is a built-in class
            # To fix it, we use the cwd of the notebook and assume local dependencies are there
            algo_file_path = Path.cwd().resolve()
        return algo_file_path

    def summary(self) -> Dict[str, str]:
        """Get a summary of what the remote struct represents.

        Returns:
            typing.Dict[str, str]: description
        """
        return {
            "type": self._cls.__name__,
            "method_name": self._method_name,
        }
