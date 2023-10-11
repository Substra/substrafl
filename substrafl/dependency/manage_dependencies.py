"""
Utility functions to manage dependencies (building wheels, compiling requirement...)
"""
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
from pathlib import Path
from pathlib import PurePosixPath
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import List
from typing import Union

from substrafl import exceptions

logger = logging.getLogger(__name__)


def build_user_dependency_wheel(lib_path: Path, dest_dir: Path) -> str:
    """Build the wheel for user dependencies passed as a local module.
    Delete the local module when the build is done.

    Args:
        lib_path (Path): where the module is located.
        dest_dir (Path): where the wheel needs to be copied.

    Returns:
        str: the filename of the wheel
    """
    # sys.executable takes the current Python interpreter instead of the default one on the computer
    try:
        ret = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                str(lib_path) + os.sep,
                "--no-deps",
            ],
            cwd=str(dest_dir),
            check=True,
            capture_output=True,
            text=True,
        )

    except subprocess.CalledProcessError as e:
        raise exceptions.InvalidUserModuleError from e

    finally:
        # Delete the folder when the wheel is computed
        shutil.rmtree(dest_dir / lib_path)  # delete directory

    wheel_name = re.findall(r"filename=(\S*)", ret.stdout)[0]

    if not wheel_name.endswith(".whl"):
        raise exceptions.WrongWheelNameError(
            f"The extracted wheel name is not authorized. Extracted name should ends with .whl but got {wheel_name} "
            "instead."
        )
    elif not (dest_dir / wheel_name).exists():
        raise exceptions.WrongWheelNameError(f"The extracted wheel {str(dest_dir / wheel_name)} does not exist.")
    return wheel_name


def local_lib_wheels(lib_modules: List[ModuleType], *, dest_dir: Path) -> List[str]:
    """Generate wheels for the private modules from lib_modules list and returns the list of names for each wheel.

     It first creates the wheel for each library. Each of the libraries must be already installed in the correct
     version locally. Use command: ``pip install -e library-name`` in the directory of each library.
     Then it copies the wheels to the given directory.

    This allows one user to use custom version of the passed modules.

    Args:
        lib_modules (list): list of modules to be installed.
        dest_dir (str): relative directory where the wheels are saved

    Returns:
        List[str]: wheel names for the given modules
    """
    wheel_names = []
    dest_dir.mkdir(exist_ok=True, parents=True)
    for lib_module in lib_modules:
        lib_path = Path(lib_module.__file__).parents[1]
        # this function is in practice only called on substra libraries, and we know we use a setup.py
        if not (lib_path / "setup.py").exists():
            msg = ", ".join([lib.__name__ for lib in lib_modules])
            raise NotImplementedError(
                f"You must install {msg} in editable mode.\n" "eg `pip install -e substra` in the substra directory"
            )
        lib_name = lib_module.__name__
        wheel_name = f"{lib_name}-{lib_module.__version__}-py3-none-any.whl"

        # if the right version of substra or substratools is not found, it will search if they are already
        # installed in 'dist' and take them from there.
        # sys.executable takes the current Python interpreter instead of the default one on the computer

        extra_args = (
            [
                "--find-links",
                dest_dir.parent / "substra",
                "--find-links",
                dest_dir.parent / "substratools",
            ]
            if lib_name == "substrafl"
            else []
        )

        with TemporaryDirectory() as tmp_dir:
            ret = subprocess.check_output(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    ".",
                    "-w",
                    tmp_dir,
                    "--no-deps",
                ]
                + extra_args,
                cwd=str(lib_path),
                text=True,
            )
            created_wheel_name = re.findall(r"filename=(\S*)", ret)[0]

            if created_wheel_name != wheel_name:
                raise exceptions.WrongWheelNameError(
                    f"The `pip wheel` command built a wheel with an unexpected name: {created_wheel_name} "
                    f"instead of {wheel_name}."
                )

            shutil.copy(pathlib.Path(tmp_dir) / wheel_name, dest_dir / wheel_name)
        wheel_names.append(wheel_name)

    return wheel_names


def get_pypi_dependencies_versions(lib_modules: List) -> List[str]:
    """Retrieve the version of the PyPI libraries installed to generate the dependency list

    Args:
        lib_modules (list): list of modules to be installed.

    Returns:
        list(str): list of dependencies to install in the Docker container
    """
    return [f"{lib_module.__name__}=={lib_module.__version__}" for lib_module in lib_modules]


def write_requirements(dependency_list: List[Union[str, Path]], *, dest_dir: Path) -> None:
    """Writes down a `requirements.txt` file with the list of explicit dependencies.

    Args:
        dependency_list: list of dependencies to install; acceptable formats are library names (eg "substrafl"),
            any constraint expression accepted by pip("substrafl==0.36.0" or "substrafl < 1.0") or wheel names
            ("substrafl-0.36.0-py3-none-any.whl")
        dest_dir: path to the directory where to write the ``requirements.txt``.
    """
    requirements_txt = dest_dir / "requirements.txt"

    _write_requirements_file(dependency_list=dependency_list, file=requirements_txt)


def compile_requirements(dependency_list: List[Union[str, Path]], *, dest_dir: Path) -> None:
    """Compile a list of requirements using pip-compile to generate a set of fully pinned third parties requirements

    Writes down a `requirements.in` file with the list of explicit dependencies, then generates a `requirements.txt`
    file using pip-compile. The `requirements.txt` file contains a set of fully pinned dependencies, including indirect
    dependencies.

    Args:
        dependency_list: list of dependencies to install; acceptable formats are library names (eg "substrafl"),
            any constraint expression accepted by pip("substrafl==0.36.0" or "substrafl < 1.0") or wheel names
            ("substrafl-0.36.0-py3-none-any.whl")
        dest_dir: path to the directory where to write the ``requirements.in`` and ``requirements.txt``.

    Raises:
        InvalidDependenciesError: if pip-compile does not find a set of compatible dependencies

    """
    requirements_in = dest_dir / "requirements.in"

    _write_requirements_file(dependency_list=dependency_list, file=requirements_in)

    command = [
        sys.executable,
        "-m",
        "piptools",
        "compile",
        "--resolver=backtracking",
        str(requirements_in),
    ]
    try:
        logger.info("Compiling dependencies.")
        subprocess.run(
            command,
            cwd=dest_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise exceptions.InvalidDependenciesError(
            f"Error in command {' '.join(command)}\nstdout: {e.stdout}\nstderr: {e.stderr}"
        ) from e


def _write_requirements_file(dependency_list: List[Union[str, Path]], *, file: Path) -> None:
    requirements = ""
    for dependency in dependency_list:
        if str(dependency).endswith(".whl"):
            # Require '/', even on windows. The double conversion resolves that.
            requirements += f"file:{PurePosixPath(Path(dependency))}\n"
        else:
            requirements += f"{dependency}\n"
    file.write_text(requirements)
