"""
Utility functions to manage dependencies (building wheels, compiling requirement...)
"""
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

from substrafl.dependency import Dependency
from substrafl.exceptions import InvalidDependenciesError
from substrafl.exceptions import InvalidUserModuleError

logger = logging.getLogger(__name__)

LOCAL_WHEELS_FOLDER = Path.home() / ".substrafl"


def build_user_dependency_wheel(lib_path: Path, operation_dir: Path) -> str:
    """Build the wheel for user dependencies passed as a local module.

    Args:
        lib_path (Path): where the module is located.
        operation_dir (Path): where the wheel needs to be copied.

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
                str(lib_path) + "/",
                "--no-deps",
            ],
            cwd=str(operation_dir),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise InvalidUserModuleError from e
    wheel_name = re.findall(r"filename=(\S*)", ret.stdout)[0]

    return wheel_name


def copy_local_wheels(path: Path, dependencies: Dependency) -> List[str]:
    """Copy the local modules given by the user, generating the wheel if necessary.

    Args:
        path (Path): the path where the local wheels will be copied.
        dependencies (Dependency): Dependency object from which the list of local installable dependencies
            (wheels or modules) will be extracted.

    Returns: list of wheel paths relative to `path`
    """
    path.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        dependencies_paths = dependencies.copy_dependencies_local_package(dest_dir=tmp_dir)
        wheel_paths = []
        for dependency in dependencies_paths:
            if dependency.__str__().endswith(".whl"):
                wheel_paths.append(str(dependency))
                shutil.copy(tmp_dir / dependency, path)
            else:
                wheel_name = build_user_dependency_wheel(
                    Path(dependency),
                    operation_dir=tmp_dir,
                )
                wheel_paths.append(wheel_name)
                shutil.copy(tmp_dir / wheel_name, path)
    return wheel_paths


def local_lib_wheels(lib_modules: List, *, dest_dir: Path) -> List[str]:
    """Generate wheels for the private modules from lib_modules list and returns the list of names for each wheel.

     It first creates the wheel for each library. Each of the libraries must be already installed in the correct
     version locally. Use command: ``pip install -e library-name`` in the directory of each library.
     Then it copies the wheels to the dest_dir.

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
        if not (lib_path / "setup.py").exists():
            msg = ", ".join([lib.__name__ for lib in lib_modules])
            raise NotImplementedError(
                f"You must install {msg} in editable mode.\n" "eg `pip install -e substra` in the substra directory"
            )
        lib_name = lib_module.__name__
        wheel_name = f"{lib_name}-{lib_module.__version__}-py3-none-any.whl"

        wheel_path = LOCAL_WHEELS_FOLDER / wheel_name
        # Recreate the wheel only if it does not exist
        if wheel_path.exists():
            logger.warning(
                f"Existing wheel {wheel_path} will be used to build {lib_name}. "
                "It may lead to errors if you are using an unreleased version of this lib: "
                "if it's the case, you can delete the wheel and it will be re-generated."
            )
        else:
            # if the right version of substra or substratools is not found, it will search if they are already
            # installed in 'dist' and take them from there.
            # sys.executable takes the current Python interpreter instead of the default one on the computer
            extra_args: list = []
            if lib_name == "substrafl":
                extra_args = [
                    "--find-links",
                    dest_dir.parent / "substra",
                    "--find-links",
                    dest_dir.parent / "substratools",
                ]
            subprocess.check_output(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    ".",
                    "-w",
                    LOCAL_WHEELS_FOLDER,
                    "--no-deps",
                ]
                + extra_args,
                cwd=str(lib_path),
            )

        shutil.copy(wheel_path, dest_dir / wheel_name)
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


def compile_requirements(dependency_list: List[str], *, operation_dir: Path, sub_dir: Path) -> None:
    """Compile a list of requirements using pip-compile to generate a set of fully pinned third parties requirements

    Writes down a `requirements.in` file with the list of explicit dependencies, then generates a `requirements.txt`
    file using pip-compile. The `requirements.txt` file contains a set of fully pinned dependencies, including indirect
    dependencies.

    Args:
        dependency_list: list of dependencies to install; acceptable formats are library names (eg "substrafl"),
            any constraint expression accepted by pip("substrafl==0.36.0" or "substrafl < 1.0") or wheel names
            ("substrafl-0.36.0-py3-none-any.whl")
        operation_dir: path to the root dir
        sub_dir: sub directory of the root dir where the `requirements.in` and `requirements.txt` files will be created

    Raises:
        InvalidDependenciesError: if pip-compile does not find a set of compatible dependencies

    """
    requirements_in = operation_dir / sub_dir / "requirements.in"

    requirements = ""
    for dependency in dependency_list:
        if dependency.__str__().endswith(".whl"):
            requirements += f"file:{dependency}\n"
        else:
            requirements += f"{dependency}\n"

    requirements_in.write_text(requirements)
    try:
        subprocess.check_output(
            [
                sys.executable,
                "-m",
                "piptools",
                "compile",
                "--resolver=backtracking",
                requirements_in,
            ],
            cwd=operation_dir,
        )
    except subprocess.CalledProcessError as e:
        raise InvalidDependenciesError from e
