"""
Generate wheels for the Connect algo.
"""
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

LOCAL_WHEELS_FOLDER = Path.home() / ".connectlib"


def local_lib_wheels(lib_modules: List, operation_dir: Path, python_major_minor: str, dest_dir: str) -> str:
    """Prepares the private modules from lib_modules list to be installed in a Docker image and generates the
    appropriated install command for a dockerfile. It first creates the wheel for each library. Each of the
    libraries must be already installed in the correct version locally. Use command:
    ``pip install -e library-name`` in the directory of each library.

    This allows one user to use custom version of the passed modules.

    Args:
        lib_modules (list): list of modules to be installed.
        operation_dir (pathlib.Path): PosixPath to the operation directory
        python_major_minor (str): version which is to be used in the dockerfile. Eg: '3.8'
        dest_dir (str): relative directory where the wheels are saved

    Returns:
        str: dockerfile command for installing the given modules
    """
    install_cmds = []
    wheels_dir = operation_dir / dest_dir
    wheels_dir.mkdir(exist_ok=True, parents=True)
    for lib_module in lib_modules:
        wheel_name = f"{lib_module.__name__}-{lib_module.__version__}-py3-none-any.whl"

        if not (Path(lib_module.__file__).parents[1] / "setup.py").exists():
            msg = ", ".join([lib.__name__ for lib in lib_modules])
            raise NotImplementedError(
                f"You must install {msg} in editable mode.\n" "eg `pip install -e substra` in the substra directory"
            )
        lib_name = lib_module.__name__
        lib_path = Path(lib_module.__file__).parents[1]
        wheel_name = f"{lib_name}-{lib_module.__version__}-py3-none-any.whl"

        wheel_path = LOCAL_WHEELS_FOLDER / wheel_name
        # Recreate the wheel only if it does not exist
        if wheel_path.exists():
            logger.warning(
                f"Existing wheel {wheel_path} will be used to build {lib_name}. "
                "It may lead to errors if you are using an unreleased version of this lib: "
                "if it's the case, you can delete the wheel and it will be re-generated."
            )
        else:
            # if the right version of substra or substratools is not found, it will search if they are already
            # installed in 'dist' and take them from there.
            # sys.executable takes the Python interpreter run by the code and not the default one on the computer
            extra_args: list = list()
            if lib_name == "connectlib":
                extra_args = [
                    "--find-links",
                    operation_dir / "dist/substra",
                    "--find-links",
                    operation_dir / "dist/substratools",
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

        shutil.copy(wheel_path, wheels_dir / wheel_name)

        # Necessary command to install the wheel in the docker image
        force_reinstall = "--force-reinstall " if lib_name == "substratools" else ""
        install_cmd = (
            f"RUN cd {dest_dir} && python{python_major_minor}" f" -m pip install {force_reinstall}{wheel_name}\n"
        )
        install_cmds.append(install_cmd)

    return "\n".join(install_cmds)


def pypi_lib_wheels(lib_modules: List, operation_dir: Path, python_major_minor: str, dest_dir: str) -> str:
    """Retrieves lib_modules' wheels from Owkin private repo (if needed) to be installed in a Docker image and generates
    the appropriated install command for a dockerfile.

    Args:
        lib_modules (list): list of modules to be installed.
        operation_dir (pathlib.Path): PosixPath to the operation directory
        python_major_minor (str): version which is to be used in the dockerfile. Eg: '3.8'
        dest_dir (str): relative directory where the wheels are saved

    Returns:
        str: dockerfile command for installing the given modules
    """
    install_cmds = []
    wheels_dir = operation_dir / dest_dir
    wheels_dir.mkdir(exist_ok=True, parents=True)

    LOCAL_WHEELS_FOLDER.mkdir(exist_ok=True)

    for lib_module in lib_modules:

        wheel_name = f"{lib_module.__name__}-{lib_module.__version__}-py3-none-any.whl"

        # Download only if exists
        if not ((LOCAL_WHEELS_FOLDER / wheel_name).exists()):
            try:
                subprocess.check_output(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "download",
                        "--only-binary",
                        ":all:",
                        "--python-version",
                        python_major_minor,
                        "--no-deps",
                        "--implementation",
                        "py",
                        "-d",
                        LOCAL_WHEELS_FOLDER,
                        f"{lib_module.__name__}=={lib_module.__version__}",
                    ]
                )
            except subprocess.CalledProcessError as e:
                raise ConnectionError(
                    "Couldn't access to Owkin pypi, please ensure you have access to https://pypi.owkin.com/simple/.",
                    e.output,
                )

        # Get wheel name based on current version
        shutil.copy(LOCAL_WHEELS_FOLDER / wheel_name, wheels_dir / wheel_name)
        install_cmd = f"RUN cd {dest_dir} && python{python_major_minor} -m pip install {wheel_name}\n"
        install_cmds.append(install_cmd)

    return "\n".join(install_cmds)
