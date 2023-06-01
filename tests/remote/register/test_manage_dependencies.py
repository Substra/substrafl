import os
import sys
import tempfile
from pathlib import Path

import pytest
import substratools

from substrafl.exceptions import IncompatibleDependenciesError
from substrafl.exceptions import InvalidUserModuleError
from substrafl.remote.register.manage_dependencies import _compile_requirements
from substrafl.remote.register.manage_dependencies import build_user_dependency_wheel
from substrafl.remote.register.manage_dependencies import local_lib_wheels
from substrafl.remote.register.manage_dependencies import pypi_lib_wheels

PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"


SETUP_CONTENT = """from setuptools import setup, find_packages

setup(
    name='mymodule',
    version='1.0.2',
    author='Author Name',
    description='Description of my package',
    packages=find_packages(),
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 3.5.1'],
)"""


def test_build_user_dependency_wheel(tmp_path):
    operation_dir = tmp_path / "local_dir"
    os.mkdir(operation_dir)
    module_root = operation_dir / "my_module"
    os.mkdir(module_root)
    setup_file = module_root / "setup.py"
    setup_file.write_text(SETUP_CONTENT)
    wheel_name = build_user_dependency_wheel(module_root, operation_dir)
    assert wheel_name == "mymodule-1.0.2-py3-none-any.whl"
    assert (operation_dir / wheel_name).exists()


def test_build_user_dependency_wheel_invalid_wheel(tmp_path):
    operation_dir = tmp_path / "local_dir"
    os.mkdir(operation_dir)
    module_root = operation_dir / "my_module"
    os.mkdir(module_root)
    setup_file = module_root / "setup.py"
    setup_file.write_text("foobar")
    with pytest.raises(InvalidUserModuleError):
        build_user_dependency_wheel(module_root, operation_dir)


def test_generate_local_lib_wheel(session_dir):
    # Test that editable wheel are generated
    libs = [substratools]
    operation_dir = Path(tempfile.mkdtemp(dir=session_dir.as_posix()))

    local_lib_wheels(
        lib_modules=libs,
        operation_dir=operation_dir,
        python_major_minor=PYTHON_VERSION,
        dest_dir="substrafl_internal/dist",
    )

    wheels_created = []

    for lib in libs:
        wheels_created.append((operation_dir / f"{lib.__name__}-{lib.__version__}-py3-none-any.whl").exists())

    assert all(wheels_created)


def test_generate_pypi_lib_wheel(session_dir):
    # Test that pypi wheel can be downloaded
    libs = [substratools]
    operation_dir = Path(tempfile.mkdtemp(dir=session_dir.as_posix()))

    # We check that we have access to the pypi repo not the specific packages version otherwise this test will fail
    # when trying to creates a new version of substrafl as the running dev version on the ci and on a local computer
    # (0.x.0) won't have been released yet.

    # save the current versions the libs to set them back later
    substratools_version = substratools.__version__
    substratools.__version__ = "0.7.0"

    pypi_lib_wheels(
        lib_modules=libs,
        python_major_minor=PYTHON_VERSION,
        operation_dir=operation_dir,
        dest_dir="substrafl_internal/dist",
    )

    wheels_created = []

    for lib in libs:
        wheels_created.append(
            (Path().home() / f".substrafl/{lib.__name__}-{lib.__version__}-py3-none-any.whl").exists()
        )

    substratools.__version__ = substratools_version

    assert all(wheels_created)


def test_compile_requirements(tmp_path):
    _compile_requirements(["substrafl", "substra", "substratools", "numpy"], tmp_path)
    requirements_path = tmp_path / "requirements.txt"
    assert requirements_path.exists()
    assert "substrafl" in requirements_path.read_text()
    assert "numpy" in requirements_path.read_text()


def test_compile_requirements_incompatible_versions(tmp_path):
    dependency_list = ["numpy == 1.11", "numpy == 1.12"]
    with pytest.raises(IncompatibleDependenciesError):
        _compile_requirements(dependency_list, tmp_path)
