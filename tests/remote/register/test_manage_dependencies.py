import os
import sys
import tempfile
from pathlib import Path

import pytest
import substratools

from substrafl.exceptions import IncompatibleDependenciesError
from substrafl.exceptions import InvalidUserModuleError
from substrafl.remote.register.manage_dependencies import build_user_dependency_wheel
from substrafl.remote.register.manage_dependencies import compile_requirements
from substrafl.remote.register.manage_dependencies import get_pypi_dependencies_versions
from substrafl.remote.register.manage_dependencies import local_lib_wheels

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
        dest_dir=operation_dir / "substrafl_internal/dist",
    )

    wheels_created = []

    for lib in libs:
        wheels_created.append(
            (
                operation_dir / "substrafl_internal" / "dist" / f"{lib.__name__}-{lib.__version__}-py3-none-any.whl"
            ).exists()
        )

    assert all(wheels_created)


def test_get_pypi_dependencies_versions():
    # Test that pypi wheel can be downloaded
    libs = [substratools]

    # We check that we have access to the pypi repo not the specific packages version otherwise this test will fail
    # when trying to create a new version of substrafl as the running dev version on the ci and on a local computer
    # (0.x.0) won't have been released yet.

    # save the current versions the libs to set them back later
    substratools_version = substratools.__version__

    dependencies = get_pypi_dependencies_versions(lib_modules=libs)
    assert dependencies == [f"substratools=={substratools_version}"]


def test_compile_requirements(tmp_path):
    os.mkdir(tmp_path / "substrafl_internals")
    compile_requirements(
        ["substrafl", "substra", "substratools", "numpy"], operation_dir=tmp_path, sub_dir="substrafl_internals"
    )
    requirements_path = tmp_path / "substrafl_internals" / "requirements.txt"
    assert requirements_path.exists()
    assert "substrafl" in requirements_path.read_text()
    assert "numpy" in requirements_path.read_text()


def test_compile_requirements_incompatible_versions(tmp_path):
    os.mkdir(tmp_path / "substrafl_internals")
    dependency_list = ["numpy == 1.11", "numpy == 1.12"]
    with pytest.raises(IncompatibleDependenciesError):
        compile_requirements(dependency_list, operation_dir=tmp_path, sub_dir="substrafl_internals")
