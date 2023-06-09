import os
import tempfile
from pathlib import Path

import numpy
import pytest
import substratools

from substrafl.dependency import Dependency
from substrafl.exceptions import InvalidDependenciesError
from substrafl.exceptions import InvalidUserModuleError
from substrafl.remote.register.manage_dependencies import build_user_dependency_wheel
from substrafl.remote.register.manage_dependencies import compile_requirements
from substrafl.remote.register.manage_dependencies import copy_local_wheels
from substrafl.remote.register.manage_dependencies import get_pypi_dependencies_versions
from substrafl.remote.register.manage_dependencies import local_lib_wheels

SETUP_CONTENT = """from setuptools import setup, find_packages

setup(
    name='mymodule',
    version='1.0.2',
    author='Author Name',
    description='Description of my package',
    packages=find_packages(),
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 3.5.1'],
)"""


def test_build_user_dependency_wheel(tmp_path, local_installable_module):
    operation_dir = tmp_path / "local_dir"
    operation_dir.mkdir()
    module_root = local_installable_module(operation_dir)
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


def test_copy_local_wheels(tmp_path, local_installable_module):
    dest_dir = tmp_path / "dest"
    src_dir = tmp_path / "src"
    dest_dir.mkdir()
    src_dir.mkdir()
    precompiled_path = src_dir / "precompiled-wheel-0.0.1-py3-none-any.whl"
    precompiled_path.touch()
    to_be_built_path = local_installable_module(src_dir)

    dependency_object = Dependency(local_installable_dependencies=[precompiled_path, to_be_built_path])
    wheel_paths = copy_local_wheels(dest_dir, dependency_object)
    assert wheel_paths == ["precompiled-wheel-0.0.1-py3-none-any.whl", "mymodule-1.0.2-py3-none-any.whl"]
    assert (dest_dir / "precompiled-wheel-0.0.1-py3-none-any.whl").is_file()
    assert (dest_dir / "mymodule-1.0.2-py3-none-any.whl").is_file()


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
    libs = [substratools, numpy]

    # We check that we have access to the pypi repo not the specific packages version otherwise this test will fail
    # when trying to create a new version of substrafl as the running dev version on the ci and on a local computer
    # (0.x.0) won't have been released yet.

    dependencies = get_pypi_dependencies_versions(lib_modules=libs)
    assert dependencies == [f"substratools=={substratools.__version__}", f"numpy=={numpy.__version__}"]


def test_compile_requirements(tmp_path):
    os.mkdir(tmp_path / "substrafl_internals")
    compile_requirements(
        ["substrafl", "substra", "substratools", "numpy"], operation_dir=tmp_path, sub_dir="substrafl_internals"
    )
    requirements_path = tmp_path / "substrafl_internals" / "requirements.txt"
    assert requirements_path.exists()
    assert "substrafl" in requirements_path.read_text()
    assert "numpy" in requirements_path.read_text()


@pytest.mark.parametrize("dependency_list", [["numpy == 1.11", "numpy == 1.12"], ["numpy", "pndas"]])
def test_compile_requirements_invalid_dependencies(tmp_path, dependency_list):
    os.mkdir(tmp_path / "substrafl_internals")
    with pytest.raises(InvalidDependenciesError):
        compile_requirements(dependency_list, operation_dir=tmp_path, sub_dir="substrafl_internals")
