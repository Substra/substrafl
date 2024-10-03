import os
import tempfile
from pathlib import Path

import numpy
import pytest
import substra

from substrafl.dependency.manage_dependencies import build_user_dependency_wheel
from substrafl.dependency.manage_dependencies import compile_requirements
from substrafl.dependency.manage_dependencies import get_pypi_dependencies_versions
from substrafl.dependency.manage_dependencies import local_lib_wheels
from substrafl.exceptions import InvalidDependenciesError
from substrafl.exceptions import InvalidUserModuleError


def test_build_user_dependency_wheel(tmp_path, local_installable_module):
    dest_dir = tmp_path / "local_dir"
    dest_dir.mkdir()
    module_root = local_installable_module(dest_dir)
    wheel_name = build_user_dependency_wheel(module_root, dest_dir)
    assert wheel_name == "mymodule-1.0.2-py3-none-any.whl"
    assert (dest_dir / wheel_name).exists()


def test_build_user_dependency_wheel_invalid_wheel(tmp_path):
    dest_dir = tmp_path / "local_dir"
    os.mkdir(dest_dir)
    module_root = dest_dir / "my_module"
    os.mkdir(module_root)
    setup_file = module_root / "setup.py"
    setup_file.write_text("foobar")
    with pytest.raises(InvalidUserModuleError):
        build_user_dependency_wheel(module_root, dest_dir)


def test_generate_local_lib_wheel(session_dir):
    # Test that editable wheel are generated
    libs = [substra]
    dest_dir = Path(tempfile.mkdtemp(dir=session_dir.as_posix()))

    local_lib_wheels(
        lib_modules=libs,
        dest_dir=dest_dir / "substrafl_internal" / "dist",
    )

    wheels_created = []

    for lib in libs:
        wheels_created.append(
            (dest_dir / "substrafl_internal" / "dist" / f"{lib.__name__}-{lib.__version__}-py3-none-any.whl").exists()
        )

    assert all(wheels_created)


def test_get_pypi_dependencies_versions():
    # Test that pypi wheel can be downloaded
    libs = [substra, numpy]

    # We check that we have access to the pypi repo not the specific packages version otherwise this test will fail
    # when trying to create a new version of substrafl as the running dev version on the ci and on a local computer
    # (0.x.0) won't have been released yet.

    dependencies = get_pypi_dependencies_versions(lib_modules=libs)
    assert dependencies == [f"substra=={substra.__version__}", f"numpy=={numpy.__version__}"]


def test_compile_requirements(tmp_path):
    compile_requirements(["substrafl", "substra", "numpy"], dest_dir=tmp_path)
    requirements_path = tmp_path / "requirements.txt"
    assert requirements_path.exists()
    assert "substrafl" in requirements_path.read_text()
    assert "numpy" in requirements_path.read_text()


@pytest.mark.parametrize("dependency_list", [["numpy == 1.11", "numpy == 1.12"], ["numpy", "pndas"]])
def test_compile_requirements_invalid_dependencies(tmp_path, dependency_list):
    with pytest.raises(InvalidDependenciesError):
        compile_requirements(dependency_list, dest_dir=tmp_path)
