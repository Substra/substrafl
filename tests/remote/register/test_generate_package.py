import sys
import tempfile
from pathlib import Path

import substratools

from substrafl.remote.register.generate_wheel import local_lib_wheels
from substrafl.remote.register.generate_wheel import pypi_lib_wheels

PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"


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
        wheels_created.append(
            (operation_dir / f"substrafl_internal/dist/{lib.__name__}-{lib.__version__}-py3-none-any.whl").exists()
        )

    assert all(wheels_created)


def test_generate_pypi_lib_wheel(session_dir):
    # Test that pypi wheel can be downloaded
    libs = [substratools]
    operation_dir = Path(tempfile.mkdtemp(dir=session_dir.as_posix()))

    # We check that we have access to the pypi repo not the specific packages version otherwise this test will fail
    # when trying to creates a new version of substrafl as the running dev version on the ci and on a local computer
    # (0.x.0) won't have been released yet (it is not in owkin pypi).

    # save the current versions the libs to set them back later
    substratools_version = substratools.__version__
    substratools.__version__ = "0.9.1"

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
