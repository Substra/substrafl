import tempfile
import pytest

from pathlib import Path


@pytest.fixture
def temp_folder():
    test_dir = tempfile.TemporaryDirectory(prefix="tests")
    yield Path(test_dir.name)

    # delete the temp test_dir
    test_dir.cleanup()
