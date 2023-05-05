import random
import string
import tempfile
from pathlib import Path
from typing import Optional

import pytest

from substrafl.dependency.path_management import TMP_SUBSTRAFL_PREFIX
from substrafl.dependency.path_management import DependencyPathManagement


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp_dir:
        yield Path(tmp_dir)


src_dir = tmp_dir
dest_dir = tmp_dir


@pytest.fixture
def create_random_path():
    def _create_random_path(base_dir: Path, *, path_name_length: int = 10) -> Path:
        path_name = "".join(random.choices(string.ascii_uppercase + string.digits, k=path_name_length))
        path = Path(base_dir / path_name)
        return path

    return _create_random_path


@pytest.fixture
def create_random_file(create_random_path):
    def _create_random_file(base_dir: Path, *, path_name_length: int = 10, suffix: Optional[str] = None) -> Path:
        path = create_random_path(base_dir, path_name_length=path_name_length)
        if suffix:
            path = path.with_suffix(suffix)
        path.parent.mkdir(exist_ok=True, parents=True)
        path.touch()
        return path

    return _create_random_file


@pytest.fixture
def create_random_folder(create_random_path):
    def _create_random_folder(base_dir: Path, *, path_name_length: int = 10) -> Path:
        path = create_random_path(base_dir, path_name_length=path_name_length)
        path.mkdir()
        return path

    return _create_random_folder


def test_expand_regex_file(tmp_dir, create_random_file):
    path = create_random_file(tmp_dir)
    paths = DependencyPathManagement.expand_regexes("*", [tmp_dir])
    assert paths == [path]


def test_expand_regex_folder(tmp_dir, create_random_folder, create_random_file):
    folder_path = create_random_folder(tmp_dir)
    path = create_random_file(folder_path)

    paths = DependencyPathManagement.expand_regexes("*", [folder_path])
    assert paths == [path]


def test_expand_paths_file(tmp_dir, create_random_file):
    path = create_random_file(tmp_dir)
    paths = DependencyPathManagement.expand_paths([tmp_dir])
    assert paths == {path}


def test_expand_paths_folders(tmp_dir, create_random_folder, create_random_file):
    folder_path = create_random_folder(tmp_dir)
    paths = {create_random_file(folder_path) for _ in range(10)}
    found_paths = DependencyPathManagement.expand_paths([tmp_dir])
    assert paths == found_paths


def test_get_excluded_paths_default(tmp_dir, create_random_file):
    csv_file = create_random_file(tmp_dir, suffix=".csv")
    # Try subfolder
    csv_subfolder_file = create_random_file(tmp_dir / "subfolder", suffix=".csv")
    xls_file = create_random_file(tmp_dir, suffix=".xls")
    xlsx_file = create_random_file(tmp_dir, suffix=".xlsx")
    npy_file = create_random_file(tmp_dir, suffix=".npy")
    png_file = create_random_file(tmp_dir, suffix=".png")
    jpg_file = create_random_file(tmp_dir, suffix=".jpg")
    jpeg_file = create_random_file(tmp_dir, suffix=".jpeg")
    git_file = create_random_file(tmp_dir / ".git")
    local_worker_file = create_random_file(tmp_dir / "local-worker")
    tmp_file = create_random_file(tmp_dir / TMP_SUBSTRAFL_PREFIX)

    paths = {
        csv_file,
        csv_subfolder_file,
        xls_file,
        xlsx_file,
        npy_file,
        png_file,
        jpg_file,
        jpeg_file,
        git_file,
        local_worker_file,
        tmp_file,
    }
    excluded_paths = DependencyPathManagement.get_excluded_paths(
        src=[tmp_dir], excluded=[], not_excluded=[], excluded_regex=[]
    )
    assert paths == excluded_paths


def test_get_excluded_paths_excluded_file_absolute(tmp_dir, create_random_file):
    path_excluded = create_random_file(tmp_dir)
    excluded_paths = DependencyPathManagement.get_excluded_paths(
        src=[tmp_dir], excluded=[path_excluded], not_excluded=[], excluded_regex=[]
    )
    assert {path_excluded} == excluded_paths


def test_get_excluded_paths_excluded_file_relative(tmp_dir, create_random_file):
    path_excluded = create_random_file(tmp_dir)
    path_excluded_relative = path_excluded.relative_to(tmp_dir.parent)
    excluded_paths = DependencyPathManagement.get_excluded_paths(
        src=[tmp_dir], excluded=[path_excluded_relative], not_excluded=[], excluded_regex=[]
    )
    assert {path_excluded} == excluded_paths


def test_get_excluded_paths_not_excluded_file_absolute(tmp_dir, create_random_file):
    csv_subfolder_file = create_random_file(tmp_dir / "subfolder", suffix=".csv")
    xls_file = create_random_file(tmp_dir, suffix=".xls")

    paths = {csv_subfolder_file}
    excluded_paths = DependencyPathManagement.get_excluded_paths(
        src=[tmp_dir], excluded=[], not_excluded=[xls_file], excluded_regex=[]
    )
    assert paths == excluded_paths


def test_get_excluded_paths_not_excluded_file_relative(tmp_dir, create_random_file):
    csv_subfolder_file = create_random_file(tmp_dir / "subfolder", suffix=".csv")
    xls_file = create_random_file(tmp_dir, suffix=".xls")
    xls_file_relative = xls_file.relative_to(tmp_dir.parent)
    paths = {csv_subfolder_file}
    excluded_paths = DependencyPathManagement.get_excluded_paths(
        src=[tmp_dir], excluded=[], not_excluded=[xls_file_relative], excluded_regex=[]
    )
    assert paths == excluded_paths


def test_get_excluded_paths_not_excluded_folder(tmp_dir, create_random_folder, create_random_file):
    subfolder = create_random_folder(tmp_dir)
    create_random_file(subfolder, suffix=".xls")
    paths = set()
    excluded_paths = DependencyPathManagement.get_excluded_paths(
        src=[tmp_dir], excluded=[], not_excluded=[subfolder], excluded_regex=[]
    )
    assert paths == excluded_paths


def test_copy_file_absolute(src_dir, dest_dir, create_random_file, create_random_folder):
    file = create_random_file(src_dir)
    subfolder = create_random_folder(src_dir)
    subfolder_file = create_random_file(subfolder)
    DependencyPathManagement.copy_paths(
        src=[src_dir], dest_dir=dest_dir, excluded=[], excluded_regex=[], not_excluded=[]
    )

    to_be_copied = [src_dir, file, subfolder, subfolder_file]
    to_be_copied_dest_dir = {dest_dir / f.relative_to(src_dir.parent) for f in to_be_copied}
    copied_files = set(dest_dir.rglob("*"))

    assert to_be_copied_dest_dir == copied_files


def test_copy_file_relative(src_dir, dest_dir, create_random_file, create_random_folder):
    file = create_random_file(src_dir)
    subfolder = create_random_folder(src_dir)
    subfolder_file = create_random_file(subfolder)

    local_path = Path()
    local_src_file = src_dir.relative_to(local_path.absolute())

    DependencyPathManagement.copy_paths(
        src=[local_src_file], dest_dir=dest_dir, excluded=[], excluded_regex=[], not_excluded=[]
    )

    to_be_copied = [src_dir, file, subfolder, subfolder_file]
    to_be_copied_dest_dir = {dest_dir / f.relative_to(src_dir.parent) for f in to_be_copied}
    copied_files = set(dest_dir.rglob("*"))
    assert to_be_copied_dest_dir == copied_files
