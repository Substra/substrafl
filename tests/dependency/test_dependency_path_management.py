import os
import random
import string
from pathlib import Path
from typing import Optional

import pytest

from substrafl.constants import TMP_SUBSTRAFL_PREFIX
from substrafl.dependency import path_management


@pytest.fixture
def src_dir(tmp_path):
    return tmp_path


@pytest.fixture
def dest_dir(tmp_path):
    return tmp_path


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


def _get_relative_path(path: Path, src_dir: Path) -> Path:
    os.chdir(src_dir)
    return path.relative_to(src_dir.resolve())


def test_expand_regex_file(tmp_path, create_random_file):
    path_1 = create_random_file(tmp_path)
    path_2 = create_random_file(tmp_path)

    subfolder = tmp_path / "subfolder"
    path_3 = create_random_file(subfolder)
    path_4 = create_random_file(subfolder)
    paths = path_management.expand_regexes("*", [tmp_path])
    assert set(paths) == {subfolder, path_1, path_2, path_3, path_4}


def test_expand_regex_folder(tmp_path, create_random_folder, create_random_file):
    folder_path = create_random_folder(tmp_path)
    path = create_random_file(folder_path)

    paths = path_management.expand_regexes("*", [folder_path])
    assert paths == [path]


def test_expand_paths_file(tmp_path, create_random_file):
    path = create_random_file(tmp_path)
    paths = path_management.expand_paths([tmp_path])
    assert paths == {path}


def test_expand_paths_folders(tmp_path, create_random_folder, create_random_file):
    folder_path = create_random_folder(tmp_path)
    paths = {create_random_file(folder_path) for _ in range(10)}
    found_paths = path_management.expand_paths([tmp_path])
    assert paths == found_paths


def test_get_excluded_paths_default(tmp_path, create_random_file):
    csv_file = create_random_file(tmp_path, suffix=".csv")
    # Try subfolder
    csv_subfolder_file = create_random_file(tmp_path / "subfolder", suffix=".csv")
    xls_file = create_random_file(tmp_path, suffix=".xls")
    xlsx_file = create_random_file(tmp_path, suffix=".xlsx")
    npy_file = create_random_file(tmp_path, suffix=".npy")
    png_file = create_random_file(tmp_path, suffix=".png")
    jpg_file = create_random_file(tmp_path, suffix=".jpg")
    jpeg_file = create_random_file(tmp_path, suffix=".jpeg")
    git_file = create_random_file(tmp_path / ".git")
    local_worker_file = create_random_file(tmp_path / "local-worker")
    tmp_file = create_random_file(tmp_path / TMP_SUBSTRAFL_PREFIX)

    # Not excluded file
    create_random_file(tmp_path)

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
    excluded_paths = path_management.get_excluded_paths(
        src=[tmp_path], excluded=[], force_included=[], excluded_regex=[]
    )
    assert paths == excluded_paths


def test_get_excluded_paths_excluded_file_absolute(tmp_path, create_random_file):
    path_excluded = create_random_file(tmp_path)
    excluded_paths = path_management.get_excluded_paths(
        src=[tmp_path], excluded=[path_excluded], force_included=[], excluded_regex=[]
    )
    assert {path_excluded} == excluded_paths


def test_get_excluded_paths_excluded_file_relative(tmp_path, create_random_file):
    path_excluded = create_random_file(tmp_path)
    path_excluded_relative = _get_relative_path(path_excluded, tmp_path)
    excluded_paths = path_management.get_excluded_paths(
        src=[tmp_path], excluded=[path_excluded_relative], force_included=[], excluded_regex=[]
    )
    assert {path_excluded} == excluded_paths


def test_get_excluded_paths_force_included_file_absolute(tmp_path, create_random_file):
    csv_subfolder_file = create_random_file(tmp_path / "subfolder", suffix=".csv")
    xls_file = create_random_file(tmp_path, suffix=".xls")

    paths = {csv_subfolder_file}
    excluded_paths = path_management.get_excluded_paths(
        src=[tmp_path], excluded=[], force_included=[xls_file], excluded_regex=[]
    )
    assert paths == excluded_paths


def test_get_excluded_paths_force_included_file_relative(tmp_path, create_random_file):
    csv_subfolder_file = create_random_file(tmp_path / "subfolder", suffix=".csv")
    xls_file = create_random_file(tmp_path, suffix=".xls")
    xls_file_relative = _get_relative_path(xls_file, tmp_path)
    paths = {csv_subfolder_file}
    excluded_paths = path_management.get_excluded_paths(
        src=[tmp_path], excluded=[], force_included=[xls_file_relative], excluded_regex=[]
    )
    assert paths == excluded_paths


def test_get_excluded_paths_force_included_folder(tmp_path, create_random_folder, create_random_file):
    subfolder = create_random_folder(tmp_path)
    create_random_file(subfolder, suffix=".xls")
    paths = set()
    excluded_paths = path_management.get_excluded_paths(
        src=[tmp_path], excluded=[], force_included=[subfolder], excluded_regex=[]
    )
    assert paths == excluded_paths


def test_copy_file_absolute(src_dir, dest_dir, create_random_file, create_random_folder):
    file = create_random_file(src_dir)
    subfolder = create_random_folder(src_dir)
    subfolder_file = create_random_file(subfolder)
    path_management.copy_paths(src=[src_dir], dest_dir=dest_dir, excluded=[], excluded_regex=[], force_included=[])

    to_be_copied = [src_dir / src_dir.name, file, subfolder, subfolder_file]
    to_be_copied_dest_dir = {dest_dir / f.relative_to(src_dir) for f in to_be_copied}
    copied_files = set(dest_dir.rglob("*"))

    assert to_be_copied_dest_dir == copied_files


def test_copy_file_relative(src_dir, dest_dir, create_random_file, create_random_folder):
    file = create_random_file(src_dir)
    subfolder = create_random_folder(src_dir)
    subfolder_file = create_random_file(subfolder)

    local_src_file = _get_relative_path(src_dir, src_dir / "..")
    path_management.copy_paths(
        src=[local_src_file], dest_dir=dest_dir, excluded=[], excluded_regex=[], force_included=[]
    )

    to_be_copied = [src_dir / src_dir.name, file, subfolder, subfolder_file]
    to_be_copied_dest_dir = {dest_dir / f.relative_to(src_dir) for f in to_be_copied}
    copied_files = set(dest_dir.rglob("*"))
    assert to_be_copied_dest_dir == copied_files


@pytest.mark.slow
def test_copy_file_relative_parent(src_dir, dest_dir, create_random_file, create_random_folder):
    parent_path = Path("..")

    copied_paths = path_management.copy_paths(
        src=[parent_path], dest_dir=dest_dir, excluded=[], excluded_regex=[], force_included=[]
    )

    to_be_copied = [parent_path]
    assert to_be_copied == copied_paths


@pytest.mark.slow
def test_copy_file_absolute_parent(src_dir, dest_dir, create_random_file, create_random_folder):
    parent_path = Path("..").resolve()

    copied_paths = path_management.copy_paths(
        src=[parent_path], dest_dir=dest_dir, excluded=[], excluded_regex=[], force_included=[]
    )

    to_be_copied = [Path(parent_path.name)]
    assert to_be_copied == copied_paths
