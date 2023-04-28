import logging
import shutil
from abc import ABC
from abc import abstractmethod
from pathlib import PosixPath
from typing import List
from typing import Optional
from typing import Set

logger = logging.getLogger(__name__)

TMP_SUBSTRAFL_PREFIX = "tmp_substrafl"
EXCLUDED_PATHS_REGEX_DEFAULT = [
    # Common data extensions
    "*.csv",
    "*.xls",
    "*.xlsx",
    # Python data extensiosn
    "*.npy",
    # Common image extensions
    "*.png",
    "*.jpg",
    "*.jpeg",
    # Common folders
    ".git/*",
    # Others
    "local-worker/*",
    TMP_SUBSTRAFL_PREFIX + "/*",
]


class BaseDependencyPathManagement(ABC):
    """Base class for different dependency paths management."""

    @classmethod
    def expand_regexes(cls, regexes: List[str], src: List[PosixPath]) -> List[PosixPath]:
        """Find all paths corresponding to a list of regex

        Args:
            regexes (List[str]): Regexes used to find strings.
            src (List[PosixPath]): Path from where the files are copied from.

        Returns:
            List[PosixPath]: All paths corresponding to any regex.
        """
        paths: List[PosixPath] = []
        for input_path in src:
            for regex in regexes:
                paths.extend(input_path.rglob(regex))

        return paths

    @classmethod
    def expand_paths(cls, paths: List[PosixPath]) -> Set[PosixPath]:
        """List all files belonging to a list of path. If the path is a file, simply ad dthe file.
        If it is a folder, add all the file inside the folder.

        Args:
            paths (List[PosixPath]): All paths to search into.

        Raises:
            ValueError: Provided a non-existing path.

        Returns:
            Set[PosixPath]: A set of unique files found in the paths
        """
        unpacked_paths: Set[PosixPath] = set()
        for path in paths:
            if path.is_file():
                unpacked_paths.add(path.absolute())
            elif path.is_dir():
                unpacked_paths.update(p.absolute() for p in path.rglob("*") if p.is_file())
            else:
                raise ValueError(f"Try to parse {path} that is neither a file or a dir.")

        return unpacked_paths

    @classmethod
    def get_excluded_paths(
        cls,
        *,
        src: List[PosixPath],
        excluded: List[PosixPath],
        excluded_regex: List[str],
        not_excluded: List[PosixPath],
    ) -> Set[PosixPath]:
        """_summary_

        Args:
            src (List[PosixPath]): Path from where the files are copied from.
            excluded (List[PosixPath]): Paths to exclude from the `src` during the copy.
            excluded_regex (List[str]): Regex to find paths in `src` that will be excluded.
            not_excluded (List[PosixPath]): Paths to remove from the paths found in `excluded`/`not_excluded`.

        Returns:
            Set[PosixPath]: Set of excluded files, after expanding regexes and respecting `not_excluded`.
        """
        expanded_excluded_regex = cls.expand_regexes(excluded_regex + EXCLUDED_PATHS_REGEX_DEFAULT, src)
        expanded_excluded = cls.expand_paths(excluded + expanded_excluded_regex)
        expanded_not_excluded = cls.expand_paths(not_excluded)
        return expanded_excluded - expanded_not_excluded

    @classmethod
    @abstractmethod
    def copy_paths(
        cls,
        *,
        dest_dir: PosixPath,
        src: List[PosixPath],
        not_excluded: Optional[List[PosixPath]] = None,
        excluded: Optional[List[PosixPath]] = None,
        excluded_regex: Optional[List[str]] = None,
    ) -> List[PosixPath]:
        """Copy paths from `src` to `dest_dir` respecting exlclusion/non-exclusion paths provided through `excluded_regex`,
            `excluded`, `not_excluded`.

        Args:
            dest_dir (PosixPath): Directory where the file are going to be copied into.
            src (List[PosixPath]): Path to copy
            not_excluded (Optional[List[PosixPath]], optional): Paths to remove from the paths found in
                `excluded`/`not_excluded`.
                Defaults to None.
            excluded (Optional[List[PosixPath]], optional): Paths to exclude from the `src` during the copy.
                Defaults to None.
            excluded_regex (Optional[List[str]], optional): Regex to find paths in `src` that will be excluded.
                Always includes common data formats (see substrafl.dependency.EXCLUDED_PATHS_REGEX_DEFAULT)
                Defaults to None.

        Returns:
            List[PosixPath]: Copied paths.
        """
        raise NotImplementedError("Abstract class do not provide implementation for this method")


class DependencyPathManagement(BaseDependencyPathManagement):
    @classmethod
    def copy_paths(
        cls,
        *,
        dest_dir: PosixPath,
        src: List[PosixPath],
        not_excluded: Optional[List[PosixPath]] = None,
        excluded: Optional[List[PosixPath]] = None,
        excluded_regex: Optional[List[str]] = None,
    ) -> List[PosixPath]:
        """Copy paths from `src` to `dest_dir` respecting exlclusion/non-exclusion paths provided through `excluded_regex`,
            `excluded`, `not_excluded`

        Args:
            dest_dir (PosixPath): Directory where the file are going to be copied into
            src (List[PosixPath]): Path to copy
            not_excluded (Optional[List[PosixPath]], optional): Paths to remove from the paths found in
                `excluded`/`not_excluded`.
                Defaults to None.
            excluded (Optional[List[PosixPath]], optional): Paths to exclude from the `src` during the copy.
                Defaults to None.
            excluded_regex (Optional[List[str]], optional): Regex to find paths in `src` that will be excluded.
                Always includes common data formats (see substrafl.dependency.EXCLUDED_PATHS_REGEX_DEFAULT)
                Defaults to None.

        Raises:
            ValueError: `dest_dir` is a file.
            ValueError: One of the paths in `src` does not exist.

        Returns:
            List[PosixPath]: Copied paths.
        """
        if dest_dir.is_file():
            raise ValueError(f"{dest_dir=} is a file. Cannot copy in a file.")

        if not not_excluded:
            not_excluded = []

        if not excluded:
            excluded = []

        if not excluded_regex:
            excluded_regex = []

        expanded_excluded = cls.get_excluded_paths(
            src=src, excluded=excluded, excluded_regex=excluded_regex, not_excluded=not_excluded
        )

        output_files = []
        for input_path in src:
            absolute_path = input_path.absolute()
            if absolute_path.is_file() and absolute_path not in expanded_excluded:
                output_path = dest_dir / input_path.name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(absolute_path, output_path)
            elif absolute_path.is_dir():
                for file in absolute_path.rglob("*"):
                    if file.is_file() and file not in expanded_excluded:
                        output_path = dest_dir / file.relative_to(absolute_path.parent)
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(file, output_path)
            else:
                raise ValueError(f"Try to parse {input_path} that does not exist.")
            output_files.append(input_path)

        return list(output_files)
