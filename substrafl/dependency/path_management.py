import logging
import shutil
from abc import ABC
from abc import abstractmethod
from pathlib import Path
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
    "local-worker*",
    TMP_SUBSTRAFL_PREFIX + "*",
]


class BaseDependencyPathManagement(ABC):
    """Base class for different dependency paths management."""

    @classmethod
    def expand_regexes(cls, regexes: List[str], src: List[Path]) -> List[Path]:
        """Find all paths corresponding to a list of regex

        Args:
            regexes (List[str]): Regexes used to find strings.
            src (List[Path]): Path from where the files are copied from.

        Returns:
            List[Path]: All paths corresponding to any regex.
        """
        paths: List[Path] = []
        for input_path in src:
            for regex in regexes:
                paths.extend(input_path.rglob(regex))

        return paths

    @classmethod
    def expand_paths(cls, paths: List[Path]) -> Set[Path]:
        """List all files belonging to a list of path. If the path is a file, simply ad dthe file.
        If it is a folder, add all the file inside the folder.

        Args:
            paths (List[Path]): All paths to search into.

        Raises:
            ValueError: Provided a non-existing path.

        Returns:
            Set[Path]: A set of unique files found in the paths
        """
        unpacked_paths: Set[Path] = set()
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
        src: List[Path],
        excluded: List[Path],
        excluded_regex: List[str],
        not_excluded: List[Path],
    ) -> Set[Path]:
        """_summary_

        Args:
            src (List[Path]): Path from where the files are copied from.
            excluded (List[Path]): Paths to exclude from the `src` during the copy.
            excluded_regex (List[str]): Regex to find paths in `src` that will be excluded.
            not_excluded (List[Path]): Paths to remove from the paths found in `excluded`/`not_excluded`.

        Returns:
            Set[Path]: Set of excluded files, after expanding regexes and respecting `not_excluded`.
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
        dest_dir: Path,
        src: List[Path],
        not_excluded: Optional[List[Path]] = None,
        excluded: Optional[List[Path]] = None,
        excluded_regex: Optional[List[str]] = None,
    ) -> List[Path]:
        """Copy paths from `src` to `dest_dir` respecting exlclusion/non-exclusion paths provided through `excluded_regex`,
            `excluded`, `not_excluded`.

        Args:
            dest_dir (Path): Directory where the file are going to be copied into.
            src (List[Path]): Path to copy
            not_excluded (Optional[List[Path]], optional): Paths to remove from the paths found in
                `excluded`/`not_excluded`.
                Defaults to None.
            excluded (Optional[List[Path]], optional): Paths to exclude from the `src` during the copy.
                Defaults to None.
            excluded_regex (Optional[List[str]], optional): Regex to find paths in `src` that will be excluded.
                Always includes common data formats (see substrafl.dependency.EXCLUDED_PATHS_REGEX_DEFAULT)
                Defaults to None.

        Returns:
            List[Path]: Copied paths.
        """
        raise NotImplementedError("Abstract class do not provide implementation for this method")


class DependencyPathManagement(BaseDependencyPathManagement):
    @classmethod
    def copy_paths(
        cls,
        *,
        dest_dir: Path,
        src: List[Path],
        not_excluded: Optional[List[Path]] = None,
        excluded: Optional[List[Path]] = None,
        excluded_regex: Optional[List[str]] = None,
    ) -> List[Path]:
        """Copy paths from `src` to `dest_dir` respecting exlclusion/non-exclusion paths provided through `excluded_regex`,
            `excluded`, `not_excluded`

        Args:
            dest_dir (Path): Directory where the file are going to be copied into
            src (List[Path]): Path to copy
            not_excluded (Optional[List[Path]], optional): Paths to remove from the paths found in
                `excluded`/`not_excluded`.
                Defaults to None.
            excluded (Optional[List[Path]], optional): Paths to exclude from the `src` during the copy.
                Defaults to None.
            excluded_regex (Optional[List[str]], optional): Regex to find paths in `src` that will be excluded.
                Always includes common data formats (see substrafl.dependency.EXCLUDED_PATHS_REGEX_DEFAULT)
                Defaults to None.

        Raises:
            ValueError: `dest_dir` is a file.
            ValueError: One of the paths in `src` does not exist.

        Returns:
            List[Path]: Copied paths.
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
            if input_path.is_file() and input_path not in expanded_excluded:
                output_path = dest_dir / input_path.name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(input_path.absolute(), output_path)
            elif input_path.is_dir():
                for file in input_path.rglob("*"):
                    if file.is_file() and file not in expanded_excluded:
                        output_path = dest_dir / file.relative_to(input_path.parent)
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(file.absolute(), output_path)
            else:
                raise ValueError(f"Try to parse {input_path} that does not exist.")
            output_files.append(input_path)

        return list(output_files)
