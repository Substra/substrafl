import logging
import shutil
from pathlib import Path
from typing import List
from typing import Optional
from typing import Set

from substrafl.dependency.constants import EXCLUDED_PATHS_REGEX_DEFAULT

logger = logging.getLogger(__name__)


def expand_regexes(regexes: List[str], src: List[Path]) -> List[Path]:
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
            excluded_paths = list(input_path.rglob(regex))
            paths.extend(excluded_paths)

            excluded_paths_count = len(excluded_paths)
            if excluded_paths_count > 0:
                logger.warning(f"Regex {regex} excludes {excluded_paths_count} file(s)")
                if logger.isEnabledFor(logging.DEBUG):
                    formatted_paths = "\n\t".join(str(p) for p in excluded_paths)
                    logger.debug(f"Regex {regex} excludes: {formatted_paths}")

    return paths


def expand_paths(paths: List[Path]) -> Set[Path]:
    """List all files belonging to a list of path. If the path is a file, simply add the file.
    If it is a folder, add all the files inside the folder.

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
            unpacked_paths.add(path.resolve())
        elif path.is_dir():
            unpacked_paths.update(p.resolve() for p in path.rglob("*") if p.is_file())
        else:
            raise ValueError(f"Try to parse {path} that is neither a file or a dir.")

    return unpacked_paths


def get_excluded_paths(
    *,
    src: List[Path],
    excluded: List[Path],
    excluded_regex: List[str],
    force_included: List[Path],
) -> Set[Path]:
    """Get all paths to exclude, after expanding the regexes and respectful of non-exclusion list

    Args:
        src (List[Path]): Path from where the files are copied from.
        excluded (List[Path]): Paths to exclude from the `src` during the copy.
        excluded_regex (List[str]): Regex to find paths in `src` that will be excluded.
        force_included (List[Path]): Paths to remove from the paths found in `excluded`/`force_included`.

    Returns:
        Set[Path]: Set of excluded files, after expanding regexes and respecting `force_included`.
    """
    expanded_excluded_regex = expand_regexes(excluded_regex + EXCLUDED_PATHS_REGEX_DEFAULT, src)
    expanded_excluded = expand_paths(excluded + expanded_excluded_regex)
    expanded_force_included = expand_paths(force_included)
    return expanded_excluded - expanded_force_included


def copy_paths(
    *,
    dest_dir: Path,
    src: List[Path],
    force_included: Optional[List[Path]] = None,
    excluded: Optional[List[Path]] = None,
    excluded_regex: Optional[List[str]] = None,
) -> List[Path]:
    """Copy paths from `src` to `dest_dir` respecting exclusion/non-exclusion paths provided through `excluded_regex`,
        `excluded`, `force_included`

    Args:
        dest_dir (Path): Directory where the file are going to be copied into
        src (List[Path]): Path to copy
        force_included (Optional[List[Path]], optional): Paths to remove from the paths found in
            `excluded`/`force_included`.
            Defaults to None.
        excluded (Optional[List[Path]], optional): Paths to exclude from the `src` during the copy.
            Defaults to None.
        excluded_regex (Optional[List[str]], optional): Regex to find paths in `src` that will be excluded.
            Always includes common data formats (see substrafl.dependency.EXCLUDED_PATHS_REGEX_DEFAULT)
            Defaults to None (only substrafl.dependency.EXCLUDED_PATHS_REGEX_DEFAULT are excluded).

    Raises:
        ValueError: `dest_dir` is a file.
        ValueError: One of the paths in `src` does not exist.

    Returns:
        List[Path]: Copied paths.
    """
    if dest_dir.is_file():
        raise ValueError(f"{dest_dir=} is a file. Cannot copy in a file.")

    if not force_included:
        force_included = []

    if not excluded:
        excluded = []

    if not excluded_regex:
        excluded_regex = []

    expanded_excluded = get_excluded_paths(
        src=src, excluded=excluded, excluded_regex=excluded_regex, force_included=force_included
    )
    output_files = []
    dest_dir.mkdir(parents=True, exist_ok=True)
    for input_path in src:
        input_resolved_path = input_path.resolve()
        if input_path.is_file() and input_path not in expanded_excluded:
            output_path = dest_dir / input_resolved_path.name
            shutil.copy(input_path, output_path)
        elif input_path.is_dir():
            output_path = dest_dir / input_resolved_path.name
            shutil.copytree(
                input_path, output_path, dirs_exist_ok=True, ignore=_ignore_files(expanded_excluded, dest_dir)
            )
        else:
            raise ValueError(f"Try to parse {input_path} that does not exist.")
        output_files.append(Path(input_path.name))

    return list(output_files)


def _ignore_files(expanded_excluded, dest_dir):
    def _ignore_files(path: str, names: List[str]) -> Set[str]:
        p = Path(path).resolve()

        # Replicate is_relative_to, added in Python 3.9
        try:
            p.relative_to(dest_dir.resolve())
        except ValueError:
            return set(name for name in names if p / name in expanded_excluded)
        else:
            return set(names)

    return _ignore_files
