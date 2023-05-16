import logging
import shutil
from pathlib import Path
from typing import List
from typing import Optional
from typing import Set

logger = logging.getLogger(__name__)

TMP_SUBSTRAFL_PREFIX = "tmp_substrafl"
# Common paths regex we want to exclude, partially based on
# https://github.com/github/gitignore/blob/main/Python.gitignore
EXCLUDED_PATHS_REGEX_DEFAULT = [
    # Byte-compiled / optimized / DLL files
    "__pycache__/*",
    "*.py[cod]",
    "*$py.class",
    "htmlcov/*",
    ".tox/*",
    ".nox/*",
    ".coverage",
    ".coverage.*",
    ".cache",
    "nosetests.xml",
    "coverage.xml",
    "*.cover",
    "*.py,cover",
    ".hypothesis/*",
    ".pytest_cache/*",
    "cover/*",
    # Jupyter Notebook
    ".ipynb_checkpoints",
    # SageMath parsed files
    "*.sage.py",
    # Environments
    ".env",
    ".venv",
    "env/*",
    "venv/*",
    "ENV/*",
    "env.bak/*",
    "venv.bak/*",
    # Spyder project settings
    ".spyderproject",
    ".spyproject",
    # Rope project settings
    ".ropeproject",
    # mypy
    ".mypy_cache/*",
    ".dmypy.json",
    "dmypy.json",
    # Pyre type checker
    ".pyre/*",
    # pytype static type analyzer
    ".pytype/*",
    # Cython debug symbols
    "cython_debug/*",
    # PyCharm
    #  JetBrains specific template is maintained in a separate JetBrains.gitignore that can
    #  be found at https://github.com/github/gitignore/blob/main/Global/JetBrains.gitignore
    #  and can be added to the global gitignore or merged into this file.  For a more nuclear
    #  option (not recommended) you can uncomment the following to ignore the entire idea folder.
    ".idea/*",
    # VS Code
    ".vscode/*",
    # Custom
    # Common data extensions
    "*.csv",
    "*.hdf",
    "*.jpg",
    "*.jpeg",
    "*.npy",
    "*.png",
    "*.pqt",
    "*.parquet",
    "*.xls",
    "*.xlsx",
    # Common folders
    ".git/*",
    ".github/*",
    # Substra internal files
    "local-worker*",
    TMP_SUBSTRAFL_PREFIX + "*",
]


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
                logger.warn(f"Regex {regex} excludes {excluded_paths_count} file(s)")
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
            unpacked_paths.add(path.absolute())
        elif path.is_dir():
            unpacked_paths.update(p.absolute() for p in path.rglob("*") if p.is_file())
        else:
            raise ValueError(f"Try to parse {path} that is neither a file or a dir.")

    return unpacked_paths


def get_excluded_paths(
    *,
    src: List[Path],
    excluded: List[Path],
    excluded_regex: List[str],
    not_excluded: List[Path],
) -> Set[Path]:
    """Get all paths to exclude, after expanding the regexes and respectful of non-exclusion list

    Args:
        src (List[Path]): Path from where the files are copied from.
        excluded (List[Path]): Paths to exclude from the `src` during the copy.
        excluded_regex (List[str]): Regex to find paths in `src` that will be excluded.
        not_excluded (List[Path]): Paths to remove from the paths found in `excluded`/`not_excluded`.

    Returns:
        Set[Path]: Set of excluded files, after expanding regexes and respecting `not_excluded`.
    """
    expanded_excluded_regex = expand_regexes(excluded_regex + EXCLUDED_PATHS_REGEX_DEFAULT, src)
    expanded_excluded = expand_paths(excluded + expanded_excluded_regex)
    expanded_not_excluded = expand_paths(not_excluded)
    return expanded_excluded - expanded_not_excluded


def copy_paths(
    *,
    dest_dir: Path,
    src: List[Path],
    not_excluded: Optional[List[Path]] = None,
    excluded: Optional[List[Path]] = None,
    excluded_regex: Optional[List[str]] = None,
) -> List[Path]:
    """Copy paths from `src` to `dest_dir` respecting exclusion/non-exclusion paths provided through `excluded_regex`,
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

    expanded_excluded = get_excluded_paths(
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
