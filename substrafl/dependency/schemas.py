import logging
import os
import shutil
import tempfile
from distutils import util
from pathlib import Path
from typing import List
from typing import Optional

import substra
import substratools
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

import substrafl
from substrafl import exceptions
from substrafl.constants import SUBSTRAFL_FOLDER
from substrafl.dependency import manage_dependencies
from substrafl.dependency import path_management

logger = logging.getLogger(__name__)


class Dependency(BaseModel):
    """Dependency pydantic class.


    .. note:: If you are using your current package as a local dependencies, be aware that folders named
        **local-worker** or with **tmp_substrafl** as prefix are ignored during the installation.

    Args:
        editable_mode (bool): If set to False, substra, substrafl and substratools used in the
            Dockerfiles submitted to Substra platform will be taken from pypi.
            If set to True, it will be the one installed in editable mode from your python environment.
            Defaults to False.
        compile (bool): If set to True, dependencies will be resolved only once (using pip compile) and
            the set of compatible versions for each dependencies (and indirect dependencies) will be reused.
            Default to False.
        pypi_dependencies (List[str]): Python packages installable from PyPI.
        local_installable_dependencies (List[pathlib.Path]): Local installable packages.
            Each one can either be a wheel or a local folder. If it's a local folder, the command
            `python -m pip wheel .` will be run, so each folder needs to be a valid Python module (containing a valid
            `setup.py` or `pyproject.toml`). See the documentation of pip wheel for more details.
        local_code (List[pathlib.Path]): Local relative imports used by your script. All files / folders will be
            pasted to the level of the running script.
        excluded_paths (List[pathlib.Path]): Local paths excluded from `local_installable_dependencies` / `local_code`.
            Default to [].
        excluded_regex (List[pathlib.Path]): Regex used to exclude files from `local_installable_dependencies` /
            `local_code`.
            Default to [].
            Always excludes common data formats (see
            `substrafl.dependency.path_management.EXCLUDED_PATHS_REGEX_DEFAULT`).
        force_included_paths (List[pathlib.Path]): Force include files otherwise excluded by `excluded_paths`
            and `excluded_regex`
            Default to []
    """

    editable_mode: bool = False
    compile: bool = False
    pypi_dependencies: List[str] = Field(default_factory=list)
    local_installable_dependencies: List[Path] = Field(default_factory=list)
    local_code: List[Path] = Field(default_factory=list)
    excluded_paths: List[Path] = Field(default_factory=list)
    excluded_regex: List[str] = Field(default_factory=list)
    force_included_paths: List[Path] = Field(default_factory=list)
    _wheels: List[Path] = []
    _local_paths: List[Path] = []
    _cache_directory: Optional[Path] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, *args, **kwargs):
        """Dependencies are computed at object initialization.
        The computation is stored in a cache directory, that will be cleaned up
        at the object deletion.
        """
        super().__init__(*args, **kwargs)
        self._compute_in_cache_directory()

    def __del__(self):
        """Delete the cache directory."""
        self._delete_cache_directory()

    @field_validator("local_installable_dependencies", "local_code")
    @classmethod
    def resolve_path(cls, v):
        """Resolve list of local code paths and check if they exist."""
        not_existing_paths = list()
        resolved_paths = list()
        for path in v:
            if not Path(path).exists():
                not_existing_paths += [f"\n\t{Path(path)} AS {Path(path).resolve()}"]
            else:
                resolved_paths.append(Path(path).resolve())

        if not_existing_paths:
            raise exceptions.InvalidPathError(
                f"Couldn't resolve :{''.join(not_existing_paths)}\nPlease explicit the input path(s)."
            )

        return resolved_paths

    @field_validator("local_installable_dependencies")
    @classmethod
    def check_setup(cls, v):
        """Check the presence of a setup.py file or a pyproject.toml in the provided paths."""
        not_installable = list()
        for dependency_path in v:
            installable_dir = (dependency_path / "setup.py").is_file() or (dependency_path / "pyproject.toml").is_file()
            if dependency_path.is_dir() and not installable_dir:
                not_installable.append(dependency_path)
            # The user can also give tar.gz archives
        if not_installable:
            raise exceptions.InvalidPathError(
                f"Passed directory must be installable python package. "
                f"But neither setup.py or pyproject.toml was found in :{', '.join([str(p) for p in not_installable])}"
            )
        return v

    def copy_dependencies_local_package(self, *, dest_dir: Path) -> List[Path]:
        return path_management.copy_paths(
            dest_dir=dest_dir,
            src=self.local_installable_dependencies,
            force_included=self.force_included_paths,
            excluded=self.excluded_paths,
            excluded_regex=self.excluded_regex,
        )

    def copy_dependencies_local_code(self, *, dest_dir: Path) -> List[Path]:
        return path_management.copy_paths(
            dest_dir=dest_dir,
            src=self.local_code,
            force_included=self.force_included_paths,
            excluded=self.excluded_paths,
            excluded_regex=self.excluded_regex,
        )

    @property
    def cache_directory(self) -> Path:
        """Getter method to retrieve the path to the cache directory where the dependencies are computed.

        Raises:
            exceptions.DependencyCacheNotFoundError: If no cache directory is found, raise an exception.

        Returns:
            Path: return the path to the cache directory of the dependency.
        """
        if self._cache_directory is not None:
            return self._cache_directory
        else:
            raise exceptions.DependencyCacheNotFoundError(
                "No cache directory found for the dependencies. Have you computed the dependencies?"
            )

    def _compute(self, dest_dir: Path) -> None:
        """Build the different wheels, copy the local code and compile or write a requirements.txt regarding the
        given dependencies.

        Dependencies computation consists in:

        - building the different wheels
        - copying the local code
        - compiling or writing a requirements.txt regarding the given dependencies

        The target tree structure is described bellow.

        .. code-block::

            dest_dir
            ├── local_code.py
            └── substrafl_internal
                ├── dist
                │   ├── substra-0.44.0-py3-none-any.whl
                │   ├── substrafl-0.36.0-py3-none-any.whl
                │   └── substratools-0.20.0-py3-none-any.whl
                ├── local_dependencies
                │   └── local-module-1.6.1-py3-none-any.whl
                ├── requirements.in  # only if compile set to True
                └── requirements.txt

        Args:
            dest_dir (Path): directory where to compute the dependencies
        """

        # Substra libraries management
        if self.editable_mode or util.strtobool(os.environ.get("SUBSTRA_FORCE_EDITABLE_MODE", "False")):
            substra_wheel_dir = dest_dir / SUBSTRAFL_FOLDER / "dist"
            substra_wheels = manage_dependencies.local_lib_wheels(
                lib_modules=[
                    substrafl,
                    substra,
                    substratools,
                ],  # We reinstall substratools in editable mode to overwrite the installed version
                dest_dir=substra_wheel_dir,
            )
            self._wheels += [substra_wheel_dir.relative_to(dest_dir) / wheel_name for wheel_name in substra_wheels]
        else:
            self.pypi_dependencies += manage_dependencies.get_pypi_dependencies_versions(
                lib_modules=[substrafl],
            )

        # Local dependencies management
        local_dep_dir = dest_dir / SUBSTRAFL_FOLDER / "local_dependencies"
        dependencies_paths = self.copy_dependencies_local_package(dest_dir=local_dep_dir)

        for dependency in dependencies_paths:
            if dependency.__str__().endswith(".whl"):
                self._wheels.append(local_dep_dir.relative_to(dest_dir) / str(dependency))
            else:
                wheel_name = manage_dependencies.build_user_dependency_wheel(
                    Path(dependency),
                    dest_dir=local_dep_dir,
                )
                self._wheels.append(local_dep_dir.relative_to(dest_dir) / wheel_name)

        # Local code management
        self._local_paths = self.copy_dependencies_local_code(dest_dir=dest_dir)

        # requirement.txt edition
        if self.compile:
            # pip-compile the requirements.in into a requirements.txt
            manage_dependencies.compile_requirements(self.pypi_dependencies + self._wheels, dest_dir=dest_dir)
        else:
            manage_dependencies.write_requirements(self.pypi_dependencies + self._wheels, dest_dir=dest_dir)

    def _compute_in_cache_directory(self) -> None:
        """Compute the dependencies in an cache directory.
        Use ``_delete_cache_directory`` method to delete it.
        Use ``_copy_cache_directory`` method to copy it in an other directory.
        """
        if self._cache_directory is None:
            self._cache_directory = Path(tempfile.mkdtemp(dir=None))
            self._compute(self._cache_directory)
        else:
            logger.warning(
                f"Dependencies already computed in cache at {self._cache_directory}. Run `_delete_cache_directory` to "
                "delete it."
            )

    def _delete_cache_directory(self) -> None:
        """Delete the cache directory if it exists. A warning message is send to the logger
        if the cache directory does not exist.
        """
        if self._cache_directory is not None:
            try:  # We don't want to raise in __del__.
                shutil.rmtree(self._cache_directory)  # delete directory
            except FileNotFoundError:
                logger.warning("Not cache directory found for the dependencies when trying to clean it.")
            self._cache_directory = None
        else:
            logger.warning("Not cache directory found for the dependencies when trying to clean it.")
