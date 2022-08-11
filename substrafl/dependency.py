from pathlib import Path
from pathlib import PosixPath
from typing import List

from pydantic import BaseModel
from pydantic import Field
from pydantic import validator

from substrafl.exceptions import InvalidPathError


class Dependency(BaseModel):
    """Dependency pydantic class.

    Args:
        editable_mode (bool): If set to False, substra, substrafl and substratools used in the
            Dockerfiles submitted to Substra platform will be taken from owkin pypi. Be sure to have
            access to owkin's pypi and configure accordingly your pip.conf.
            If set to True, it will be the one installed in editable mode from your python environment.
            Defaults to False.
        dependencies (List[str]): Python packages installable from pypi.
        local_dependencies (List[pathlib.Path]): Local installable packages. The command
            `pip install -e .` will be executed in each of those folders hence a `setup.py` must be present in each
            folder.
        local_code (List[pathlib.Path]): Local relative imports used by your script. All files / folders must be
            at the same level than your script.
    """

    editable_mode: bool = False
    pypi_dependencies: List[str] = Field(default_factory=list)
    local_dependencies: List[PosixPath] = Field(default_factory=list)
    local_code: List[PosixPath] = Field(default_factory=list)

    @validator("local_dependencies", "local_code")
    def resolve_path(cls, v):  # noqa: N805
        """Resolve list of local code paths and check if they exist."""
        not_existing_paths = list()
        resolved_paths = list()
        for path in v:
            if not Path(path).exists():
                not_existing_paths += [f"\n\t{Path(path)} AS {Path(path).resolve()}"]
            else:
                resolved_paths.append(Path(path).resolve())

        if not_existing_paths:
            raise InvalidPathError(
                f"Couldn't resolve :{''.join(not_existing_paths)}\nPlease explicit the input path(s)."
            )

        return resolved_paths

    @validator("local_dependencies")
    def check_setup(cls, v):  # noqa: N805
        """Check the presence of a setup.py file in the provided paths."""
        not_installable = list()
        for dependency_path in v:
            installable_dir = (dependency_path / "setup.py").is_file() or (dependency_path / "pyproject.toml").is_file()
            if dependency_path.is_dir() and not installable_dir:
                not_installable.append(dependency_path)
            # The user can also give tar.gz archives
        if not_installable:
            raise InvalidPathError(
                f"Passed folder must be installable python package. "
                f"But neither setup.py or pyproject.toml was found in :{', '.join([str(p) for p in not_installable])}"
            )
        return v
