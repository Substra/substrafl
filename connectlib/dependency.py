from pathlib import Path, PosixPath
from typing import List, Optional

from pydantic import BaseModel, validator

from connectlib.exceptions import InvalidPathException


class Dependency(BaseModel):
    """Dependency pydantic class.

    Args:
        dependencies (Optional[List[str]]): Python packages installable form pypi.
        local_dependencies (Optional[List[Path]]): Local installable packages. The command
            `pip install -e .` will be executed in each of those folders hence a `setup.py` must be present in each
            folder.
        local_code (Optional[List[Path]]): Local relative imports used by your script. All files / folders must be at the same
            level than your script.
    """

    pypi_dependencies: Optional[List[str]] = list()
    local_dependencies: Optional[List[PosixPath]] = list()
    local_code: Optional[List[PosixPath]] = list()

    @validator("local_dependencies", "local_code")
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
            raise InvalidPathException(
                f"Couldn't resolve :{''.join(not_existing_paths)}\nPlease explicit the input path(s)."
            )

        return resolved_paths

    @validator("local_dependencies")
    def check_setup(cls, v):
        """Check the presence of a setup.py file in the provided paths."""
        not_installable = list()
        for dependency_path in v:
            if (
                dependency_path.is_dir()
                and not (dependency_path / "setup.py").is_file()
            ):
                not_installable.append(dependency_path)
            # The user can also give tar.gz archives
        if not_installable:
            raise InvalidPathException(
                f"Passed folder must be installable python package. "
                f"But no setup.py was found in :{', '.join([str(p) for p in not_installable])}"
            )
        return v
