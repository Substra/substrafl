from substrafl.constants import TMP_SUBSTRAFL_PREFIX

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
