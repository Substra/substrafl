"""Packaging settings."""

import os
from codecs import open

from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))


with open(os.path.join(here, "README.md"), "r", "utf-8") as fp:
    readme = fp.read()

about: dict = dict()
with open(os.path.join(here, "substrafl", "__version__.py"), "r", "utf-8") as fp:
    exec(fp.read(), about)

setup(
    name="substrafl",
    version=about["__version__"],
    description="""A high-level federated learning Python library to run
     federated learning experiments at scale on a Substra network""",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://docs.substra.org/",
    author="Owkin, Inc.",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Utilities",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords=["substrafl"],
    packages=find_packages(exclude=["tests*", "benchmark*"]),
    # Not compatible with substratools 0.8.0 because
    # that release is private and in the Docker container
    # it has access only to the public PyPi
    install_requires=[
        "numpy>=1.24",
        "cloudpickle>=1.6.0",
        "substra~=0.51.0",
        "substratools~=0.21.2",
        "pydantic>=2.3.0,<3.0",
        "pip>=21.2",
        "tqdm",
        "wheel",
        "six",
        "packaging",
        "pip-tools",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.4",
            "pytest-cov>=2.12.0",
            "pytest-mock",
            "pre-commit>=2.13.0",
            "types-PyYAML>=6.0.0",
            "torch>=1.9.1,!=1.12.0",  # bug in 1.12.0 (https://github.com/pytorch/pytorch/pull/80345)
            "nbmake>=1.4.3",
            "docker",
        ],
    },
    python_requires=">=3.9",
)
