# Copyright 2018 Owkin, inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Packaging settings."""
from setuptools import find_packages, setup

setup(
    name="connectlibtestlibrary",
    version="0.0.1",
    description="testlib",
    packages=find_packages(exclude=["tests*"]),
    # Not compatible with substratools 0.8.0 because
    # that release is private and in the Docker container
    # it has access only to the public PyPi
    install_requires=[
        "numpy>=1.20.3",
        "cloudpickle>=1.6.0",
        "substratools>=0.9.0",
        "substra>=0.13.0",
        "wheel",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.4",
            "pytest-cov>=2.12.0",
            "pre-commit>=2.13.0",
            "types-PyYAML>=6.0.0",
        ],
    },
    python_requires=">=3.7",
)
