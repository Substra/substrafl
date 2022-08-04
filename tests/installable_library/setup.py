"""Packaging settings."""
from setuptools import find_packages
from setuptools import setup

setup(
    name="substrafltestlibrary",
    version="0.0.1",
    description="testlib",
    packages=find_packages(),
    python_requires=">=3.7",
)
