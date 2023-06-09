import pytest

SETUP_CONTENT = """from setuptools import setup, find_packages

setup(
    name='mymodule',
    version='1.0.2',
    author='Author Name',
    description='Description of my package',
    packages=find_packages(),
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 3.5.1'],
)"""


@pytest.fixture
def local_installable_module():
    def _local_installable_module(root_dir):
        module_root = root_dir / "my_module"
        module_root.mkdir()
        setup_file = module_root / "setup.py"
        setup_file.write_text(SETUP_CONTENT)
        return module_root

    return _local_installable_module
