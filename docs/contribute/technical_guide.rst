Technical Guide
===============

Merge requests
--------------

Description
^^^^^^^^^^^

- Fill in the required template:

    -  if there in a associated issue, a reference to it (only need to write ``#{issue_number}`` in GitHub, e.g. ``#85``)
    -  if there is no issue, a description of the problem it solves
    -  a technical description of the implementation

    Adding ``Closes #{issue_number}`` in the merge request description to automatically close the
    issue once the request is merged into master.

- Be sure that your pull request contains tests that cover the changed or added code.
- If your changes warrant a documentation change, the pull request must also update the documentation.

Commits - format
^^^^^^^^^^^^^^^^

Commits on the master branch and all branches where several people may
work at the same time follow the **conventional commit** convention. If
someone works on his own branch, then he is expected to **squash** his
commits into one commit in the conventional format at the merge.

::

   type(optional scope): one-liner description (less than 100 characters)

   optional body

   optional footer

The main possible types are:

-  ``feat`` for feature
-  ``test`` for tests
-  ``fix`` for bug fixes
-  ``doc`` for documentation
-  and
   `more <https://github.com/commitizen/conventional-commit-types/blob/master/index.json>`__

The scopes are the names of the modules in ConnectLib. It is not recommended but possible to have several scopes,
in that case separate them with commas.

Local development
-----------------

You need to clone the repository using ``git`` and place yourself in its directory:

.. code:: bash

    git clone git@github.com:owkin/connectlib.git
    cd connectlib

Now, install the required dependency and be sure that the current
tests are passing on your machine:

.. code:: bash

	# create virtual env and install dev requirements
	pip install -e '.[dev]'

	# Run tests
	pytest tests/

If you want to run tests in local/debug mode :

.. code:: bash

	# set your spawner mode : 'docker'(default) or 'subprocess'
	export DEBUG_SPAWNER=docker

	# Run tests
	pytest tests --local

Pre-commit hooks
^^^^^^^^^^^^^^^^

ConnectLib uses the black coding style and you must ensure that your code follows it.
If not, the CI will fail and your Pull Request will not be merged.

Similarly, we use Flake8 for linting. If you don't respect the coding conventions, the CI will fail as well.

The imports must be sorted, we are using isort in the CI, it is also included in the precomit.
Use full paths for imports from Connectlib, that is, avoid writing import .blabla, put the full path
instead: import from connectlib.blabla

Because relative imports can be messy, particularly for shared projects where directory structure is likely to change.
Relative imports are also not as readable as absolute ones, and it’s not easy to tell the location of the imported
resources.

The line length used in the repository (for black auto formatting) is 95.

To make sure that you don't accidentally commit code that does not follow the coding style,
you can install a pre-commit hook that will check that everything is in order:

.. code:: bash

    pre-commit install

You can also run it anytime using:

.. code:: bash

    pre-commit run --all-files

Tests
^^^^^

Your code must always be accompanied by corresponding tests, if tests are not present your code will not be merged.

This is the tests structure:

   -  tests

      -  resources (CAUTION: ADD YOUR RESOURCES IN GIT LFS)
      -  conftest.py
      -  <module_name>
          - test_<file_name>.py
          - ...
      - ...

Write a test
~~~~~~~~~~~~

You can refer to the `pytest <https://docs.pytest.org/en/latest/>`__
documentation to understand fixtures and test cases.

In ``conftest.py``, there are the
`fixtures <https://docs.pytest.org/en/latest/fixture.html#fixture>`__
used by all tests. You can also write your fixtures directly in the test
file.

The structure of the test files mirrors the structure of the package.
The test file names must start with ``test_``.

The test function names are of the format
``test_{function_name}_{what_is_tested}``

**Example**:

- I wrote a function `my_function` in `package > utils > functional.py`.
- I add relevant tests in the test file: `tests > utils > test_functional.py`
- My test functions are named: `test_my_function_accepts_nan`, `test_my_function_error_if_input_dim_2`
