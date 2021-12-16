Technical Guide
===============

Pull Requests
--------------

Description
^^^^^^^^^^^

- Fill in the required template:

  -  if there in a associated issue, a reference to it (only need to write ``#{issue_number}`` in GitHub, e.g. ``#85``)
  -  if there is no issue, a description of the problem it solves
  -  a technical description of the implementation

  Adding ``Closes #{issue_number}`` in the PR description to automatically close the
  issue once the request is merged into master.

- Be sure that your pull request contains tests that cover the changed or added code.
- If your changes warrant a documentation change, the pull request must also update the documentation.
- The PR name should use :ref:`conventional commit naming<Commits - format>`.
- If your PR add changes that are visible to the user (usually ``feat``, ``fix`` and ``doc`` commits),
  please update the changelog with at least your commit name.
  The changelog is made for the connectlib users : you can add descriptions and precise api changes if needed.

Commits - format
^^^^^^^^^^^^^^^^

Commits on the master branch and all branches where several people may
work at the same time follow the **conventional commit** convention. If
someone works on his own branch, then he is expected to **squash** his
commits into one commit in the conventional format at the merge.

.. code::

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

Now, install connectlib and its required dependencies:

.. code:: bash

	# create virtual env and install dev requirements
	pip install -e '.[dev]'


Tests
^^^^^

Run the tests
~~~~~~~~~~~~~

Some tests (marked with ``@substra``) use ``substra``, which has different backends.

Requirements:

   * If you want to use the local debug mode of substra with docker, you will need docker to be running on your machine

   * | If you want to use the remote mode of substra, you will need to deploy the whole substra stack (backend, orchestrator, hlf-k8s) following this
      `guide <https://github.com/owkin/tech-team/wiki/Deploy-Connect-locally-with-k3s>`__.


#. | If you want to run the tests with every ``substra`` mode (subprocess, docker, remote), e.g. before a release
    (``@substra`` tests are ran with each different modes):

    .. code:: bash

            make test

#. If you want to run the tests using the local/debug mode of substra:

    .. code:: bash

            # Run all the tests in subprocess mode (no specific requirements)
            make test-subprocess

            # Run all the tests in docker mode
            make test-docker

            # Run tests with both docker and subprocess mode (@substra tests are ran twice with different modes)
            make test-local

#. If you want to run the tests using the remote mode of substra:

    .. code:: bash

            make test-remote

#. Additionnal informations

    Some tests are marked as ``docker_only`` and won't be run in subprocess mode with the ``make`` commands
    because they test features that are not compatible with this mode.
    If you need to force these tests into subprocess mode:

    .. code:: bash

        # Install the library (done in docker before)
        cd tests/dependency/installable_library
        pip install -e .
        cd -
        export DEBUG_SPAWNER=subprocess
        pytest tests --local


Write a test
~~~~~~~~~~~~

When appropriate, your code should be accompanied by corresponding tests.

This is the tests structure:

-  tests

   -  resources (CAUTION: ADD YOUR RESOURCES IN GIT LFS)
   -  conftest.py
   -  <module_name>
       - test_<file_name>.py
       - ...
   - ...

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

- I wrote a function ``my_function`` in ``package > utils > functional.py``.
- I add relevant tests in the test file: ``tests > utils > test_functional.py``
- My test functions are named: ``test_my_function_accepts_nan``, ``test_my_function_error_if_input_dim_2``


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
