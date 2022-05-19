Task execution
===============

For the task to execute, the first step is to submit the algos to Connect. Then we define the tasks and submit a compute plan of tasks.

To submit the algo, we:

- write the Dockerfile
- write the algo description
- create a folder and put all the necessary files in it
- compress the folder into an archive
- submit the algo to Connect

Algo
-----

What we need to submit to Connect:

- a Dockerfile
- the file `algo.py`
- a way to install the dependencies

Base Dockerfile
^^^^^^^^^^^^^^^^

The Dockerfile describes the commands to run to create a container with all the needed dependencies.
Its base image is the connect-tools Docker image, accessible from the private Owkin docker registry (which is a google container registry).

The base image is chosen following two criteria:

- the version of Python is the same as the one the code is run with, to satisfy cloudpickle requirements
- the version of connect-tools is the version of the connect-tools installed in the Python environment (can be
  overriden, see below). If the version is inferior to 0.10.0, we use 0.10.0 as the name of the connect-tools images
  changed.

This means that:

- in remote, the Connect platform must have access to the private Owkin Docker registry (or pre-register the images)
- in local Docker mode, the user must have access to this Docker registry
- in local subprocess mode, the Dockerfile is generated but the image is not built so no need to access the Docker
  registry

Connectlib dependencies
^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: In local subprocess mode we use the packages installed in the user Python environment, so none of what
   follows is applicable

Connectlib needs the following libraries to be installed in the container:

- connect-tools
- substra
- connectlib

These libraries are available either from the user's computer or from a private Owkin PyPi.
When we are building the container, we do not have access to the private Owkin PyPi, because it would require giving the
credentials to the Docker image, which would then be available to anyone who has the rights to download the algo from
Connect.

There are two modes: the **release mode** and the **editable mode**, chosen with the ``editable_mode`` parameter in the ``dependency`` argument.

In **release mode**, Connectlib downloads the wheels of substra and connectlib (connect-tools is already installed) from
the private Owkin PyPi and copies them to the Docker image. The download is made through a subprocess, and it needs pip
to be configured to access Owkin's PyPi.

In **editable mode**, those libraries must be installed in editable mode (`pip install -e .`) in the Python's environment
Connectlib is executed with. The script goes through each library, and:

- if the wheel for the installed version already exists (looking for it in the ``$HOME/.connectlib`` folder of
  the target directory), reuse it
- otherwise generate the wheel in the ``$HOME/.connectlib`` folder and copy it to the Docker image

Then copy the wheel to the Docker image.
This is not the preferred method as it can lead to difficulties of knowing which version was used: there may be local changes to the code.

Please note that in editable mode connect-tools is re-installed in the image.


User dependencies
^^^^^^^^^^^^^^^^^^

The dependencies are defined via the Dependency class. See the API reference for its documentation.
