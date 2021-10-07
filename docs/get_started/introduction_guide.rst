Introduction Guide
==================

Installation
------------

Install the repository:

With pip >= 19.0.0:

Connectlib uses Owkin private Pypi repository, if you do not have credentials ask
`Olivier Léobal <mailto:olivier.leobal@owkin.com>`_
You can setup the credentials once and for all inside your pip.conf

Private dependencies used by Connectlib are SubstraTools and Substra.
You need to have the correct rights to access those repositories.
You can get them from:

- `Substra <https://github.com/owkin/substra>`_
- `SubstraTools <https://github.com/owkin/connect-tools>`_

Next, install them. First, from substra directory:

.. code-block:: bash

    $ pip install -e substra

And from connect-tools directory:

.. code-block:: bash

    $ pip install -e substratools

Now, install connectlib:

.. code-block:: bash

    $ pip install --extra-index-url <login>:<password>@pypi.owkin.com connectlib
