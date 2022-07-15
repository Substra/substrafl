Algorithms
==========

Torch Algorithms
^^^^^^^^^^^^^^^^

TorchFedAvgAlgo
-----------------

.. autoclass:: connectlib.algorithms.pytorch.TorchFedAvgAlgo
    :private-members: _local_train
    :inherited-members:

TorchScaffoldAlgo
-----------------

.. automodule:: connectlib.algorithms.pytorch.torch_scaffold_algo
    :private-members: _local_train, _scaffold_parameters_update
    :inherited-members:

TorchNewtonRaphsonAlgo
----------------------

.. automodule:: connectlib.algorithms.pytorch.torch_newton_raphson_algo
    :private-members: _local_train, _update_gradients_and_hessian
    :inherited-members:

TorchSingleOrganizationAlgo
----------------------------

.. autoclass:: connectlib.algorithms.pytorch.TorchSingleOrganizationAlgo
    :private-members: _local_train
    :inherited-members:


Torch Base Class
-----------------

.. autoclass:: connectlib.algorithms.pytorch.torch_base_algo.TorchAlgo
    :private-members:
    :inherited-members:


Torch functions
^^^^^^^^^^^^^^^^^^

.. automodule:: connectlib.algorithms.pytorch.weight_manager


Base Class
^^^^^^^^^^

.. autoclass:: connectlib.algorithms.algo.Algo
    :private-members:
