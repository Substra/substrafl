Algorithms
==========

Torch Algorithms
^^^^^^^^^^^^^^^^

TorchFedAvgAlgo
-----------------

.. autoclass:: substrafl.algorithms.pytorch.TorchFedAvgAlgo
    :private-members: _local_train
    :inherited-members:

TorchScaffoldAlgo
-----------------

.. automodule:: substrafl.algorithms.pytorch.torch_scaffold_algo
    :private-members: _local_train, _scaffold_parameters_update
    :inherited-members:

TorchNewtonRaphsonAlgo
----------------------

.. automodule:: substrafl.algorithms.pytorch.torch_newton_raphson_algo
    :private-members: _local_train, _update_gradients_and_hessian
    :inherited-members:

TorchSingleOrganizationAlgo
----------------------------

.. autoclass:: substrafl.algorithms.pytorch.TorchSingleOrganizationAlgo
    :private-members: _local_train
    :inherited-members:


Torch Base Class
-----------------

.. autoclass:: substrafl.algorithms.pytorch.torch_base_algo.TorchAlgo
    :private-members:
    :inherited-members:


Torch functions
^^^^^^^^^^^^^^^^^^

.. automodule:: substrafl.algorithms.pytorch.weight_manager


Base Class
^^^^^^^^^^

.. autoclass:: substrafl.algorithms.algo.Algo
    :private-members:
