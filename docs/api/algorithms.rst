Algorithms
==========

Torch Algorithms
^^^^^^^^^^^^^^^^

TorchFedAvgAlgo
-----------------

.. autoclass:: substrafl.algorithms.pytorch.TorchFedAvgAlgo
    :private-members: _local_train, _local_predict, _save_predictions
    :inherited-members:

TorchScaffoldAlgo
-----------------

.. automodule:: substrafl.algorithms.pytorch.torch_scaffold_algo
    :private-members: _local_train, _local_predict, _scaffold_parameters_update, _save_predictions
    :inherited-members:

TorchNewtonRaphsonAlgo
----------------------

.. automodule:: substrafl.algorithms.pytorch.torch_newton_raphson_algo
    :private-members: _local_train, _local_predict, _update_gradients_and_hessian, _save_predictions
    :inherited-members:

TorchSingleOrganizationAlgo
----------------------------

.. autoclass:: substrafl.algorithms.pytorch.TorchSingleOrganizationAlgo
    :private-members: _local_train, _local_predict, _save_predictions
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
