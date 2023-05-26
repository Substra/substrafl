Model Loading
=============

To use use the output of a task submitted on Substra, you need to first download the files with the ``download_algo_files`` function.
These files are used to re-instantiate and load all variables to retrieve the SubstraFL object in its wanted state by the ``load_algo`` function.

An example on how to download a model is available in the MNIST Substrafl FedAvg example.

.. automodule:: substrafl.model_loading
