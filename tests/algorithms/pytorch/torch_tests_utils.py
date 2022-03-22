import torch

from connectlib.algorithms.pytorch.weight_manager import get_parameters


def assert_tensor_list_equal(tensor_list_1, tensor_list_2):
    assert len(tensor_list_1) == len(tensor_list_2)

    for params1, params2 in zip(tensor_list_1, tensor_list_2):
        assert torch.equal(params1, params2)


def assert_tensor_list_not_zeros(tensor_list):
    assert len(tensor_list) != 0
    for tensor in tensor_list:
        assert torch.nonzero(tensor).numel()  # Will raise if the number of non zeros element is 0


def assert_model_parameters_equal(model1, model2):
    model1_params = get_parameters(model1, with_batch_norm_parameters=True)
    model2_params = get_parameters(model2, with_batch_norm_parameters=True)
    assert_tensor_list_equal(model1_params, model2_params)
