import pytest
import torch

from substrafl.algorithms.pytorch import weight_manager


def test_get_parameters(torch_linear_model):
    # test that the correct parameters are returned

    model = torch_linear_model()
    torch.nn.init.zeros_(model.linear1.weight)
    model.linear1.bias.data.fill_(0.01)

    parameters = weight_manager.get_parameters(model=model, with_batch_norm_parameters=False)

    assert len(parameters) == 2
    assert torch.equal(parameters[0].data, torch.tensor([[0.0, 0.0]]))
    assert torch.equal(parameters[1].data, torch.tensor([0.0100]))


def test_get_batch_norm_layer(batch_norm_cnn):
    # Check that batch norm layer are retrieved properly

    torch.manual_seed(42)
    model = batch_norm_cnn()

    model_parameters = list(weight_manager.get_parameters(model=model, with_batch_norm_parameters=True))

    # The defined models has 12 "classic" parameters and 2 batch norm layers (hence 4 batch norm parameters)
    # We should retrieve 16 parameters

    assert len(model_parameters) == 16

    # By default the running mean is set to 0 and the running variance to 1 :
    # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    bn_1_rm = model_parameters[-4]
    bn_1_rv = model_parameters[-3]
    bn_2_rm = model_parameters[-2]
    bn_2_rv = model_parameters[-1]

    assert torch.equal(torch.zeros_like(bn_1_rm), bn_1_rm)
    assert torch.equal(torch.ones_like(bn_1_rv), bn_1_rv)
    assert torch.equal(torch.zeros_like(bn_2_rm), bn_2_rm)
    assert torch.equal(torch.ones_like(bn_2_rv), bn_2_rv)


@pytest.mark.parametrize("model", ["torch_linear_model", "batch_norm_cnn"])
@pytest.mark.parametrize("with_batch_norm_parameters", [True, False])
def test_torch_set_parameters(model, with_batch_norm_parameters, request):
    # Check that get_parameters retrieve the parameters given to set_parameters

    torch.manual_seed(42)
    my_model = request.getfixturevalue(model)()
    random_parameters = []
    for parameters in weight_manager.get_parameters(
        model=my_model, with_batch_norm_parameters=with_batch_norm_parameters
    ):
        random_parameters.append(torch.randn_like(parameters))

    weight_manager.set_parameters(
        model=my_model,
        parameters=random_parameters,
        with_batch_norm_parameters=with_batch_norm_parameters,
    )

    retrieved_parameters = weight_manager.get_parameters(
        model=my_model, with_batch_norm_parameters=with_batch_norm_parameters
    )

    for parameter, retrieved_parameter in zip(random_parameters, retrieved_parameters):
        assert torch.equal(parameter, retrieved_parameter)


@pytest.mark.parametrize("model", ["torch_linear_model", "batch_norm_cnn"])
@pytest.mark.parametrize("with_batch_norm_parameters", [True, False])
def test_subtract_parameters(model, with_batch_norm_parameters, request):
    # Test that the subtract_parameters method of two identical models returns zeros like
    # parameters

    torch.manual_seed(42)
    my_model1 = request.getfixturevalue(model)()

    torch.manual_seed(42)
    my_model2 = request.getfixturevalue(model)()

    subtracted_parameters = weight_manager.subtract_parameters(
        weight_manager.get_parameters(my_model1, with_batch_norm_parameters=with_batch_norm_parameters),
        weight_manager.get_parameters(my_model2, with_batch_norm_parameters=with_batch_norm_parameters),
    )

    for parameter in subtracted_parameters:
        assert torch.equal(parameter, torch.zeros_like(parameter))


@pytest.mark.parametrize("model", ["torch_linear_model", "batch_norm_cnn"])
@pytest.mark.parametrize("with_batch_norm_parameters", [True, False])
def test_increment_parameters(model, with_batch_norm_parameters, request):
    # From two identical models, check that if we add their weights, the resulting weights are the double
    # of the original ones. Also check that the running mean is still 0 (0+0) and the running variance is 2 (1+1)
    # for the batch norm layers

    torch.manual_seed(42)
    my_model1 = request.getfixturevalue(model)()

    torch.manual_seed(42)
    my_model2 = request.getfixturevalue(model)()

    weight_manager.increment_parameters(
        my_model1,
        weight_manager.get_parameters(my_model2, with_batch_norm_parameters=with_batch_norm_parameters),
        with_batch_norm_parameters=with_batch_norm_parameters,
    )

    parameters1 = weight_manager.get_parameters(my_model1, with_batch_norm_parameters=with_batch_norm_parameters)
    parameters2 = weight_manager.get_parameters(my_model2, with_batch_norm_parameters=with_batch_norm_parameters)

    for parameter1, parameter2 in zip(parameters1, parameters2):
        assert torch.equal(parameter1, 2 * parameter2)


@pytest.mark.parametrize("model", ["torch_linear_model", "batch_norm_cnn"])
@pytest.mark.parametrize("with_batch_norm_parameters", [True, False])
def test_add_parameters(model, with_batch_norm_parameters, request):
    # Test that the add_parameters method of two identical models returns the first model with parameters mutiplied by 2

    torch.manual_seed(42)
    my_model1 = request.getfixturevalue(model)()

    torch.manual_seed(42)
    my_model2 = request.getfixturevalue(model)()

    added_parameters = weight_manager.add_parameters(
        weight_manager.get_parameters(my_model1, with_batch_norm_parameters=with_batch_norm_parameters),
        weight_manager.get_parameters(my_model2, with_batch_norm_parameters=with_batch_norm_parameters),
    )

    for parameter, parameter_1 in zip(
        added_parameters,
        weight_manager.get_parameters(my_model1, with_batch_norm_parameters=with_batch_norm_parameters),
    ):
        assert torch.equal(parameter, 2 * parameter_1)


def test_weighted_sum_parameters():
    # Test that -1 * my_parameters + 2 * my_parameters = my_parameters

    my_parameters = [torch.Tensor([1.0, 2.0, 3.0]), torch.Tensor([1.0, 2.0, 3.0])]

    result = weight_manager.weighted_sum_parameters(
        parameters_list=[my_parameters, my_parameters],
        coefficient_list=[-1.0, 2.0],
    )

    for parameter, parameter_1 in zip(result, my_parameters):
        assert torch.equal(parameter, parameter_1)
