from typing import OrderedDict

import pytest
import torch

from substrafl.algorithms.pytorch import weight_manager


@pytest.fixture
def batch_norm_network():
    class BatchNormNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn1 = torch.nn.BatchNorm1d(num_features=1)

        def forward(self, x):
            pass

    return BatchNormNetwork()


@pytest.mark.parametrize(
    "layer, num_parameters",
    [
        (torch.nn.Linear(1, 1), 2),
        (torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1), 2),
        (torch.nn.BatchNorm1d(num_features=1), 4),
        (torch.nn.BatchNorm2d(num_features=1), 4),
        (torch.nn.BatchNorm3d(num_features=1), 4),
    ],
)
def test_get_parameters(layer, num_parameters):
    # test that the correct parameters are returned

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = layer

    model = Network()
    # torch.nn.init.zeros_(model.linear1.weight)
    # model.linear1.bias.data.fill_(0.01)

    parameters = weight_manager.get_parameters(model=model, with_batch_norm_parameters=True)

    assert len(parameters) == num_parameters
    # assert torch.equal(parameters[0].data, torch.tensor([[0.0, 0.0]]))
    # assert torch.equal(parameters[1].data, torch.tensor([0.0100]))


def test_get_parameters_no_batch_norm(batch_norm_network):
    # Check that batch norm parameters are ignored if with_batch_norm_parameters is False

    torch.manual_seed(42)
    model = batch_norm_network()

    state_dict = OrderedDict(
        [
            ("bn1.weight", torch.tensor([5.0])),
            ("bn1.bias", torch.tensor([3.0])),
            ("bn1.running_mean", torch.tensor([0.0])),
            ("bn1.running_var", torch.tensor([1.0])),
            ("bn1.num_batches_tracked", torch.tensor(0)),
        ]
    )
    model.load_state_dict(state_dict)

    model_parameters = list(weight_manager.get_parameters(model=model, with_batch_norm_parameters=False))

    assert len(list(model_parameters)) == 0


def test_get_batch_norm_layer(batch_norm_network):
    # Check that batch norm layer are retrieved properly

    torch.manual_seed(42)
    model = batch_norm_network()

    state_dict = OrderedDict(
        [
            ("bn1.weight", torch.tensor([5.0])),
            ("bn1.bias", torch.tensor([3.0])),
            ("bn1.running_mean", torch.tensor([0.0])),
            ("bn1.running_var", torch.tensor([1.0])),
            ("bn1.num_batches_tracked", torch.tensor(0)),
        ]
    )
    model.load_state_dict(state_dict)

    model_parameters = list(weight_manager.get_parameters(model=model, with_batch_norm_parameters=True))

    assert len(list(model_parameters)) == 2

    bn_1_rm = model_parameters[-2]
    bn_1_rv = model_parameters[-1]

    assert torch.equal(torch.tensor([0.0]), bn_1_rm)
    assert torch.equal(torch.tensor([1.0]), bn_1_rv)


@pytest.mark.parametrize("model", ["torch_linear_model", "batch_norm_network"])
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


@pytest.mark.parametrize("model", ["torch_linear_model", "batch_norm_network"])
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


@pytest.mark.parametrize("model", ["torch_linear_model", "batch_norm_network"])
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


@pytest.mark.parametrize("model", ["torch_linear_model", "batch_norm_network"])
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
