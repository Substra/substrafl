from typing import Any
from typing import Generator
from typing import List
from typing import Union

import numpy as np
import torch


def is_batchnorm_layer(layer: torch.nn.Module) -> bool:
    """Checks if the provided layer is a Batch Norm layer (either 1D, 2D or 3D).

    Args:
        layer (torch.nn.Module): Pytorch module.

    Returns:
        bool: Whether the given module is a batch norm one.
    """
    list_bn_layers = [torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d]
    for bn_layer_class in list_bn_layers:
        if isinstance(layer, bn_layer_class):
            return True

    return False


def batch_norm_param(
    model: torch.nn.Module,
) -> Generator[torch.nn.parameter.Parameter, Any, Any]:
    """Generator of the internal parameters of the batch norm layers
    of the model. This yields references hence all modification done to the yielded object will
    be applied to the input model.

    The internal parameters of a batch norm layer include the running mean and
    the running variance.

    Args:
        model (torch.nn.Module): A torch model.

    Yields:
        The running mean and variance of all batch norm layers parameters from the given model.
    """
    for _, module in model.named_modules():
        if is_batchnorm_layer(module):
            yield module.running_mean
            yield module.running_var


def model_parameters(model: torch.nn.Module, with_batch_norm_parameters: bool) -> torch.nn.parameter.Parameter:
    """A generator of the given model parameters. The returned generator yields references hence all modification done
    to the yielded object will be applied to the input model. If with_batch_norm_parameters is set to True, the running
    mean and the running variance of each batch norm layer will be added after the "classic" parameters.

    Args:
        model (torch.nn.Module): A torch model.
        with_batch_norm_parameters (bool): If set to True, the running mean
            and the running variance of each batch norm layer will be added
            after the "classic" parameters.

    Returns:
        Generator[torch.nn.parameter.Parameter, Any, Any]: A python generator of torch parameters.
    """

    def my_iterator():
        for p in model.parameters():
            yield p

        if with_batch_norm_parameters:
            for p in batch_norm_param(model):
                yield p

    return my_iterator


def get_parameters(
    model: torch.nn.Module,
    with_batch_norm_parameters: bool,
) -> List[torch.nn.parameter.Parameter]:
    """Model parameters from the provided torch model. This function returns a copy not a reference. If
    with_batch_norm_parameters is set to True, the running mean and the running variance of the batch norm
    layers will be added after the "classic" parameters of the model.

    Args:
        model (torch.nn.Module): A torch model.
        with_batch_norm_parameters (bool): If set to True, the running mean
            and the running variance of each batch norm layer will be added
            after the "classic" parameters.

    Returns:
        List[torch.nn.parameter.Parameter]: The list of torch parameters of the provided model.
    """
    with torch.inference_mode():
        iter_params = model_parameters(model, with_batch_norm_parameters=with_batch_norm_parameters)
        parameters = [p.clone() for p in iter_params()]

    return parameters


def increment_parameters(
    model: torch.nn.Module,
    gradients: Union[List[torch.nn.parameter.Parameter], np.ndarray],
    with_batch_norm_parameters: bool,
):
    """Add the given gradient to the model parameters. If with_batch_norm_parameters is set to True, the operation
    will include the running mean and the running variance of the batch norm layers (in this case, they must be
    included in the given gradient). This function modifies the given model internally and therefore returns nothing.

    Args:
        model (torch.nn.Module): The torch model to modify.
        gradients (List[torch.nn.parameter.Parameter]): A list of torch parameters to add to the model, as ordered by
            the standard iterators.
        with_batch_norm_parameters (bool): If set to True, the running mean and the running variance of each batch norm
            layer will be included in the model parameters to modify.
    """
    with torch.inference_mode():
        # INFO: this is the faster way I found of checking that both model.parameters() and shared states has the
        # same length as model.parameters() is a generator.
        iter_params = model_parameters(model=model, with_batch_norm_parameters=with_batch_norm_parameters)
        n_parameters = len(list(iter_params()))
        assert n_parameters == len(gradients), "Length of model parameters and gradients are unequal."

        for weights, gradient in zip(iter_params(), gradients):
            if isinstance(gradient, np.ndarray):
                gradient = torch.from_numpy(gradient)
            assert gradient.data.shape == weights.data.shape, (
                f"The shape of the model weights ({weights.data.shape}) and of the gradient ({gradient.data.shape}) "
                "passed in the gradients argument are unequal."
            )
            weights.data += gradient.data


def subtract_parameters(
    parameters: List[torch.nn.parameter.Parameter],
    old_parameters: List[torch.nn.parameter.Parameter],
) -> List[torch.nn.parameter.Parameter]:
    """Subtract the given list of torch parameters i.e. : parameters - old_parameters.
    Those elements can be extracted from a model thanks to the :func:`~get_parameters` function.

    Args:
        parameters (List[torch.nn.parameter.Parameter]): A list of torch parameters.
        old_parameters (List[torch.nn.parameter.Parameter]): A list of torch parameters.

    Returns:
        List[torch.nn.parameter.Parameter]: The subtraction of the given parameters.
    """

    model_gradient = []

    assert len(parameters) == len(old_parameters), "Length of model parameters and old_parameters are unequal."

    for weights, old_weights in zip(parameters, old_parameters):
        assert weights.data.shape == old_weights.data.shape, (
            f"The shape of the parameter weights ({weights.data.shape}) and of the old parameter weights "
            f"({old_weights.data.shape}) are unequal."
        )
        model_gradient.append((weights - old_weights))

    return model_gradient


def set_parameters(
    model: torch.nn.Module,
    parameters: List[torch.nn.parameter.Parameter],
    with_batch_norm_parameters: bool,
):
    """Sets the parameters of a pytorch model to the provided parameters. If with_batch_norm_parameters is set to True,
    the operation will include the running mean and the running variance of the batch norm layers (in this case, they
    must be included in the given gradient). This function modifies the given model internally and therefore returns
    nothing.

    Args:
        model (torch.nn.Module): The torch model to modify.
        parameters (List[torch.nn.parameter.Parameter]): Model parameters, as ordered by the standard iterators.
        with_batch_norm_parameters (bool): Whether to the batch norm layers' internal parameters are provided and
            need to be included in the operation
    """
    with torch.inference_mode():
        iter_params = model_parameters(model, with_batch_norm_parameters=with_batch_norm_parameters)
        n_parameters = len(list(iter_params()))
        assert n_parameters == len(parameters), "Length of model parameters and provided parameters are unequal."
        for (p, w) in zip(iter_params(), parameters):
            p.data = w.data
