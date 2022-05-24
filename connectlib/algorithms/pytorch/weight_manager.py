from typing import Generator
from typing import List

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
) -> Generator[torch.nn.parameter.Parameter, None, None]:
    """Generator of the internal parameters of the batch norm layers
    of the model. This yields references hence all modification done to the yielded object will
    be applied to the input model.

    The internal parameters of a batch norm layer include the running mean and
    the running variance.

    Args:
        model (torch.nn.Module): A torch model.

    Yields:
        float: The running mean and variance of all batch norm layers parameters from the given model.
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
        typing.Generator[torch.nn.parameter.Parameter, typing.Any, typing.Any]: A python generator of torch parameters.
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
        typing.List[torch.nn.parameter.Parameter]: The list of torch parameters of the provided model.
    """
    with torch.inference_mode():
        iter_params = model_parameters(model, with_batch_norm_parameters=with_batch_norm_parameters)
        parameters = [p.clone() for p in iter_params()]

    return parameters


def increment_parameters(
    model: torch.nn.Module,
    updates: List[torch.nn.parameter.Parameter],
    with_batch_norm_parameters: bool,
    updates_multiplier: float = 1.0,
):
    """Add the given update to the model parameters. If with_batch_norm_parameters is set to True, the operation
    will include the running mean and the running variance of the batch norm layers (in this case, they must be
    included in the given update). This function modifies the given model internally and therefore returns nothing.

    Args:
        model (torch.nn.Module): The torch model to modify.
        updates (typing.List[torch.nn.parameter.Parameter]): A list of torch parameters
            to add to the model, as ordered by the standard iterators. The trainable parameters should come first
            followed by the batch norm parameters if `with_batch_norm_parameters` is set to `True`.
            If the type is np.ndarray, it is converted in `torch.Tensor`.
        with_batch_norm_parameters (bool): If set to True, the running mean and the running variance of
            each batch norm layer will be included, after the trainable layers, in the model parameters to modify.
        updates_multiplier (float, Optional): The coefficient which multiplies the updates before being added to the
            model. Defaults to 1.0.
    """
    with torch.inference_mode():
        # INFO: this is the faster way I found of checking that both model.parameters() and shared states has the
        # same length as model.parameters() is a generator.
        iter_params = model_parameters(model=model, with_batch_norm_parameters=with_batch_norm_parameters)
        n_parameters = len(list(iter_params()))
        assert n_parameters == len(updates), "Length of model parameters and updates are unequal."

        for weights, update in zip(iter_params(), updates):
            assert update.data.shape == weights.data.shape, (
                f"The shape of the model weights ({weights.data.shape}) and of the update ({update.data.shape}) "
                "passed in the updates argument are unequal."
            )
            weights.data += updates_multiplier * update.data


def subtract_parameters(
    parameters: List[torch.nn.parameter.Parameter],
    parameters_to_subtract: List[torch.nn.parameter.Parameter],
) -> List[torch.nn.parameter.Parameter]:
    """
    subtract the given list of torch parameters i.e. : parameters - parameters_to_subtract.
    Those elements can be extracted from a model thanks to the :func:`~get_parameters` function.

    Args:
        parameters (typing.List[torch.nn.parameter.Parameter]): A list of torch parameters.
        parameters_to_subtract (typing.List[torch.nn.parameter.Parameter]): A list of torch parameters.

    Returns:
        typing.List[torch.nn.parameter.Parameter]: The subtraction of the given parameters.
    """
    return weighted_sum_parameters(
        parameters_list=[parameters, parameters_to_subtract],
        coefficient_list=[1, -1],
    )


def add_parameters(
    parameters: List[torch.nn.parameter.Parameter],
    parameters_to_add: List[torch.nn.parameter.Parameter],
) -> List[torch.nn.parameter.Parameter]:
    """
    add the given list of torch parameters i.e. : parameters - parameters_to_add.
    Those elements can be extracted from a model thanks to the :func:`~get_parameters` function.

    Args:
        parameters (typing.List[torch.nn.parameter.Parameter]): A list of torch parameters.
        parameters_to_add (typing.List[torch.nn.parameter.Parameter]): A list of torch parameters.

    Returns:
        typing.List[torch.nn.parameter.Parameter]: The addition of the given parameters.
    """
    return weighted_sum_parameters(
        parameters_list=[parameters, parameters_to_add],
        coefficient_list=[1, 1],
    )


def weighted_sum_parameters(
    parameters_list: List[List[torch.Tensor]],
    coefficient_list: List[float],
) -> List[torch.Tensor]:
    """
    Do a weighted sum of the given lists of torch parameters.
    Those elements can be extracted from a model thanks to the :func:`~get_parameters` function.

    Args:
        parameters_list (typing.List[List[torch.Tensor]]): A list of List of torch parameters.
        coefficient_list (typing.List[float]): A list of coefficients which will be applied to each list of parameters.
    Returns:
        typing.List[torch.nn.parameter.Parameter]: The weighted sum of the given list of torch parameters.
    """

    weighted_sum = []

    assert all(
        len(parameters_list[0]) == len(parameters) for parameters in parameters_list
    ), "The number of parameters in each List is not the same"

    assert len(parameters_list) == len(coefficient_list), "There must be a coefficient for each List of parameters"

    for parameters_to_sum in zip(*parameters_list):
        assert all(
            parameters_to_sum[0].data.shape == parameter.data.shape for parameter in parameters_to_sum
        ), "The shape of the parameters are unequal."
        with torch.inference_mode():
            weighted_sum.append(sum(param * coeff for param, coeff in zip(parameters_to_sum, coefficient_list)))

    return weighted_sum


def set_parameters(
    model: torch.nn.Module,
    parameters: List[torch.nn.parameter.Parameter],
    with_batch_norm_parameters: bool,
):
    """Sets the parameters of a pytorch model to the provided parameters. If with_batch_norm_parameters is set to True,
    the operation will include the running mean and the running variance of the batch norm layers (in this case, they
    must be included in the given parameters). This function modifies the given model internally and therefore returns
    nothing.

    Args:
        model (torch.nn.Module): The torch model to modify.
        parameters (typing.List[torch.nn.parameter.Parameter]): Model parameters, as ordered by the standard iterators.
        The trainable parameters should come first followed by the batch norm parameters if `with_batch_norm_parameters`
            is set to `True`.
        with_batch_norm_parameters (bool): Whether to the batch norm layers' internal parameters are provided and
            need to be included in the operation.
    """
    with torch.inference_mode():
        iter_params = model_parameters(model, with_batch_norm_parameters=with_batch_norm_parameters)
        n_parameters = len(list(iter_params()))
        assert n_parameters == len(parameters), "Length of model parameters and provided parameters are unequal."
        for (p, w) in zip(iter_params(), parameters):
            p.data = w.data


def zeros_like_parameters(
    model: torch.nn.Module,
    with_batch_norm_parameters: bool,
    device: torch.device,
) -> List[torch.Tensor]:
    """Copy the model parameters from the provided torch model and sets values to zero.
    If with_batch_norm_parameters is set to True, the running mean and the running variance of the batch norm
    layers will be added after the "classic" parameters of the model.

    Args:
        model (torch.nn.Module): A torch model.
        with_batch_norm_parameters (bool): If set to True, the running mean
            and the running variance of each batch norm layer will be added
            after the "classic" parameters.
        device (torch.device): torch device on which to save the parameters

    Returns:
        typing.List[torch.nn.parameter.Parameter]: The list of torch parameters of the provided model
        with values set to zero.
    """
    with torch.inference_mode():
        iter_params = model_parameters(model, with_batch_norm_parameters=with_batch_norm_parameters)
        parameters = [torch.zeros_like(p).to(device) for p in iter_params()]

    return parameters
