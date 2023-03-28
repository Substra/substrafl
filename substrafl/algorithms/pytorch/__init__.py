from torch import __version__ as torch_version

from substrafl.algorithms.pytorch.torch_fed_avg_algo import TorchFedAvgAlgo
from substrafl.algorithms.pytorch.torch_fed_pca_algo import TorchFedPCAAlgo
from substrafl.algorithms.pytorch.torch_newton_raphson_algo import TorchNewtonRaphsonAlgo
from substrafl.algorithms.pytorch.torch_scaffold_algo import TorchScaffoldAlgo
from substrafl.algorithms.pytorch.torch_single_organization_algo import TorchSingleOrganizationAlgo
from substrafl.exceptions import UnsupportedPytorchVersionError

if torch_version == "1.12.0":
    raise UnsupportedPytorchVersionError(
        "Please use an other pytorch version. There is a regression bug in torch 1.12.0, that impacts optimizers that "
        "have been pickled and unpickled. "
        "This bug occurs for Adam optimizer for example (but not for SGD). Here is a link to one issue covering it: "
        "https://github.com/pytorch/pytorch/pull/80345"
    )


__all__ = [
    "TorchFedAvgAlgo",
    "TorchFedPCAAlgo",
    "TorchSingleOrganizationAlgo",
    "TorchScaffoldAlgo",
    "TorchNewtonRaphsonAlgo",
]
