"""
==================================
Substrafl template example
==================================

State the objective of this example, what will the user learn. For instance:
This example illustrate the basic usage of Substrafl, and proposes a model training by federated learning
using the federated averaging strategy.

Then state the main information about the example. For instance:
- Dataset: MNIST,  database of handwritten digits.
- Model type: CNN implemented in PyTorch
- FL setup: three organizations, two data providers and one algo provider

Specify that this example uses the local mode:
This example does not use the deployed platform of Substra, it runs in local mode.

**Requirements:**

  - To run this example locally, please make sure to download and unzip the assets needed
in the same directory as the example:

    .. only:: builder_html or readthedocs

        :download:`assets required to run this example <../../../../../tmp/substrafl_fedavg_assets.zip>`

    Please ensure that all the libraries are installed, a *requirements.txt* file is included in the zip file.
    Run the command: `pip install -r requirements.txt` to install the requirements.

  - To install **Substra** and **Substrafl** follow the instructions described here:
    :ref:`get_started/installation:Installation`

"""

# %%
# Setup
# *****
# At the top of each section explain briefly the objective of the section if this is not obvious.

import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import substra
import torch
from substra.sdk.schemas import DatasetSpec
from substra.sdk.schemas import MetricSpec
from substra.sdk.schemas import Permissions
from torch import nn

from substrafl.algorithms.pytorch import TorchFedAvgAlgo
from substrafl.dependency import Dependency
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.experiment import execute_experiment
from substrafl.index_generator import NpIndexGenerator
from substrafl.nodes import AggregationNode
from substrafl.strategies import FedAvg

# The list of their associated ids (for substra permissions)
NODES_ID = ["org-1MSP", "org-2MSP"]

# The node id on which your computation tasks are registered
ALGO_NODE_ID = NODES_ID[1]

assets_directory = ""

client = substra.Client(backend_type=substra.BackendType.LOCAL_SUBPROCESS)
clients = {node_name: client for node_name in NODES_ID}

# %%
# Data and metrics
# ****************

# %%
# Data preparation
# ================

# Download/Create data
data = np.random.rand(44, 55)

splited_data = np.split(data, 2)

# %%
# Dataset registration
# ====================

dataset = DatasetSpec(
    name="MNIST",
    type="npy",
    data_opener=assets_directory / "dataset" / "opener.py",
    description=assets_directory / "dataset" / "description.md",
    permissions=Permissions(),
    logs_permission=Permissions(),
)

dataset_key = client.add_metric(dataset)

# %%
# Metrics registration
# ====================

metric_spec = MetricSpec(
    name="Accuracy",
    description=assets_directory / "metric" / "description.md",
    file=assets_directory / "metric" / "metrics.zip",
    permissions=Permissions(),
)

metric_key = client.add_metric(metric_spec)

# %%
# Specify the machine learning components
# ***************************************
# In this section, you will register an algorithm and its dependencies,
# and specify the federated learning strategy
# as well as the nodes on which to train and to test.

# %%
# CNN definition
# ==============
# We choose to use a classic torch CNN as the model to train. The model structure is defined by the user independently
# of Substrafl.

seed = 42
torch.manual_seed(seed)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, eval=False):
        x = x.relu(self.conv1(x))
        x = x.relu(x.max_pool2d(self.conv2(x), 2))
        x = x.dropout(x, p=0.5, training=not eval)
        x = x.relu(x.max_pool2d(self.conv3(x), 2))
        x = x.dropout(x, p=0.5, training=not eval)
        x = x.view(-1, 3 * 3 * 64)
        x = x.relu(self.fc1(x))
        x = x.dropout(x, p=0.5, training=not eval)
        x = self.fc2(x)
        return x.log_softmax(x, dim=1)


model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
# Substrafl algo definition
# ==========================

# Number of model update between each FL strategy aggregation.
NUM_UPDATES = 1
BATCH_SIZE = 124

index_generator = NpIndexGenerator(
    batch_size=BATCH_SIZE,
    num_updates=NUM_UPDATES,
)


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, x: Any, y: Any, is_inference: bool):
        self.x = x
        self.y = y
        self.is_inference = is_inference

    def __getitem__(self, idx):
        if not self.is_inference:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]

    def __len__(self):
        return len(self.x)


class MyAlgo(TorchFedAvgAlgo):
    def __init__(self):
        super().__init__(
            model=model,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer=optimizer,
            index_generator=index_generator,
            dataset=TorchDataset,
        )


# %%
# Algo dependencies
# =================

algo_deps = Dependency(pypi_dependencies=["numpy==1.24.3", "torch==2.0.1"])

# %%
# Federated Learning strategies
# =============================

strategy = FedAvg()

# %%
# Where to train where to aggregate
# =================================

train_data_nodes = list()
aggregation_org = AggregationNode(ALGO_NODE_ID)

# %%
# Where and when to test
# ======================

test_data_nodes = list()
my_eval_strategy = EvaluationStrategy(test_data_nodes=test_data_nodes, eval_frequency=1)

# %%
# Running the experiment
# **********************

NUM_ROUNDS = 2

computed_plan = execute_experiment(
    client=clients[ALGO_NODE_ID],
    algo=MyAlgo(),
    strategy=strategy,
    train_data_nodes=train_data_nodes,
    evaluation_strategy=my_eval_strategy,
    aggregation_node=ALGO_NODE_ID,
    num_rounds=NUM_ROUNDS,
    experiment_folder=str(pathlib.Path.cwd() / "tmp" / "experiment_summaries"),
    dependencies=algo_deps,
)

# %%
# Explore the results
# ===================

performances = client.get_performances(computed_plan.key)

plt.title("Test dataset results")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.plot(performances)
plt.legend(loc="lower right")
plt.show()
