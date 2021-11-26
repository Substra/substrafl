from logging import getLogger
from pathlib import Path

import numpy as np
import substra
import utils

from connectlib import execute_experiment
from connectlib.algorithms import Algo
from connectlib.dependency import Dependency
from connectlib.nodes import AggregationNode, TestDataNode, TrainDataNode
from connectlib.remote import remote_data
from connectlib.strategies import FedAVG

logger = getLogger("tests")

ASSETS_DIR = Path(__file__).parent / "end_to_end" / "test_assets"
DEFAULT_PERMISSIONS = substra.sdk.schemas.Permissions(
    public=True, authorized_ids=list()
)
LOCAL_WORKER_PATH = Path.cwd() / "local-worker"


def test_fed_avg(asset_factory, network):
    # makes sure that federated average strategy leads to the averaging output of the models from both partners.
    # The data for the two partners consists of only 0s or 1s respectively. The train() returns the data.
    # predict() returns the data, score returned by AccuracyMetric (in the metric) is the mean of all the y_pred
    # passed to it. The tests asserts if the score is 0.5
    # This test only runs on two nodes.
    class MyAlgo(Algo):
        # this class must be within the test, otherwise the Docker will not find it correctly (ie because of the way
        # pytest calls it)
        def delayed_init(self, seed: int, *args, **kwargs):
            pass

        @remote_data
        def train(
            self,
            x: np.ndarray,
            y: np.ndarray,
            num_updates: int,
            n_rounds: int,
            batch_size: int,
            shared_state,
        ):
            return dict(test=np.array(x), n_samples=len(x))

        @remote_data
        def predict(self, x: np.array, shared_state):
            return shared_state["test"]

        def load(self, path: Path):
            return self

        def save(self, path: Path):
            assert path.parent.exists()
            with path.open("w") as f:
                f.write("test")

    # TODO: metrics, datasets, data_samples will be instantiate as fixture

    # generate the data for partner "0" and "1"
    for k, client in enumerate(network.clients[:2]):
        path_data = ASSETS_DIR / f"datasample_{k}"
        if not path_data.is_dir():
            path_data.mkdir()
        # all the data will be either 0s or 1s depending on the partner
        new_data = np.ones([8, 16]) * int(k)
        np.save(path_data / "data.npy", new_data)

    opener_path = ASSETS_DIR / "opener" / "opener.py"
    with open(opener_path, "r") as myfile:
        opener_script = myfile.read()

    dataset_query = asset_factory.create_dataset(
        metadata={substra.DEBUG_OWNER: network.msp_ids[0]},
        py_script=opener_script,
    )
    dataset_1_key = network.clients[0].add_dataset(dataset_query)

    dataset_2_query = asset_factory.create_dataset(
        metadata={substra.DEBUG_OWNER: network.msp_ids[1]},
        py_script=opener_script,
    )
    dataset_2_key = network.clients[1].add_dataset(dataset_2_query)

    # by assigning content we are ensuring that in the first dataset there are only 0s and in the other 1s, to be able
    # to correctly test the strategy
    data_sample = asset_factory.create_data_sample(
        datasets=[dataset_1_key], test_only=False, content="0,0"
    )
    sample_1_key = network.clients[0].add_data_sample(data_sample)

    data_sample = asset_factory.create_data_sample(
        datasets=[dataset_2_key], test_only=False, content="1,1"
    )
    sample_2_key = network.clients[1].add_data_sample(data_sample)

    data_sample = asset_factory.create_data_sample(
        datasets=[dataset_1_key], test_only=True
    )
    sample_1_test_key = network.clients[0].add_data_sample(data_sample)

    data_sample = asset_factory.create_data_sample(
        datasets=[dataset_2_key], test_only=True
    )
    sample_2_test_key = network.clients[1].add_data_sample(data_sample)

    train_data_nodes = [
        TrainDataNode(network.msp_ids[0], dataset_1_key, [sample_1_key]),
        TrainDataNode(network.msp_ids[1], dataset_2_key, [sample_2_key]),
    ]

    # define metrics using data_factory (sdk/data_factory)
    METRIC = asset_factory.create_metric(
        data_samples=[sample_1_test_key],
        permissions=DEFAULT_PERMISSIONS,
        dataset=client.get_dataset(dataset_1_key),
        metrics=str(ASSETS_DIR / "metric"),
    )
    org1_metric_key = network.clients[0].add_metric(METRIC)

    METRIC = asset_factory.create_metric(
        data_samples=[sample_2_test_key],
        permissions=DEFAULT_PERMISSIONS,
        dataset=client.get_dataset(dataset_2_key),
        metrics=str(ASSETS_DIR / "metric"),
    )
    org2_metric_key = network.clients[0].add_metric(METRIC)

    test_data_nodes = [
        TestDataNode(
            network.msp_ids[0],
            dataset_1_key,
            [sample_1_test_key],
            metric_keys=[org1_metric_key],
        ),
        TestDataNode(
            network.msp_ids[1],
            dataset_2_key,
            [sample_2_test_key],
            metric_keys=[org2_metric_key],
        ),
    ]
    num_rounds = 3  # TODO For now, num_rounds is passed to both algorithm and
    # execute_experiment because of the way the indexer works. It is to be refactored as
    # a generator with a fixed seed.
    aggregation_node = AggregationNode(network.msp_ids[0])
    my_algo0 = MyAlgo()
    algo_deps = Dependency(pypi_dependencies=["pytest"])
    strategy = FedAVG(num_rounds=num_rounds, num_updates=2, batch_size=3)

    compute_plan = execute_experiment(
        client=client,
        algo=my_algo0,
        strategy=strategy,
        train_data_nodes=train_data_nodes,
        test_data_nodes=test_data_nodes,
        aggregation_node=aggregation_node,
        num_rounds=num_rounds,
        dependencies=algo_deps,
    )

    # Wait for the compute plan to be finished
    utils.wait(network.clients[0], compute_plan)

    # read the results from saved performances
    testtuples = client.list_testtuple(
        filters=[f"testtuple:compute_plan_key:{compute_plan.key}"]
    )
    testtuple = testtuples[0]

    # assert that the calculated score matches the expected score
    assert list(testtuple.test.perfs.values())[0] == 0.5
