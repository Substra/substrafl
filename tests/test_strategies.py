import numpy as np
from pathlib import Path
from logging import getLogger

from connectlib.algorithms import Algo
from connectlib.nodes import TrainDataNode, AggregationNode, TestDataNode
from connectlib.orchestrator import Orchestrator
from connectlib.remote import remote_data
from connectlib.strategies import FedAVG

import substra

logger = getLogger("tests")

ASSETS_DIR = Path(__file__).parent / "end_to_end" / "test_assets"
DEFAULT_PERMISSIONS = substra.sdk.schemas.Permissions(
    public=True, authorized_ids=list()
)
LOCAL_WORKER_PATH = Path.cwd() / "local-worker"


def test_fed_avg(asset_factory, client):
    # makes sure that federated average strategy leads to the averaging output of the models from both partners.
    # The data for the two partners consists of only 0s or 1s respectively. The train() returns the data.
    # predict() returns the data, score returned by AccuracyMetric (in the metric) is the mean of all the y_pred
    # passed to it. The tests asserts if the score is 0.5
    class MyAlgo(Algo):
        # this class must be within the test, otherwise the Docker will not find it correctly (ie because of the way
        # pytest calls it)
        def delayed_init(self, seed: int, *args, **kwargs):
            pass

        @remote_data
        def train(
            self,
            x: np.array,
            y: np.array,
            num_updates: int,
            shared_state,
        ):
            return dict(test=x)

        @remote_data
        def predict(self, x: np.array, shared_state):
            return shared_state["test"]

        def load(self, path: Path):
            return self

        def save(self, path: Path):
            assert path.parent.exists()
            with path.open("w") as f:
                f.write("test")

    # org_client = client
    partners = ["0", "1"]

    # generate the data for partner "0" and "1"
    for partner_name in partners:
        # TODO: remove datasample dirs at the end of the tests
        path_data = ASSETS_DIR / f"datasample_{partner_name}"
        if not path_data.is_dir():
            path_data.mkdir()
        # all the data will be either 0s or 1s depending on the partner
        new_data = np.ones([8, 16]) * int(partner_name)
        np.save(path_data / "data.npy", new_data)

    # client = substra.Client(debug=True)
    opener_path = ASSETS_DIR / "opener" / "opener.py"
    with open(opener_path, "r") as myfile:
        opener_script = myfile.read()

    dataset_query = asset_factory.create_dataset(
        metadata={substra.DEBUG_OWNER: partners[0]}, py_script=opener_script
    )
    dataset_1_key = client.add_dataset(dataset_query)

    dataset_2_query = asset_factory.create_dataset(
        metadata={substra.DEBUG_OWNER: partners[1]}, py_script=opener_script
    )
    dataset_2_key = client.add_dataset(dataset_2_query)

    # by assigning content we are ensuring that in the first dataset there are only 0s and in the other 1s, to be able
    # to correctly test the strategy
    data_sample = asset_factory.create_data_sample(
        datasets=[dataset_1_key], test_only=False, content="0,0"
    )
    sample_1_key = client.add_data_sample(data_sample)

    data_sample = asset_factory.create_data_sample(
        datasets=[dataset_2_key], test_only=False, content="1,1"
    )
    sample_2_key = client.add_data_sample(data_sample)

    data_sample = asset_factory.create_data_sample(
        datasets=[dataset_1_key], test_only=True
    )
    sample_1_test_key = client.add_data_sample(data_sample)

    data_sample = asset_factory.create_data_sample(
        datasets=[dataset_2_key], test_only=True
    )
    sample_2_test_key = client.add_data_sample(data_sample)

    train_data_nodes = [
        TrainDataNode(partners[0], dataset_1_key, [sample_1_key]),
        TrainDataNode(partners[1], dataset_2_key, [sample_2_key]),
    ]

    # define metrics using data_factory (sdk/data_factory)
    METRIC = asset_factory.create_metric(
        data_samples=[sample_1_test_key],
        permissions=DEFAULT_PERMISSIONS,
        dataset=client.get_dataset(dataset_1_key),
        metrics=str(ASSETS_DIR / "metric"),
    )
    org1_metric_key = client.add_metric(METRIC)

    METRIC = asset_factory.create_metric(
        data_samples=[sample_2_test_key],
        permissions=DEFAULT_PERMISSIONS,
        dataset=client.get_dataset(dataset_2_key),
        metrics=str(ASSETS_DIR / "metric"),
    )
    org2_metric_key = client.add_metric(METRIC)

    test_data_nodes = [
        TestDataNode(
            partners[0],
            dataset_1_key,
            [sample_1_test_key],
            metric_keys=[org1_metric_key],
        ),
        TestDataNode(
            partners[1],
            dataset_2_key,
            [sample_2_test_key],
            metric_keys=[org2_metric_key],
        ),
    ]

    aggregation_node = AggregationNode(partners[0])
    my_algo0 = MyAlgo()
    strategy = FedAVG(num_rounds=3, num_updates=2, batch_size=3)

    orchestrator = Orchestrator(my_algo0, strategy, num_rounds=1)
    compute_plan = orchestrator.run(
        client, train_data_nodes, aggregation_node, test_data_nodes=test_data_nodes
    )

    # read the results from saved performances
    testtuples = client.list_testtuple(
        filters=[f"testtuple:compute_plan_key:{compute_plan.key}"]
    )
    testtuple = testtuples[0]

    # assert that the calculated score matches the expected score
    assert list(testtuple.test.perfs.values())[0] == 0.5
