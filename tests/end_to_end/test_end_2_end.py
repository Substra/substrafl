import substra

from pathlib import Path
from loguru import logger

ASSETS_DIR = Path(__file__).parent / "test_assets"
DEFAULT_PERMISSIONS = substra.sdk.schemas.Permissions(public=True, authorized_ids=list())


def register_dataset(client: substra.Client, asset_dir: Path):
    # Add the dataset
    # This is the data opener that for nows returns directly a mdf
    logger.info("Adding dataset")
    dataset_key = client.add_dataset(
        substra.sdk.schemas.DatasetSpec(
            name="opener - MDF",
            data_opener=asset_dir / "opener" / "opener.py",
            type="opener - MDF",
            description=asset_dir / "opener" / "description.md",
            permissions=DEFAULT_PERMISSIONS,
        )
    )

    # Add the data samples : one data sample with the MDF inside
    logger.info("Adding data sample")
    data_sample_key = client.add_data_sample(
        substra.sdk.schemas.DataSampleSpec(
            path=asset_dir / "datasample",
            test_only=False,
            data_manager_keys=[dataset_key],
        )
    )

    return dataset_key, data_sample_key


def test_end_2_end():
    org1_client = substra.Client(debug=True)
    org2_client = substra.Client(debug=True)

    org1_dataset_key, org1_data_sample_key = register_dataset(org1_client, ASSETS_DIR)
    org2_dataset_key, org2_data_sample_key = register_dataset(org2_client, ASSETS_DIR)

    from connectlib.algorithms.substra_utils import add_algo
    from algo.algo import MyAlgo

    algo_key = add_algo(org1_client, MyAlgo(), permisions=DEFAULT_PERMISSIONS)

    print(algo_key)

    logger.info("Adding dimension reduction train tuple")
    traintuple = org1_client.add_traintuple(
        substra.sdk.schemas.TraintupleSpec(
            algo_key=algo_key,
            data_manager_key=org1_dataset_key,
            train_data_sample_keys=[org1_data_sample_key],
            in_models_keys=None,
            tag="dimension reduction",
            rank=None,
            metadata=None,
        )
    )
    print(traintuple)
