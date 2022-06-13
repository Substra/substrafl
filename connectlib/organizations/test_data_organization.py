from typing import List

from connectlib.organizations.organization import Organization


class TestDataOrganization(Organization):
    """A organization on which you will test your algorithm.
    A TestDataOrganization must also be a train data organization for now.

    Args:
        organization_id (str): The substra organization ID (shared with other organizations if permissions are needed)
        data_manager_key (str): Substra data_manager_key opening data samples used by the strategy
        test_data_sample_keys (List[str]): Substra data_sample_keys used for the training on this organization
        metric_keys (List[str]):  Substra metric keys to the metrics, use substra.Client().add_metric()
    """

    def __init__(
        self,
        organization_id: str,
        data_manager_key: str,
        test_data_sample_keys: List[str],
        metric_keys: List[str],  # key to the metric, use substra.Client().add_metric()
    ):
        self.data_manager_key = data_manager_key
        self.test_data_sample_keys = test_data_sample_keys
        self.metric_keys = metric_keys

        super(TestDataOrganization, self).__init__(organization_id)

    def update_states(
        self,
        traintuple_id: str,
        round_idx: int,
    ):
        """Creating a test tuple based on the organization characteristic.

        Args:
            traintuple_id (str): The substra parent id
            round_idx: (int): Round number of the strategy starting at 1.

        """
        self.tuples.append(
            {
                "metric_keys": self.metric_keys,
                "traintuple_id": traintuple_id,
                "data_manager_key": self.data_manager_key,
                "test_data_sample_keys": self.test_data_sample_keys,
                "metadata": {
                    "round_idx": round_idx,
                },
            }
        )

    def summary(self) -> dict:
        """Summary of the class to be exposed in the experiment summary file

        Returns:
            dict: a json-serializable dict with the attributes the user wants to store
        """
        summary = super().summary()
        summary.update(
            {
                "data_manager_key": self.data_manager_key,
                "data_sample_keys": self.test_data_sample_keys,
                "metric_keys": self.metric_keys,
            }
        )
        return summary
