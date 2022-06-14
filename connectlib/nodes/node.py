from typing import Dict
from typing import List
from typing import NewType

OperationKey = NewType("OperationKey", str)


class Node:
    def __init__(self, organization_id: str):
        self.organization_id = organization_id
        self.tuples: List[Dict] = []

    def summary(self) -> dict:
        """Summary of the class to be exposed in the experiment summary file
        For inherited classes, override this function and add ``super.summary()``

        Example:

                .. code-block:: python

                    def summary(self):

                        summary = super().summary()
                        summary.update(
                            {
                                "attribute": self.attribute,
                                ...
                            }
                        )
                        return summary

        Returns:
            dict: a json-serializable dict with the attributes the user wants to store
        """
        return {
            "organization_id": self.organization_id,
        }
