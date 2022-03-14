from typing import Dict
from typing import List
from typing import NewType

OperationKey = NewType("OperationKey", str)


class Node:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.tuples: List[Dict] = []

    def summary(self) -> dict:
        """Summary of the class to be exposed in the experiment summary file
        For heriting classes, override this function and add ``super.summary()``

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
            "node_id": self.node_id,
        }
