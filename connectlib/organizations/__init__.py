from connectlib.organizations.organization import Organization  # isort:skip
from connectlib.organizations.organization import OperationKey  # isort:skip

from connectlib.organizations.aggregation_organization import AggregationOrganization
from connectlib.organizations.test_data_organization import TestDataOrganization
from connectlib.organizations.train_data_organization import TrainDataOrganization

# This is needed for auto doc to find that Organization module's is organizations.organization, otherwise when
# trying to link Organization references from one page to the Organization documentation page, it fails.
AggregationOrganization.__module__ = "organizations.aggregation_organization"
Organization.__module__ = "organizations.organization"

__all__ = ["Organization", "AggregationOrganization", "TrainDataOrganization", "TestDataOrganization", "OperationKey"]
