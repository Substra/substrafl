# Copyright 2018 Owkin, inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sdk
import substra


def test_client(network):
    for client in network.clients:
        assert isinstance(client, substra.sdk.client.Client)


def test_dataset_query(dataset_query):
    assert type(dataset_query) is dict
    assert {
        "name",
        "data_opener",
        "type",
        "description",
        "metric_key",
        "permissions",
    } <= dataset_query.keys()


def test_metric_query(metric_query):
    assert type(metric_query) is dict
    assert {
        "name",
        "metrics",
        "metrics_name",
        "description",
        "test_data_manager_key",
        "test_data_sample_keys",
        "permissions",
    } <= metric_query.keys()


def test_data_sample_query(data_sample_query):
    assert type(data_sample_query) is dict
    assert {"path", "data_manager_keys", "test_only"} <= data_sample_query.keys()


def test_data_samples_query(data_samples_query):
    assert type(data_samples_query) is dict
    assert {"paths", "data_manager_keys", "test_only"} <= data_samples_query.keys()
    assert len(data_samples_query["paths"]) > 1


def test_asset_factory(asset_factory):
    assert isinstance(asset_factory, sdk.data_factory.AssetsFactory)


def test_data_sample(data_sample):
    assert isinstance(data_sample, substra.sdk.schemas.DataSampleSpec)
