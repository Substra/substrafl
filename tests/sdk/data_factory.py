# The purpose of this file is to generate assests for testing purposes
# data samples, dataset, objective for any tests

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

import os
import pathlib
import shutil
import tempfile
import zipfile

import substra


DEFAULT_DATA_SAMPLE_FILENAME = "data.npy"

DEFAULT_SUBSTRATOOLS_VERSION = "0.9.0-minimal"

DEFAULT_OPENER_SCRIPT = f"""
import csv
import numpy as np
import shutil
import os
import substratools as tools
class TestOpener(tools.Opener):
    def get_X(self, folders):
        res = []
        for folder in folders:
            with open(os.path.join(folder, '{DEFAULT_DATA_SAMPLE_FILENAME}'), 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    res.append(int(row[0]))
        print(f'get_X: {{res}}')
        return res  # returns a list of 1's
    def get_y(self, folders):
        res = []
        for folder in folders:
            with open(os.path.join(folder, '{DEFAULT_DATA_SAMPLE_FILENAME}'), 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    res.append(int(row[1]))
        print(f'get_y: {{res}}')
        return res  # returns a list of 2's
    def fake_X(self, n_samples=None):
        if n_samples is None:
            n_samples = 1
        res = [10] * n_samples
        print(f'fake_X: {{res}}')
        return res
    def fake_y(self, n_samples=None):
        if n_samples is None:
            n_samples = 1
        res = [30] * n_samples
        print(f'fake_y: {{res}}')
        return res
    def get_predictions(self, path):
        return np.load(path)
    def save_predictions(self, y_pred, path):
        np.save(path, y_pred)
        shutil.move(str(path) + ".npy", path)
"""

DEFAULT_METRICS_SCRIPT = """
import substratools as tools
class TestMetrics(tools.Metrics):
    def score(self, y_true, y_pred):
        res = sum(y_pred) - sum(y_true)
        print(f'metrics, y_true: {{y_true}}, y_pred: {{y_pred}}, result: {{res}}')
        return res
if __name__ == '__main__':
    tools.metrics.execute(TestMetrics())
"""

DEFAULT_METRICS_DOCKERFILE = f"""
FROM gcr.io/connect-314908/connect-tools:{DEFAULT_SUBSTRATOOLS_VERSION}
COPY metrics.py .
ENTRYPOINT ["python3", "metrics.py"]
"""

DEFAULT_PERMISSIONS = substra.sdk.schemas.Permissions(public=True, authorized_ids=[])
DEFAULT_OUT_TRUNK_MODEL_PERMISSIONS = substra.sdk.schemas.PrivatePermissions(
    authorized_ids=[]
)


def zip_folder(path, destination=None):
    if not destination:
        destination = os.path.join(
            os.path.dirname(path), os.path.basename(path) + ".zip"
        )
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(path):
            for f in files:
                abspath = os.path.join(root, f)
                archive_path = os.path.relpath(abspath, start=path)
                zf.write(abspath, arcname=archive_path)
    return destination


def create_archive(tmpdir, *files):
    tmpdir.mkdir()
    for path, content in files:
        with open(tmpdir / path, "w") as f:
            f.write(content)
    return zip_folder(str(tmpdir))


def _shorten_name(name):
    """Format asset name to ensure they match the backend requirements."""
    if len(name) < 100:
        return name
    return name[:75] + "..." + name[:20]


def _get_key(obj, field="key"):
    """Get key from asset/spec or key."""
    if isinstance(obj, str):
        return obj
    return getattr(obj, field)


def _get_keys(obj, field="key"):
    """Get keys from asset/spec or key.
    This is particularly useful for data samples to accept as input args a list of keys
    and a list of data samples.
    """
    if not obj:
        return []
    return [_get_key(x, field=field) for x in obj]


class Counter:
    def __init__(self):
        self._idx = -1

    def inc(self):
        self._idx += 1
        return self._idx


class AssetsFactory:
    def __init__(self, name):
        self._data_sample_counter = Counter()
        self._dataset_counter = Counter()
        self._objective_counter = Counter()
        self._algo_counter = Counter()
        self._workdir = pathlib.Path(tempfile.mkdtemp(prefix="/tmp/"))
        self._uuid = name

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        shutil.rmtree(str(self._workdir), ignore_errors=True)

    def create_data_sample(self, content=None, datasets=None, test_only=False):
        idx = self._data_sample_counter.inc()
        tmpdir = self._workdir / f"data-{idx}"
        tmpdir.mkdir()

        content = content or "10,20"
        content = content.encode("utf-8")

        data_filepath = tmpdir / DEFAULT_DATA_SAMPLE_FILENAME
        with open(data_filepath, "wb") as f:
            f.write(content)

        datasets = datasets or []

        return substra.sdk.schemas.DataSampleSpec(
            path=str(tmpdir),
            test_only=test_only,
            data_manager_keys=datasets,
        )

    def create_dataset(
        self, py_script, objective=None, permissions=None, metadata=None
    ):
        py_script = DEFAULT_OPENER_SCRIPT
        idx = self._dataset_counter.inc()
        tmpdir = self._workdir / f"dataset-{idx}"
        tmpdir.mkdir()
        name = _shorten_name(f"{self._uuid} - Dataset {idx}")

        description_path = tmpdir / "description.md"
        description_content = name
        with open(description_path, "w") as f:
            f.write(description_content)

        opener_path = tmpdir / "opener.py"
        with open(opener_path, "w") as f:
            f.write(py_script)

        return substra.sdk.schemas.DatasetSpec(
            name=name,
            data_opener=str(opener_path),
            type="Test",
            metadata=metadata,
            description=str(description_path),
            objective_key=objective.key if objective else None,
            permissions=permissions or DEFAULT_PERMISSIONS,
        )

    def create_objective(
        self,
        dataset=None,
        data_samples=None,
        permissions=None,
        metadata=None,
        metrics=None,
    ):
        idx = self._objective_counter.inc()
        tmpdir = self._workdir / f"objective-{idx}"
        tmpdir.mkdir()
        name = _shorten_name(f"{self._uuid} - Objective {idx}")

        description_path = tmpdir / "description.md"
        description_content = name
        with open(description_path, "w") as f:
            f.write(description_content)

        if metrics:
            # if directory to the metric is given zip it
            metrics = pathlib.Path(metrics)
            metrics_zip = tmpdir / "metrics.zip"
            with zipfile.ZipFile(metrics_zip, "w") as z:
                for filepath in metrics.glob("*[!.zip]"):
                    print(f"zipped in {filepath}")
                    z.write(filepath, arcname=os.path.basename(filepath))
        else:
            # otherwise use the default factory metrics
            metrics_zip = create_archive(
                tmpdir / "metrics",
                ("metrics.py", DEFAULT_METRICS_SCRIPT),
                ("Dockerfile", DEFAULT_METRICS_DOCKERFILE),
            )

        data_samples = data_samples or []

        return substra.sdk.schemas.ObjectiveSpec(
            name=name,
            description=str(description_path),
            metrics_name="test metrics",
            metrics=str(metrics_zip),
            metadata=metadata,
            permissions=permissions or DEFAULT_PERMISSIONS,
            test_data_sample_keys=_get_keys(data_samples),
            test_data_manager_key=dataset.key if dataset else None,
        )
