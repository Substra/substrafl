# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.21.1](https://github.com/owkin/connectlib/releases/tag/0.21.1) - 2022-07-19

### Fixed

- fix: support several items in the Dependency - local_dependencies field

## [0.21.0](https://github.com/owkin/connectlib/releases/tag/0.21.0) - 2022-07-11

### Changed

- BREAKING CHANGE: convert (test task) to (predict task + test task) (#217)

### Added

- Added functions to download the model of a strategy (#208):

  - The function `connectlib.model_loading.download_algo_files` downloads the files needed to load the output model of a strategy
    according to the given round. These files are downloaded to the given folder.

  - The `connectlib.model_loading.load_algo` function to load the output model of a strategy from the files previously downloaded via the
    the function `connectlib.model_loading.download_algo_files` .

  Those two functions works together:

  ```python
  download_algo_files(client=substra_client, compute_plan_key=key, round_idx=None, dest_folder=session_dir)
  model = load_algo(input_folder=session_dir)
  ```

## [0.20.0](https://github.com/owkin/connectlib/releases/tag/0.20.0) - 2022-07-05

### Added

- compatibility with substra 0.28.0

## [0.19.0](https://github.com/owkin/connectlib/releases/tag/0.19.0) - 2022-06-27

### Added

- feat: Newton Raphson strategy (#187)

## [0.18.0](https://github.com/owkin/connectlib/releases/tag/0.18.0) - 2022-06-20

### Fixed

- added [packaging](https://github.com/pypa/packaging) to the install requirements (#209)

## Changed

- Stop using metrics APIs, use algo APIs instead (#210)

## [0.17.0](https://github.com/owkin/connectlib/releases/tag/0.17.0) - 2022-06-14

### Changed

- BREAKING CHANGE: Strategy rounds starts at `1` and initialization round is now `0`. It used to start at `0`
and the initialization round was `-1` (#200)
For each composite train tuple, aggregate tuple and test tuple the meta data `round_idx` has changed
accordingly to the rule stated above.
- BREAKING CHANGE: rename node to organization in Connect (#201)
- Rename the ``OneNode`` strategy to ``SingleOrganization`` (#206)

### Added

- when using the `TorchScaffoldAlgo`: (#199)
  - The number of time the `_scaffold_parameters_update` method  must be called within the `_local_train` method is now checked
  - A warning is thrown if an other optimizer than `SGD`
  - If multiple learning rates are set for the optimizer, a warning is thrown and the smallest learning rate is used for
  the shared state aggregation operation. `0` is not considered as a learning rate for this choice as it could be used to
  deactivate the learning process of certain layers from the model.

## [0.16.0](https://github.com/owkin/connectlib/releases/tag/0.16.0) - 2022-06-07

### Changed

- BREAKING CHANGE: add initialization round to centralized strategies (#188):
  - Each centralized strategy starts with an initialization round composed of one composite train tuple on each train data node
  - One round of a centralized strategy is now: `Aggregation` -> `Training on composite`
  - Composite train tuples before test tuples have been removed
  - All torch algorithm have now a common `predict` method
  - The `algo` argument has been removed from the `predict` method of all strategies
  - The `fake_traintuple` attribute of the `RemoteStruct` class has been removed

The full discussion regarding this feature can be found [here](https://github.com/owkin/tech-team/pull/128)

## [0.15.0](https://github.com/owkin/connectlib/releases/tag/0.15.0) - 2022-05-31

### Added

- feat: meaningful name for algo (#175). You can use the `_algo_name` parameter to set a custom algo name for the registration. By default, it is set to `method-name_class-name`.

  ```py
  algo.train(
            node.data_sample_keys,
            shared_state=self.avg_shared_state,
            _algo_name=f"Training with {algo.__class__.__name__}",
            )
  ```

## [0.14.0](https://github.com/owkin/connectlib/releases/tag/0.14.0) - 2022-05-23

### Added

- chore: add latest connect-tools docker image selection (#173)
- Torch algorithms now support GPUs, there is a parameter `use_gpu` in the `__init__` of the Torch algo classes.
    If `use_gpu` is True and there is no GPU detected, the code runs on CPU. (#145)

### Changed

- The wheels of the libraries installed with `editable=True` are now in `$HOME/.connectlib` instead of `$LIB_PATH/dist` (#177)
- benchmark:
  - `make benchmark` runs the default remote benchmark on the connect platform specified in the [config file](./benchmark/camelyon/connect_conf/ci.yaml)
  - `make benchmark-local` runs the default local benchmark in subprocess mode

## [0.13.0](https://github.com/owkin/connectlib/releases/tag/0.13.0) - 2022-05-16

### Changed

- BREAKING CHANGE: replace "tag" argument with "name" in execute_experiment (#176)
- `execute_experiment` checks that the algo and strategy are compatible. You can override the list of strategies the
algo is compatible with using the `strategies` property (#):

  ```python
  from connectlib.algorithms.algo import Algo
  from connectlib import StrategyName

  class MyAlgo(Algo):
      @property
      def strategies(self):
          return [StrategyName.FEDERATED_AVERAGING, StrategyName.SCAFFOLD]
      # ...
  ```

## [0.12.0](https://github.com/owkin/connectlib/releases/tag/0.12.0) - 2022-05-09

### Added

- feat: the compute plan key of the experiment is saved in the experiment summary before submitting or executing it (#163)

## [0.11.0](https://github.com/owkin/connectlib/releases/tag/0.11.0) - 2022-05-03

### Added

- feat: add the possibility for the user to pass additional metadata to the compute plan metadata (#161)

### Fixed

- Force the reinstallation of connect-tools in the Docker image, necessary for the editable mode (#156)

### Changed

- BREAKING CHANGE: the default value of `drop_last` in the `NpIndexGenerator` is now False (#142)

- BREAKING CHANGE: the index generator is now required when implementing a strategy (#142)

  ```python
  from connectlib.index_generator import NpIndexGenerator

  nig = NpIndexGenerator(
        batch_size=batch_size,
        num_updates=num_updates,
        drop_last=False,  # optional, defaults to False
        shuffle=True,  # optional, defaults to True
    )

  class MyAlgo(TorchFedAvgAlgo):
    def __init__(self):
        super().__init__(
            index_generator=nig,
            # other parameters
        )
    # ...
  ```

- The user can now initialize his `TorchAlgo` function with custom parameters (only primitive types are supported) (#130):

  ```python
  class MyAlgo(TorchFedAvgAlgo):
      def __init__(self, my_arg):
          super.__init__(
              model=model,
              criterion=criterion
              optimizer=optimizer,
              index_generator=nig,
              my_arg=my_arg,  # This is necessary
          )
          # ...
  ```

## [0.10.0](https://github.com/owkin/connectlib/releases/tag/0.10.0) - 2022-04-19

### Fixed

- Fix the format of the asset ids: the right format is `str(uuid.uuid4())` and not `uuid.uuid4().hex` (#141)

## [0.9.0](https://github.com/owkin/connectlib/releases/tag/0.9.0) - 2022-04-11

## Changed

- feat: rename "compute_plan_tag" to "tag" #131
- feat: Add the optional argument "compute_plan_tag" to give the user the possibility to choose its own tag (timestamp by default) #128
- feat: Scaffold strategy (#89)
- feat: add one node strategy (#106)
- The Connect tasks have a `round_idx` attribute in their metadata (#101)
- doc: add python api to documentation (#105)

### Improved

- API documentation: fix the docstrings and the display of the documentation for some functions (#122)
- (BREAKING CHANGE) FedAvg strategy: the train function must return a FedAvgSharedState, the average function returns a FedAvgAveragedState.
  No need to change your code if you use TorchFedAvgAlgo (#126)
- benchmark:
  - Use the same batch sampler between the torch and Connectlib examples (#94)
  - Make it work with `num_workers` > 0 (#94)
  - Explain the effect of the sub-sampling (#94)
  - Update the default benchmark parameters in `benchmarks.sh` (#94)
  - Add new curves to the plotting: when one parameter changes while the others stay the same (#94)
  - Use connect-tools 0.10.0 as a base image for the Dockerfile

### Fixed

- fix: naming changed from FedAVG to FedAvg (#114)
- fix: log a warning if an existing wheel is used to build the docker image (#116)

## [0.8.0](https://github.com/owkin/connectlib/releases/tag/0.8.0) - 2022-03-01

### Fixed

- fix: `execute_experiment` has no side effects on its arguments (#90)
- fix: `Dependency.local_package` are installed in no editable mode and additionally accepts `pyproject.yaml` as configuration file (#88)
- fix: `execute_experiment` accepts `None` as `evaluation_strategy` (#83)
- fix: The `connectlib.algorithms.algo.Algo` `abstractmethod` decorator is now taken into account (#96)

### Improved

- feat: `EvaluationStrategy` can now be reinitialized (#90)
- Refactoring `connectlib.algorithms.pytorch.fed_avg.TorchFedAvgAlgo`  (#92):
  - replace the `_preprocess` and `_postprocess` functions by `_local_train` and `_local_predict`
  - the user can override the `_get_len_from_x` function to get the number of samples in the dataset from x
  - `batch_size` is now a required argument, and a warning is issued if it is None
- The `connectlib.index_generator.np_index_generator.NpIndexGenerator` class now works with `torch.utils.data.DataLoader`, with `num_workers` > 0 (#92)
- The benchmark uses `connectlib.algorithms.pytorch.fed_avg.TorchFedAvgAlgo` instead of its own custom algorithm (#92)
- Add the `clean_models` option to the `execute_experiment` function (#100)

### Added

- feat: make a base class for the index generator and document it (#85)
- The `Algo` now exposes a `model` property to get the model after downloading it from Connect (#99)
- (BREAKING CHANGE) experiment summary is saved as a json in `experiment_folder` (#98)

## [0.7.0](https://github.com/owkin/connectlib/releases/tag/0.7.0) - 2022-02-01

### Fixed

- fix: notebook dependency failure (#78)
  You can now run a connectlib experiment with local dependencies in a Jupyter notebook

### Added

- feat: models can now be tested every n rounds, on the same nodes they were trained on (#79)
  This feature introduces a new parameter `evaluation_strategy` in `execute_experiment`, which takes an `EvaluationStrategy` instance from `connectlib.evaluation_strategy`.
  If this parameter is not given, performance will not be measured at all (previously, it was measured at the end of the experiment by default).

- feat: install connectlib from pypi (#71)

## [0.6.0](https://github.com/owkin/connectlib/releases/tag/0.6.0) - 2021-12-31

### Fixed

- fix: Update pydantic version to enable autocompletion (#70)

### Added

- feat: Add a FL algorithm wrapper in PyTorch for the federated averaging strategy (#60)
- test: connect-test integration (#68)
- feat: Add a possibility to test an algorithm on selected rounds or every n rounds (#79)

## [0.5.0](https://github.com/owkin/connectlib/releases/tag/0.5.0) - 2021-12-31

### Fixed

- fix: dependency management: the `local_code` dependencies are copied to the same folder structure relatively to the algo (#58)
- fix: dependency management - it failed when resolving the `local_code` dependencies because the path to the algo was relative (#65)

### Added

- feat: batch indexer (#67)
- feat: more logs + function to set the logging level (#56)
- Subprocess mode is now faster as it fully reuses the user environment instead of re building the connect related parts (substra #119 and #63)

## [0.4.0](https://github.com/owkin/connectlib/releases/tag/0.4.0) - 2021-12-10

### Fixed

- fix: error message for local dependency (#52)

## [0.3.0](https://github.com/owkin/connectlib/releases/tag/0.3.0) - 2021-11-29

### Added

- feat: User custom dependencies (#41)
- feat: support substra subprocess mode (#43)

## [0.2.0](https://github.com/owkin/connectlib/releases/tag/0.2.0) - 2021-11-08

[unreleased]: https://github.com/owkin/connectlib/compare/0.3.0...HEAD
