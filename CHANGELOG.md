# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

- fix: `Dependency.local_package` are installed in no editable mode and additionally accepts `pyproject.yaml` as configuration file (#88)
- fix: `execute_experiment` accepts `None` as `evaluation_strategy` (#83)

### Added

- feat: make a base class for the index generator and document it (#85)

## [0.7.0] - 2022-02-01

### Fixed

- fix: notebook dependency failure (#78)
  You can now run a connectlib experiment with local dependencies in a Jupyter notebook

### Added

- feat: models can now be tested every n rounds, on the same nodes they were trained on (#79)
  This feature introduces a new parameter `evaluation_strategy` in `execute_experiment`, which takes an `EvaluationStrategy` instance from `connectlib.evaluation_strategy`.
  If this parameter is not given, performance will not be measured at all (previously, it was measured at the end of the experiment by default).

- feat: install connectlib from pypi (#71)

## [0.6.0] - 2021-12-31

### Fixed

- fix: Update pydantic version to enable autocompletion (#70)

### Added

- feat: Add a FL algorithm wrapper in PyTorch for the federated averaging strategy (#60)
- test: connect-test integration (#68)
- feat: Add a possibility to test an algorithm on selected rounds or every n rounds (#79)

## [0.5.0] - 2021-12-31

### Fixed

- fix: dependency management: the `local_code` dependencies are copied to the same folder structure relatively to the algo (#58)
- fix: dependency management - it failed when resolving the `local_code` dependencies because the path to the algo was relative (#65)

### Added

- feat: batch indexer (#67)
- feat: more logs + function to set the logging level (#56)
- Subprocess mode is now faster as it fully reuses the user environment instead of re building the connect related parts (substra #119 and #63)

## [0.4.0] - 2021-12-10

### Fixed

- fix: error message for local dependency (#52)

## [0.3.0] - 2021-11-29

### Added

- feat: User custom dependencies (#41)
- feat: support substra subprocess mode (#43)

## [0.2.0] - 2021-11-08

[Unreleased]: https://github.com/owkin/connectlib/compare/0.3.0...HEAD
