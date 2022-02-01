# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

- fix: notebook dependency failure (#78)
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
