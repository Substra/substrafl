# ConnectLib

## Installation

With pip >= 21.2.0:

```bash
# Uses Owkin private Pypi repository, if you do not have credentials ask Olivier Léobal: olivier.leobal@owkin.com
# Please setup the credentials once and for all in your ~/.pip/pip.conf file as followed :

# For basic install
[global]
extra-index-url = https://<username>:<password>@pypi.owkin.com/simple
```

## Contribute

ConnectLib is open to contributions. Please have a look at the [Contribution Guidelines](https://owkin-connectlib.readthedocs-hosted.com/en/latest/contribute/contribution_process.html).

## Release

See the release process on the tech-team [releasing guide](https://github.com/owkin/tech-team/blob/main/releasing_guide.md#connectlib).

## Development of FL strategies by FLow with the FL group

Before the strategy implementation, the FLow person in charge of the implementation (also called "dev" later) must:

- read and understand the paper
- ask who is the FL group referent on this strategy: the referent is available for questions and discussion

During the implementation:

- discuss with the referent and ask any questions
- the dev should read pre-existing code (e.g. ruche if applicable), but copy-pasting is not advised: the goal of this step is to gain a detailed understand of the strategy and how to implement it

Th FL group has two weeks to go over the PR once it is merged. The review by the FL group includes:

- checking the mathematical and algorithmic validity of the implementation
- asking to add some tests on corner cases
- pointing out inefficiencies: eg a matrix multiplication is very slow because of the implementation

The reviewer should be as clear as possible to avoid going back and forth between reviewer and dev. If necessary, the reviewer can
propose a call with the dev.

## Documentation

Documentation of the api is generated from the connectlib repository thanks to the auto doc module.

### Requirements

You need to have connectlib.dev installed on your machine and some extra requirements. From the connectlib repo:

```sh
pip install -e .[dev]
cd docs
pip install -r requirements.txt
```

### Building the documentation

You can build the documentation to see if your changes are well taken into account.
From ./docs folder :

```sh
make clean && make html
```

No warning should be thrown by this command.

Then open the `./docs/_build/index.html` file to see the results.

You can also generate the documentation live so each of your changes are taken into account on the fly:

```sh
make livehtml
```

NB: Some time, `make livehtml` do not taken changes into account so running the `make html` command in parallel can be needed.
