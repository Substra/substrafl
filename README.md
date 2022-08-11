# Substrafl

## Installation

With pip >= 21.2.0:

```bash
pip install substrafl
```

## Documentation

The API documentation is generated from the substrafl repository thanks to the auto doc module.
It is automatically built by <https://github.com/Substra/substra-documentation> and integrated into the general documentation [here](https://connect-docs.owkin.com/).

You can build the API documentation locally to see the changes made by your PR.

### Requirements

You need to have substrafl.dev installed on your machine and some extra requirements. From the substrafl repo:

```sh
pip install -e .[dev]
cd docs
pip install -r requirements.txt
```

### Building the documentation

You can build the documentation to see if your changes are well taken into account.
From the ./docs folder :

```sh
make clean html
```

No warning should be thrown by this command.

Then open the `./docs/_build/index.html` file to see the results.

You can also generate the documentation live so each of your changes are taken into account on the fly:

```sh
make livehtml
```

NB: Sometimes `make livehtml` does not take changes into account so running the `make html` command in parallel might be needed.
