<div align="left">
<a href="https://join.slack.com/t/substra-workspace/shared_invite/zt-1fqnk0nw6-xoPwuLJ8dAPXThfyldX8yA"><img src="https://img.shields.io/badge/chat-on%20slack-blue?logo=slack" /></a> <a href="https://docs.substra.org/"><img src="https://img.shields.io/badge/read-docs-purple?logo=mdbook" /></a>
<br /><br /></div>

<div align="center">
<picture>
  <object-position: center>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/substra/substra/main/Substra-logo-white.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/substra/substra/main/Substra-logo-colour.svg">
  <img alt="Substra" src="https://raw.githubusercontent.com/substra/substra/main/Substra-logo-colour.svg" width="500">
</picture>
</div>
<br>
<br>

Substra is an open source federated learning (FL) software. It enables the training and validation of machine learning models on distributed datasets. It provides a flexible Python interface and a web application to run federated learning training at scale. This specific repository is about SubstraFL the high-level federated learning Python library based on the low-level [Substra](https://github.com/Substra/substra) python library. SubstraFL is used to run complex federated learning experiments at scale.

Substra's main usage is in production environments. It has already been deployed and used by hospitals and biotech companies (see the [MELLODDY](https://www.melloddy.eu/) project for instance). Substra can also be used on a single machine to perform FL simulations and debug code.

Substra was originally developed by [Owkin](https://owkin.com/) and is now hosted by the [Linux Foundation for AI and Data](https://lfaidata.foundation/). Today Owkin is the main contributor to Substra.

Join the discussion on [Slack](https://join.slack.com/t/substra-workspace/shared_invite/zt-1fqnk0nw6-xoPwuLJ8dAPXThfyldX8yA) and [subscribe here](https://lists.lfaidata.foundation/g/substra-announce/join) to our newsletter.

## How to install

```sh
pip install substrafl
```

## To start using Substra

Have a look at our [documentation](https://docs.substra.org/).

Try out our [MNIST example](https://docs.substra.org/en/stable/substrafl_doc/examples/index.html#example-to-get-started-using-the-pytorch-interface).

## Support

If you need support, please either raise an issue on Github or ask on [Slack](https://join.slack.com/t/substra-workspace/shared_invite/zt-1fqnk0nw6-xoPwuLJ8dAPXThfyldX8yA).



## Contributing

Substra warmly welcomes any contribution. Feel free to fork the repo and create a pull request.

## How to test

Install `substrafl` in editable mode with developper dependencies.
In addition, install `substra` and `substra-tools` in editable mode.
It is recommended to install all the libraries in a Python virtual env.

```sh
git clone git@github.com:Substra/substrafl.git
pip install -e "substrafl[dev]"
git clone git@github.com:Substra/substra.git
pip install -e substra
git clone git@github.com:Substra/substra-tools.git
pip install -e substra-tools
```

Now you can use the following command from `subtrafl` top level directory to run tests:

```sh
cd substrafl
make test-subprocess
```

### Running advanced test suites

Substra can be used in three different modes: using Python subprocesses (`subprocess`),
 using Docker (`docker`) and using Kubernetes (`remote`).

The command `make test-subprocess` runs the test suite in subprocess mode. It's lightweight and perfect to start.

To test with the Docker mode, you will need Docker installed and running on your machine.
If necessary, install it using [Docker Desktop](https://www.docker.com/products/docker-desktop/).

The following command runs the test suites in subprocess and Docker mode:

```sh
make test-local
``

Please be warned that some of these tests are slow and the whole test suite might require a couple hours to complete.

To try out a local deployment with Kubernetes, please follow the [installation instructions](https://docs.substra.org/en/stable/contributing/local-deployment.html) provided in the documentation.
The following command runs the remote tests:

```sh
make test-remote
```



# Appendix

## Building the documentation

The API documentation is generated from the SubstraFL repository thanks to the auto doc module.
It is automatically built by <https://github.com/Substra/substra-documentation> and integrated into the general documentation [here](https://docs.substra.org/).

You can build the API documentation locally to see the changes made by your PR.

### Requirements

You need to have substrafl.dev installed on your machine and some extra requirements. From the SubstraFL repo:

```sh
pip install -e '.[dev]'
cd docs
pip install -r requirements.txt
```

### Build

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
