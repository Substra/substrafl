<div align="left">
<a href="https://join.slack.com/t/substra-workspace/shared_invite/zt-1fqnk0nw6-xoPwuLJ8dAPXThfyldX8yA"><img src="https://img.shields.io/badge/chat-on%20slack-blue?logo=slack" /></a> <a href="https://docs.substra.org/"><img src="https://img.shields.io/badge/read-docs-purple?logo=mdbook" /></a>
<br /><br /></div>

# Substrafl

Substrafl is a high-level federated learning Python library.
Substrafl is used to run complex federated learning experiments on a Substra deployed platform. It can also be used locally to run federated learning simulations.

Note that substrafl will be soon merged with the [substra](https://github.com/substra/substra) library.

For more information on what is substrafl and how to use it, please refer to the [user documentation](https://docs.substra.org).

Join the discussion on [Slack](https://join.slack.com/t/substra-workspace/shared_invite/zt-1fqnk0nw6-xoPwuLJ8dAPXThfyldX8yA)!

## Building the documentation

The API documentation is generated from the substrafl repository thanks to the auto doc module.
It is automatically built by <https://github.com/Substra/substra-documentation> and integrated into the general documentation [here](https://connect-docs.owkin.com/).

You can build the API documentation locally to see the changes made by your PR.

### Requirements

You need to have substrafl.dev installed on your machine and some extra requirements. From the substrafl repo:

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
