FROM python:3.11

WORKDIR /usr/src/app

### Dirty build to install docker client ###
# We need to install a docker client because substra debug mode relies on Docker (DinD in a side container)
# and we need a docker client to authenticate this docker against our container registry to download substra-tools.
RUN apt update && apt install --yes apt-transport-https ca-certificates curl gnupg lsb-release
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
RUN apt update && apt install --yes docker-ce-cli

COPY  substra-tools/ substra-tools/
COPY  substra/ substra/
COPY  substrafl/ substrafl/

RUN cd substrafl && python -m pip install --no-cache-dir -e .[dev]

RUN cd substra && python -m pip install --no-cache-dir -e .

RUN cd substra-tools && python -m pip install --no-cache-dir -e .

RUN cd substrafl/benchmark/camelyon && python -m pip install -r requirements.txt

WORKDIR /usr/src/app/substrafl
