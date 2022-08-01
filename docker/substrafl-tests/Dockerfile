FROM python:3.8

WORKDIR /usr/src/app

### Dirty build to install docker client ###
# We need to install a docker client because substra debug mode relies on Docker (DinD in a side container)
# and we need a docker client to authenticate this docker against our container registry to download substra-tools.
RUN apt update && apt install --yes apt-transport-https ca-certificates curl gnupg lsb-release
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Gcloud is required to authenticate gsutil
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get update && apt-get install google-cloud-sdk -y

COPY  connect-tools/ connect-tools/
COPY  substra/ substra/
COPY  substrafl/ substrafl/
COPY  classic-ml-algorithms/ classic-ml-algorithms/

RUN cd classic-ml-algorithms && python -m pip install --no-cache-dir -e .

RUN cd connect-tools && python -m pip install --no-cache-dir -e .

RUN cd substra && python -m pip install --no-cache-dir -e .

RUN cd substrafl && python -m pip install --no-cache-dir -e .[dev]

RUN cd substrafl/benchmark/camelyon && python -m pip install -r requirements.txt

WORKDIR /usr/src/app/substrafl
