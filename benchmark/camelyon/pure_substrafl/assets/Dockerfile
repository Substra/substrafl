# this base image works in both CPU and GPU enabled environments
FROM ghcr.io/substra/substra-tools:latest-nvidiacuda11.8.0-base-ubuntu22.04-python3.9

# install dependencies # TODO remove sklearn ?
RUN pip3 install scikit-learn

# add your metrics script to docker image
ADD metrics.py .

# define how script is run
ENTRYPOINT ["python3", "metrics.py", "--function-name", "score"]
