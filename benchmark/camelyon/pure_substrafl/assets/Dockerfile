# this base image works in both CPU and GPU enabled environments

# TODO: change before merge
# FROM ghcr.io/substra/substra-tools:latest-nvidiacuda11.6.0-base-ubuntu20.04-python3.9
FROM ghcr.io/substra/substra-tools:test_gt_amd64

# install dependencies # TODO remove sklearn ?
RUN pip3 install scikit-learn

# add your metrics script to docker image
ADD metrics.py .

# define how script is run
ENTRYPOINT ["python3", "metrics.py", "--method-name", "score"]
