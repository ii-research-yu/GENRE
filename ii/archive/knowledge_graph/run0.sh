#!/bin/bash

CURRENT_PATH=$(pwd)
IMAGE_NAME="denden047/genre"

docker build -t ${IMAGE_NAME} "$CURRENT_PATH"/docker && \
docker run -it --rm --ipc=host \
    --gpus device=0 \
    -v "$CURRENT_PATH"/src:/workdir \
    -v "$CURRENT_PATH"/data:/data \
    -v "$CURRENT_PATH"/models:/models \
    -w /workdir \
    ${IMAGE_NAME} \
    /bin/bash -c "\
        python sample.py \
    "
