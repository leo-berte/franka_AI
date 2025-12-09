#!/bin/bash

xhost +local:docker

# Set data folder based on the PC you are using 
if [ -d "/mnt/Data" ]; then
    DATA_DIR="/mnt/Data/datasets/lerobot"
else
    DATA_DIR="$(pwd)/data"
fi

docker run -it --rm \
    --net=host \
    --gpus all \
    --shm-size=32g \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd)/src/franka_ai:/workspace/src/franka_ai \
    -v $(pwd)/configs:/workspace/configs \
    -v $(pwd)/setup.py:/workspace/setup.py \
    -v ${DATA_DIR}:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    -w /workspace \
    franka_ai_train