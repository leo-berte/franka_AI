#!/bin/bash

xhost +local:docker

# ROOT_DIR = project root folder
ROOT_DIR=$(dirname $(realpath $0))/..

docker run -it --rm \
    --net=host \
    --gpus all \
    --shm-size=32g \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $ROOT_DIR/src/franka_ai:/workspace/src/franka_ai \
    -v $ROOT_DIR/configs:/workspace/configs \
    -v $ROOT_DIR/setup.py:/workspace/setup.py \
    -v $ROOT_DIR/launch_training.sh:/workspace/launch_training.sh \
    -v $ROOT_DIR/launch_training1.sh:/workspace/launch_training1.sh \
    -v $ROOT_DIR/launch_training2.sh:/workspace/launch_training2.sh \
    -v /mnt/Data/datasets/lerobot:/workspace/data \
    -v $ROOT_DIR/outputs:/workspace/outputs \
    -w /workspace \
    franka_ai_train


