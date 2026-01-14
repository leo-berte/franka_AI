#!/bin/bash

xhost +local:docker

# ROOT_DIR = project root folder
ROOT_DIR=$(dirname $(realpath $0))/..

docker run -it --rm \
    --env QT_X11_NO_MITSHM=1 \
    --env RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    --env ROS_DOMAIN_ID=39 \
    --env DISPLAY=$DISPLAY \
    --net host \
    --ipc host \
    --pid host \
    --gpus all \
    --shm-size=32g \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $ROOT_DIR/src/franka_ai:/workspace/src/franka_ai \
    -v $ROOT_DIR/configs:/workspace/configs \
    -v $ROOT_DIR/setup.py:/workspace/setup.py \
    -v /mnt/Data/datasets/lerobot:/workspace/data \
    -v $ROOT_DIR/outputs:/workspace/outputs \
    -v $ROOT_DIR/ros2/franka_ai_inference:/ros2_ws/src/franka_ai_inference \
    -w /ros2_ws \
    franka_ai_inference

