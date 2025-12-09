#!/bin/bash

xhost +local:docker

# Set data folder based on the PC you are using 
if [ -d "/mnt/Data" ]; then
    DATA_DIR="/mnt/Data/datasets/lerobot"
else
    DATA_DIR="$(pwd)/data"
fi

docker run -it --rm \
    --env QT_X11_NO_MITSHM=1 \
    --env RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    --env ROS_DOMAIN_ID=39 \
    --net host \
    --ipc host \
    --pid host \
    --gpus all \
    --shm-size=32g \
    -v $(pwd)/src/franka_ai:/workspace/src/franka_ai \
    -v $(pwd)/configs:/workspace/configs \
    -v $(pwd)/setup.py:/workspace/setup.py \
    -v ${DATA_DIR}:/workspace/data \
    -v $(pwd)/outputs:/workspace/outputs \
    -v $(pwd)/ros2/franka_ai_inference:/ros2_ws/src/franka_ai_inference \
    -w /ros2_ws \
    franka_ai_inference