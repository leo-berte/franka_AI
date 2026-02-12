#!/bin/bash

# Need to start from root directory
cd ..

# Define dataset path and policy name
DATA=/mnt/Data/datasets/lerobot/one_bag # conda
# DATA=data/one_bag # docker
POLICY=flowLeonardo

# python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config_test
# python src/franka_ai/training/train.py --dataset $DATA --config config_test/config_test1 --policy $POLICY

python src/franka_ai/training/train.py --dataset $DATA --config config_test/config_test1 --policy $POLICY
python src/franka_ai/training/train.py --dataset $DATA --config config_test/config_test2 --policy $POLICY
python src/franka_ai/training/train.py --dataset $DATA --config config_test/config_test3 --policy $POLICY
python src/franka_ai/training/train.py --dataset $DATA --config config_test/config_test4 --policy $POLICY