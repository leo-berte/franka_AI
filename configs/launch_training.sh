#!/bin/bash

DATA=/mnt/Data/datasets/lerobot/one_bag
POLICY=flowLeonardo

# python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config_test
python src/franka_ai/training/train.py --dataset $DATA --config config_test --policy $POLICY

# DATA=data/cubes_no_grasp
# POLICY=act
