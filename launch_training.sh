#!/bin/bash

## NOTE: Check if it creates the same folder name for checkpoints since when 2 trainings are launched together ##

# DATA=/mnt/Data/datasets/lerobot/cubes_no_grasp
# POLICY=act

# python src/franka_ai/training/train.py --dataset $DATA --config config_test --policy $POLICY

# DATA=data/cubes_with_grasp
# POLICY=act

# python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config_cubes_with_grasp/config_act3B
# python src/franka_ai/training/train.py --dataset $DATA --config config_cubes_with_grasp/config_act11 --policy $POLICY &

# python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config_cubes_with_grasp/config_act5
# python src/franka_ai/training/train.py --dataset $DATA --config config_cubes_with_grasp/config_act5 --policy $POLICY

# python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config_cubes_with_grasp/config_act8
# python src/franka_ai/training/train.py --dataset $DATA --config config_cubes_with_grasp/config_act8 --policy $POLICY

# python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config_cubes_with_grasp/config_act9B
# python src/franka_ai/training/train.py --dataset $DATA --config config_cubes_with_grasp/config_act11 --policy $POLICY &

# python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config_cubes_with_grasp/config_act10
# python src/franka_ai/training/train.py --dataset $DATA --config config_cubes_with_grasp/config_act10 --policy $POLICY

#DATA=data/cubes_with_grasp_outlier
#POLICY=act
#python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config_cubes_with_grasp_outliers/config_act3

#python src/franka_ai/dataset/generate_updated_stats.py --dataset data/new_cubes_grasp_small_outliers --config config_new_cubes_grasp_small_outliers/config_act11
#python src/franka_ai/dataset/generate_updated_stats.py --dataset data/new_cubes_grasp_small --config config_new_cubes_grasp_small/config_act11



python src/franka_ai/dataset/generate_updated_stats.py --dataset data/new_cubes_grasp_small_outliers --config config_new_cubes_grasp_small_outliers/config_act11
python src/franka_ai/training/train.py --dataset data/new_cubes_grasp_small_outliers --config config_new_cubes_grasp_small_outliers/config_act11 --policy act