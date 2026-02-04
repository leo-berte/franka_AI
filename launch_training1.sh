#!/bin/bash

# DATA=/mnt/Data/datasets/lerobot/cubes_no_grasp # conda
DATA=data/columns_mathis # docker
POLICY=act

python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config columns_mathis/config_act3
python src/franka_ai/training/train.py --dataset $DATA --config columns_mathis/config_act3 --policy $POLICY

# python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config grasp_4pos_new_outliers/config_act9
# python src/franka_ai/training/train.py --dataset $DATA --config grasp_4pos_new_outliers/config_act9 --policy $POLICY

# python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config grasp_4pos_new_outliers/config_act5
# python src/franka_ai/training/train.py --dataset $DATA --config grasp_4pos_new_outliers/config_act5 --policy $POLICY