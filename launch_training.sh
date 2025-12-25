#!/bin/bash

## NOTE: Check if it creates the same folder name for checkpoints since when 2 trainings are launched together ##

DATA=/mnt/Data/datasets/lerobot/single_outliers
# DATA=/workspace/data/single_outliers

POLICY=act

python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config_kin_act1 &
wait
python src/franka_ai/training/train.py --dataset $DATA --config config_kin_act1 --policy $POLICY &
wait

python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config_kin_act2 &
wait
python src/franka_ai/training/train.py --dataset $DATA --config config_kin_act2 --policy $POLICY &
wait

python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config_kin_act3 &
wait
python src/franka_ai/training/train.py --dataset $DATA --config config_kin_act3 --policy $POLICY &
wait

python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config_kin_act4 &
wait
python src/franka_ai/training/train.py --dataset $DATA --config config_kin_act4 --policy $POLICY &
wait

python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config_kin_act5 &
wait
python src/franka_ai/training/train.py --dataset $DATA --config config_kin_act5 --policy $POLICY &
wait

python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config_kin_act6 &
wait
python src/franka_ai/training/train.py --dataset $DATA --config config_kin_act6 --policy $POLICY &
wait




