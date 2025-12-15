#!/bin/bash

## NOTE: Check if it creates the same folder name for checkpoints since when 2 trainings are launched together ##

# DATA=/mnt/Data/datasets/lerobot/single_outliers
DATA=/workspace/data/single_outliers

POLICY=act

python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config1 &
wait
python src/franka_ai/training/train.py --dataset $DATA --config config1 --policy $POLICY &
wait

python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config2 &
wait
python src/franka_ai/training/train.py --dataset $DATA --config config2 --policy $POLICY &
wait

python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config3 &
wait
python src/franka_ai/training/train.py --dataset $DATA --config config3 --policy $POLICY &
wait

python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config4 &
wait
python src/franka_ai/training/train.py --dataset $DATA --config config4 --policy $POLICY &
wait

python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config5 &
wait
python src/franka_ai/training/train.py --dataset $DATA --config config5 --policy $POLICY &
wait

python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config6 &
wait
python src/franka_ai/training/train.py --dataset $DATA --config config6 --policy $POLICY &
wait




