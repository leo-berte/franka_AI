#!/bin/bash

## NOTE: Check if it creates the same folder name for checkpoints since when 2 trainings are launched together ##

DATA=/home/leonardo/Documents/Coding/franka_AI/data/single/single_outliers
POLICY=diffusion

python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config6 &
wait
python src/franka_ai/training/train.py --dataset $DATA --config config6 --policy $POLICY &
wait

python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config7 &
wait
python src/franka_ai/training/train.py --dataset $DATA --config config7 --policy $POLICY &
wait

# python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config3 &
# wait
# python src/franka_ai/training/train.py --dataset $DATA --config config3 --policy $POLICY &
# wait

# python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config4 &
# wait
# python src/franka_ai/training/train.py --dataset $DATA --config config4 --policy $POLICY &
# wait

# python src/franka_ai/dataset/generate_updated_stats.py --dataset $DATA --config config5 &
# wait
# python src/franka_ai/training/train.py --dataset $DATA --config config5 --policy $POLICY &
# wait




