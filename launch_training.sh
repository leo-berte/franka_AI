#!/bin/bash

DATA=/home/leonardo/Documents/Coding/franka_AI/data/single/single_outliers
POLICY=diffusion

python src/franka_ai/training/train.py --dataset $DATA --config config1 --policy $POLICY &
wait

python src/franka_ai/training/train.py --dataset $DATA --config config2 --policy $POLICY &
wait


## Check if creates the same folder name for checkpoints since they are launched together -->

# python src/franka_ai/training/train.py --dataset $DATA --config config1 --policy $POLICY &
# python src/franka_ai/training/train.py --dataset $DATA --config config2 --policy $POLICY &
# wait

# python src/franka_ai/training/train.py --dataset $DATA --config config3 --policy $POLICY &
# python src/franka_ai/training/train.py --dataset $DATA --config config4 --policy $POLICY &
# wait