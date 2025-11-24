import argparse
import torch
import time
import os

from franka_ai.dataset.transforms import CustomTransforms
from franka_ai.dataset.load_dataset import make_dataloader
from franka_ai.dataset.utils import get_configs_dataset, benchmark_loader, benchmark_loader2

"""
Run the code: python src/franka_ai/tests/test_dataloader.py --dataset /home/leonardo/Documents/Coding/franka_AI/data/today_data/today

Visualize lerobot dataset uploaded in HF:

python -m lerobot.scripts.visualize_dataset --repo-id lerobot/pusht --episode-index 0

Visualize custom dataset saved locally:

python -m lerobot.scripts.visualize_dataset --repo-id today --root /home/leonardo/Documents/Coding/franka_AI/data/today_data/today --episode-index 0
"""

# TODO: 
# 1) dataloader profiler based on transforms ON/OFF and dataloader params


def get_dataset_path():

    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Absolute path to the dataset folder"
    )
    args = parser.parse_args()

    # return absolute path to the dataset
    return args.dataset


def main():

    # Get path to dataset (via argparser)
    dataset_path = get_dataset_path()

    # Get configs about dataloader and dataset 
    dataloader_cfg, dataset_cfg, transforms_cfg = get_configs_dataset("configs/dataset.yaml")

    # Prepare transforms for training
    transforms_train = CustomTransforms(
        dataloader_cfg=dataloader_cfg,
        dataset_cfg=dataset_cfg,
        transforms_cfg=transforms_cfg,
        train=False
    )

    # Prepare transforms for inference
    transforms_val = CustomTransforms(
        dataloader_cfg=dataloader_cfg,
        dataset_cfg=dataset_cfg,
        transforms_cfg=transforms_cfg,
        train=False
    )

    # Create loaders
    train_loader, _, val_loader, _ = make_dataloader(
        dataset_path=dataset_path,
        dataloader_cfg=dataloader_cfg,
        feature_groups=dataset_cfg["features"],
        transforms_train=transforms_train,
        transforms_val=transforms_val
    )

    # iterate over dataloader
    for batch in train_loader:

        # print all keys in dataset
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
            else:
                print(k, type(v))

        # print data
        print(f"{batch['observation.images.front_cam1'].shape=}")  # (B, N_h, c, h, w)
        print(f"{batch['observation.state'].shape=}")  # (B, N_h, n_state)
        print(f"{batch['observation.state'][0,0,:]}")  # (_, _, n_state)
        print(f"{batch['action'].shape=}")  # (B, N_c, n_actions)
        print(f"{batch['action'][0,0,:]}")  # (_, _, n_actions)
        break

    # benchmark dataloader
    benchmark_loader(train_loader)
    benchmark_loader2(train_loader)


if __name__ == "__main__":
    main()