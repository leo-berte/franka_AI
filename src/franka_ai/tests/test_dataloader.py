import argparse
import torch
import time
import os

from franka_ai.dataset.transforms import CustomTransforms
from franka_ai.dataset.load_dataset import make_dataloader
from franka_ai.dataset.utils import get_configs_dataset, benchmark_loader, benchmark_loader2

"""
Run the code: python src/franka_ai/tests/test_dataloader.py --dataset /home/leonardo/Documents/Coding/franka_AI/data/test1
"""


# TODO: 
# 1) dataloader profiler based on transforms ON/OFF and dataloader params


def get_dataset_path(default_rel_path="data/test1"):

    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Absolute path to the dataset folder"
    )
    args = parser.parse_args()

    # return absolute path to the dataset
    return os.path.join(os.getcwd(), default_rel_path) if not args.abs_path else args.abs_path


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
        train=True
    )

    # Prepare transforms for inference
    transforms_val = CustomTransforms(
        dataloader_cfg=dataloader_cfg,
        dataset_cfg=dataset_cfg,
        transforms_cfg=transforms_cfg,
        train=False
    )

    # Prepare transforms for computing dataset statistics only
    transforms_stats = CustomTransforms(
        dataloader_cfg=dataloader_cfg,
        dataset_cfg=dataset_cfg,
        transforms_cfg=transforms_cfg,
        train=False
    )

    # Create loaders
    train_loader, val_loader, stats_loader = make_dataloader(
        dataseth_path=dataset_path,
        dataloader_cfg=dataloader_cfg,
        feature_groups=dataset_cfg["features"],
        transforms_train=transforms_train,
        transforms_val=transforms_val,
        transforms_stats=transforms_stats
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