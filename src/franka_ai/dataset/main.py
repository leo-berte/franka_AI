import argparse
import torch
import os

from franka_ai.dataset.transforms import CustomTransforms
from franka_ai.dataset.load_dataset import make_dataloader
from franka_ai.dataset.utils import get_configs_dataset, get_dataset_path

"""
Run the code: python src/franka_ai/dataset/main.py --abs_path /home/leonardo/Documents/Coding/franka_AI/data/test1
"""

def main():

    # Get path to dataset (via argparser)
    local_root = get_dataset_path()

    # Get configs about dataloader and dataset 
    dataloader_cfg, dataset_cfg, transformations_cfg = get_configs_dataset("configs/dataset.yaml")

    # Prepare transforms for training
    transforms_train = CustomTransforms(
        dataset_cfg=dataset_cfg,
        transformations_cfg=transformations_cfg,
        train=True
    )

    # Prepare transforms for inference
    transforms_val = CustomTransforms(
        dataset_cfg=dataset_cfg,
        transformations_cfg=transformations_cfg,
        train=False
    )

    # Create loaders
    train_loader, val_loader = make_dataloader(
        local_root=local_root,
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




if __name__ == "__main__":
    main()