import argparse
import torch
import torchvision

from franka_ai.dataset.transforms import CustomTransforms
from franka_ai.dataset.load_dataset import make_dataloader
from franka_ai.dataset.utils import get_configs_dataset, benchmark_loader, benchmark_loader2
from franka_ai.models.utils import get_configs_models

"""
Run the code: 

python src/franka_ai/tests/test_dataloader.py --dataset /mnt/Data/datasets/lerobot/one_bag --policy act --config config_test
python src/franka_ai/tests/test_dataloader.py --dataset /workspace/data/single_outliers --policy diffusion --config config1

Visualize lerobot dataset uploaded in HF:

python -m lerobot.scripts.visualize_dataset --repo-id lerobot/pusht --episode-index 0

Visualize custom dataset saved locally:

python -m lerobot.scripts.visualize_dataset --repo-id one_bag --root /mnt/Data/datasets/lerobot/one_bag --episode-index 0
"""


# TODO: 
# 1) dataloader profiler based on transforms ON/OFF and dataloader params


def parse_args():

    # set parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
        type=str,
        required=True,
        help="Absolute path to the dataset folder")
    
    parser.add_argument("--policy", type=str, default="diffusion",
                    choices=["diffusion", "act"],
                    help="Policy name")
    
    parser.add_argument("--config", type=str, default="config",
                    help="Config folder name")
    
    args = parser.parse_args()

    # return args
    return args.dataset, args.policy, args.config


def main():

    # Get path to dataset and policy name
    dataset_path, policy_name, config_folder= parse_args()

    # Get configs
    dataloader_cfg, dataset_cfg, transforms_cfg = get_configs_dataset(f"configs/{config_folder}/dataset.yaml")
    models_cfg = get_configs_models(f"configs/{config_folder}/models.yaml")
    
    # Get params
    device = torch.device(dataloader_cfg["device"])

    # Prepare transforms for training
    transforms_train = CustomTransforms(
        dataset_cfg=dataset_cfg,
        transforms_cfg=transforms_cfg,
        model_cfg=models_cfg[policy_name],
        train=True
    )

    # Create loaders
    train_loader, _, val_loader, _ = make_dataloader(
        dataset_path=dataset_path,
        dataloader_cfg=dataloader_cfg,
        dataset_cfg=dataset_cfg,
        model_cfg=models_cfg[policy_name]
    )

    # iterate over dataloader
    for batch in train_loader:
        
        # Move data to device
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}      

        # Apply custom transforms
        batch = transforms_train.transform(batch) 

        # # debug to visualise images, use path: "outputs/test.png" OR "/workspace/outputs/test.png"
        # torchvision.utils.save_image(batch["observation.images.front_cam1"][:,-1], "outputs/batch_images_last.png", nrow=4) # conda env
        # torchvision.utils.save_image(batch["observation.images.front_cam1"][0,:], "outputs/history_images.png", nrow=4) # conda env

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