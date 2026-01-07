import argparse
import torch

from franka_ai.dataset.transforms import CustomTransforms
from franka_ai.dataset.load_dataset import make_dataloader
from franka_ai.dataset.utils import get_configs_dataset
from franka_ai.models.utils import get_configs_models
from franka_ai.utils.robotics_math import *


"""
Run the code: 

python src/franka_ai/tests/test_relative_poses.py --dataset /mnt/Data/datasets/lerobot/one_bag --policy act --config test_orientations2/config_rel

"""


torch.set_printoptions(
    precision=4,       # numero di decimali
    sci_mode=False     # DISABILITA notazione scientifica
)


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
    model_cfg = models_cfg[policy_name]
    
    # Get params
    device = torch.device(dataloader_cfg["device"])
    N_history = model_cfg["params"].get("n_obs_steps") or model_cfg["params"].get("N_history")
    N_chunk = model_cfg["params"].get("horizon") or model_cfg["params"].get("chunk_size") or model_cfg["params"].get("N_chunk")
    state_ranges = dataset_cfg["state_slices"]
    state_slices = {k: slice(v[0], v[1]) for k, v in state_ranges.items()}
    
    # Set params
    dataloader_cfg["batch_size"] = 1
    dataloader_cfg["shuffle"] = False
    dataloader_cfg["num_workers"] = 1
    dataloader_cfg["prefetch_factor"] = 1

    # Prepare transforms for training
    transforms_train = CustomTransforms(
        dataset_cfg=dataset_cfg,
        transforms_cfg=transforms_cfg,
        model_cfg=model_cfg,
        train=True
    )

    # Create loaders
    train_loader, _, val_loader, _ = make_dataloader(
        dataset_path=dataset_path,
        dataloader_cfg=dataloader_cfg,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        selected_episodes=[0]
    )

    # iterate over dataloader
    for step, batch in enumerate(train_loader):
        
        # Choose N_hist>1 for stress test, and skip first frames to avoid padding
        if step<5:
            continue

        # Move data to device
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}      

        print("\nPre-transform\n")

        # Get dataset states and actions for final comparison (dataset: x y z w)
        states_absolute_dataset = batch["observation.state"][:,:,21:28]
        actions_absolute_dataset = batch["action"][:,N_history:,:-1]

        # Get base pose (last observed pose) to compute ee_pose_relative --> ee_pose_absolute
        p_base = batch["observation.state"][..., state_slices["ee_pos"]][:, -1, :]   # (B,3)
        quat_base = batch["observation.state"][..., state_slices["ee_quaternion"]][:, -1, :]  # (B,4)
        quat_base = quat_base[:, [3,0,1,2]]          # xyzw → wxyz
        R_base = quaternion_to_matrix(quat_base)     # (B,3,3)
        p_base = p_base.squeeze(0)
        R_base = R_base.squeeze(0)

        print("p_base, R_base (from MAIN loop)")
        print(p_base)
        print(R_base)

        # CONVERT IN RELATIVE

        # Apply custom transforms
        batch = transforms_train.transform(batch) 

        print("\nPost-transform\n")

        # CONVERT BACK IN ABSOLUTE
        
        # states

        print("states absolute roundtrip: ")

        for i in range(N_history):

            # Get relative poses
            pos = batch["observation.state"][0, i,:3]
            ori_6d = batch["observation.state"][0, i,3:9]
            R = rotation_6d_to_matrix(ori_6d)

            # Transform relative ee_pose in absolute ee_pose wrt last observed state
            states_absolute_roundtrip = get_absolute_pose_wrt_last_state(pos, R, p_base, R_base) # quaternion notation for orientation
            print(states_absolute_roundtrip.shape)
            print(states_absolute_roundtrip)

        print("states absolute dataset: ")
        print(states_absolute_dataset.shape)
        print(states_absolute_dataset)

        # actions

        print("actions absolute roundtrip: ")

        for i in range(N_chunk):

            # Get relative poses
            pos = batch["action"][0, i,:3]
            ori_6d = batch["action"][0, i,3:9]
            R = rotation_6d_to_matrix(ori_6d)
            
            # Transform relative ee_pose in absolute ee_pose wrt last observed state
            actions_absolute_roundtrip = get_absolute_pose_wrt_last_state(pos, R, p_base, R_base) # quaternion notation for orientation
            print(actions_absolute_roundtrip.shape)
            print(actions_absolute_roundtrip)

        print("actions absolute dataset: ")
        print(actions_absolute_dataset.shape)
        print(actions_absolute_dataset)

        break





if __name__ == "__main__":
    main()