from franka_ai.datasets.load_dataset import make_dataloader
import argparse
import torch

"""
Run the code: python src/franka_ai/datasets/main.py --path /home/leonardo/Documents/Coding/franka_AI/data/test1
"""

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=False,
        help="Absolute path to the dataset folder"
    )
    args = parser.parse_args()

    # Assign path to local dataset
    local_root="/home/leonardo/Documents/Coding/franka_AI/data/test1" if not args.path else args.path

    # 1) load dataset from local folder
    train_loader, val_loader = make_dataloader(
        local_root="/home/leonardo/Documents/Coding/franka_AI/data/test1", 
        visual_obs_names=['observation.images.front_cam1', 'observation.images.front_cam2', 'observation.images.front_cam3', 'observation.images.gripper_camera'],
        batch_size=32, 
        N_history=10, 
        N_chunk=5,
        fps_sampling=10,
        print_ds_info=True
    )

    # # 2) load dataset from HF
    # train_loader, val_loader = make_dataloader(
    #     repo_id="lerobot/pusht",
    #     batch_size=32,
    #     N_history=10, 
    #     N_chunk=5,
    #     fps=10,
    #     print_ds_info=True
    # )

    # iterate over dataloader
    for batch in train_loader:
        
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
            else:
                print(k, type(v))

        print(f"{batch['observation.images.front_cam1'].shape=}")  # (32, N_h, c, h, w)
        print(f"{batch['observation.state'].shape=}")  # (32, N_h, 14)
        print(f"{batch['action'].shape=}")  # (32, N_c, 14)
        break




if __name__ == "__main__":
    main()