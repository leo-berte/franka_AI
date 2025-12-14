import argparse
import torch
import numpy as np


from franka_ai.dataset.transforms import CustomTransforms
from franka_ai.dataset.load_dataset import make_dataloader
from franka_ai.dataset.utils import get_configs_dataset
from franka_ai.models.utils import get_configs_models
from franka_ai.models.factory import get_policy_class


"""
Run the code: 

python src/franka_ai/inference/evaluate.py --dataset /home/leonardo/Documents/Coding/franka_AI/data/single/single_outliers \
                                           --checkpoint /outputs/checkpoints/single_outliers_diffusion_2025-12-08_15-14-40 \
                                           --policy diffusion 
"""


# TODO:

# Add plotting of errors


def parse_args():

    # set parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
        type=str,
        required=True,
        help="Absolute path to the dataset folder")
    
    parser.add_argument("--checkpoint",
        type=str,
        required=True,
        help="Absolute path to the checkpoint folder")
    
    parser.add_argument("--policy", type=str, default="diffusion",
                    choices=["diffusion", "act"],
                    help="Policy name")
    
    args = parser.parse_args()

    # return args
    return args.dataset, args.checkpoint, args.policy


def main():

    # Get paths
    dataset_path, checkpoint_path, policy_name= parse_args()

    # Get configs
    dataloader_cfg, dataset_cfg, transforms_cfg = get_configs_dataset(f"{checkpoint_path}/dataset.yaml")
    models_cfg = get_configs_models(f"{checkpoint_path}/models.yaml")
    model_cfg = models_cfg[policy_name]
    
    # Get params
    device = torch.device(dataloader_cfg["device"])
    # N_history = model_cfg["params"].get("n_obs_steps") or model_cfg["params"].get("N_history")
    # N_chunk = model_cfg["params"].get("horizon") or model_cfg["params"].get("chunk_size") or model_cfg["params"].get("N_chunk")
    # fps_sampling_hist = model_cfg["sampling"]["fps_sampling_hist"]
    # fps_sampling_chunk = model_cfg["sampling"]["fps_sampling_chunk"]

    # Prepare transforms for inference
    tf_inference = CustomTransforms(
        dataset_cfg=dataset_cfg,
        transforms_cfg=transforms_cfg,
        model_cfg=model_cfg,
        train=False
    )

    # Load policy
    PolicyClass = get_policy_class(policy_name)
    policy = PolicyClass.from_pretrained(f"{checkpoint_path}/best_model.pt")
    policy.reset() # reset the policy to prepare for rollout

    # Create loaders
    train_loader, _, val_loader, _ = make_dataloader(
        dataset_path=dataset_path,
        dataloader_cfg=dataloader_cfg,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg
    )

    # Counter
    step = 1

    # iterate over dataloader
    for batch in train_loader:
        
        # Move data to device
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}      

        # Apply custom transforms
        batch = tf_inference.transform(batch) ## devo rimuovere "action" ????

        # Inference
        with torch.inference_mode():
            action = policy.select_action(batch) # (B, D) --> (B, D)
            # actions = self.policy.diffusion.generate_actions(obs) # (B, N_hist, D) --> (N_chunk, D)
            print("step: ", step)
            # print("policy action: ", action)
            step+=1

        # Move to CPU and convert to numpy
        action = action.squeeze(0).to("cpu")

        # Convert axis-angle to quaternion
        quat = CustomTransforms.axis_angle2quaternion(action[3:6])

        # # Convert gripper in binary {0,1}
        # action[-1] = CustomTransforms.gripper_continuous2discrete(action[-1])

        # Convert tensors to numpy
        action_np = action.numpy()
        quat_np = quat.numpy()

        # Build final action as expected by controller
        # action_pre_tf = np.concatenate([action_np[:3], quat_np, [action_np[-1]]])
        action_pre_tf = np.concatenate([action_np[:3], quat_np])
        print("action policy", action_pre_tf)
        
        # # Save action in buffer safely --> con dataloader se uso past actions = true come faccio? input policy è batch dataloader, devo mettere output policy
        # with self.buffer_lock:
        #     self.buffers["action"].append((self.get_clock().now().to_msg(), action_pre_tf))

        # # print all keys in dataset
        # for k, v in batch.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.shape)
        #     else:
        #         print(k, type(v))

        # # print data
        # print(f"{batch['observation.images.front_cam1'].shape=}")  # (B, N_h, c, h, w)
        # print(f"{batch['observation.state'].shape=}")  # (B, N_h, n_state)
        # print(f"{batch['observation.state'][0,0,:]}")  # (_, _, n_state)
        # print(f"{batch['action'].shape=}")  # (B, N_c, n_actions)
        # print(f"{batch['action'][0,0,:]}")  # (_, _, n_actions)
        # break


if __name__ == "__main__":
    main()