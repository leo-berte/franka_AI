import argparse
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from franka_ai.dataset.transforms import CustomTransforms
from franka_ai.utils.robotics_math import *
from franka_ai.dataset.load_dataset import make_dataloader
from franka_ai.dataset.utils import get_configs_dataset
from franka_ai.models.utils import get_configs_models
from franka_ai.models.factory import get_policy_class


"""
Run the code: 

python src/franka_ai/inference/evaluate.py --dataset /mnt/Data/datasets/lerobot/one_bag \
                                           --checkpoint outputs/checkpoints/one_bag_flowLeonardo_config_test_2026-02-10_10-10-40 \
                                           --policy flowLeonardo

python src/franka_ai/inference/evaluate.py --dataset /workspace/data/single_outliers \
                                           --checkpoint /workspace/outputs/checkpoints/single_outliers_diffusion_2025-12-24_21-15-57 \
                                           --policy diffusion
"""


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
                    choices=["diffusion", "act", "flowLeonardo"],
                    help="Policy name")
    
    args = parser.parse_args()

    # return args
    return args.dataset, args.checkpoint, args.policy

def init_action_buffer(loader, N_history): 
    
    # get batch
    batch = next(iter(loader))

    # init action buffer
    action_buffer = deque(maxlen=N_history)
    
    for _ in range(N_history):
        action_buffer.append(batch["action"][0, 0].numpy())

    return action_buffer

def get_action_buffer_tensor(buffer):

    tensor_values = []
    for v in buffer:  
        v = torch.from_numpy(v).float()
        tensor_values.append(v)

    # Stack to (N_history, ... ) + add B dimension
    stacked_values = torch.stack(tensor_values, dim=0).unsqueeze(0)

    return stacked_values


def main():

    """
    Run offline evaluation of a trained policy on a single dataset episode.

    This function performs a sequential rollout over a dataset episode,
    emulating closed-loop inference by:
      - Maintaining an internal action buffer for past actions
      - Aligning policy inference frequency with the dataset FPS
      - Applying the same preprocessing pipeline used during training
      - Respecting the policy's internal chunking and receding-horizon logic

    Key features of the evaluation pipeline:

    - Temporal alignment:
        * Dataset samples are subsampled to match the policy sampling rate.

    - Past action conditioning:
        * When enabled, previously predicted actions are injected into the
          observation state using a fixed-length action buffer.

    - Chunk-based policy execution:
        * The policy internally handles action chunking and receding-horizon
          execution via the `select_action` method.

    - Relative end-effector pose handling:
        * When using relative end-effector actions, the reference (base) pose
          is captured at chunk boundaries and reused consistently across the
          corresponding action steps.

    - Action reconstruction:
        * Policy outputs are converted to the dataset action format (dataset.yaml).

    - Metrics computation:
        * Position error (L2)
        * Orientation error (geodesic distance)
        * Gripper command accuracy

    The resulting predictions are compared against ground-truth dataset
    actions to assess tracking performance.
    """

    # Get paths
    dataset_path, checkpoint_path, policy_name= parse_args()

    # Create folder to save plots
    img_dir = os.path.join(checkpoint_path, "plots")
    os.makedirs(img_dir, exist_ok=True)  

    # Get configs
    dataloader_cfg, dataset_cfg, transforms_cfg = get_configs_dataset(f"{checkpoint_path}/dataset.yaml")
    models_cfg = get_configs_models(f"{checkpoint_path}/models.yaml")
    model_cfg = models_cfg[policy_name]
    
    # Get params
    device = torch.device(dataloader_cfg["device"])
    N_history = model_cfg["params"].get("n_obs_steps") or model_cfg["params"].get("N_history")
    N_chunk = model_cfg["params"].get("horizon") or model_cfg["params"].get("chunk_size") or model_cfg["params"].get("N_chunk")
    n_action_steps = model_cfg["params"].get("n_action_steps")
    orientation_type = transforms_cfg["orientations"]["type"]
    include_actions = transforms_cfg["action"]["include"]
    state_ranges = dataset_cfg["state_slices"]
    state_slices = {k: slice(v[0], v[1]) for k, v in state_ranges.items()}
    
    # Set params
    dataloader_cfg["batch_size"] = 1
    dataloader_cfg["shuffle"] = False
    dataloader_cfg["num_workers"] = 1
    dataloader_cfg["prefetch_factor"] = 1

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
    train_loader, _, _, _ = make_dataloader(
        dataset_path=dataset_path,
        dataloader_cfg=dataloader_cfg,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        selected_episodes=[0] # with seed=50 --> idx=10 val_set, idx=0 train_set
    )

    # Save data
    real_action_list = []
    net_action_list = []

    # Adjust fps_dataset to fps_sampling
    dataset_meta = LeRobotDatasetMetadata(repo_id=None, root=dataset_path)
    fps_dataset = dataset_meta.fps
    fps_sampling_chunk = model_cfg["sampling"]["fps_sampling_chunk"]
    fps_sampling_hist = model_cfg["sampling"]["fps_sampling_hist"]
    step_chunk = round(fps_dataset / fps_sampling_chunk)

    # Consistency checks
    if fps_sampling_hist != fps_sampling_chunk:
        raise ValueError("fps_sampling_hist must be the same as fps_sampling_chunk.")

    # Pre-fill past actions
    action_buffer = init_action_buffer(train_loader, N_history)

    # Compute policy steps
    current_step = 0

    # iterate over dataloader
    for step, batch in enumerate(train_loader):
        
        # Skip sample to align fps_sampling_chunk with dataloader (it slides every single elements otherwise)
        if step % step_chunk != 0:
            continue
        
        # Eventually include past actions (taken from policy output)
        if transforms_cfg["state"]["use_past_actions"]:
            batch["action"][:, :N_history, ...] = get_action_buffer_tensor(action_buffer)

        # Move data to device
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}            

        # Get base pose (last observed pose) to compute ee_pose_relative --> ee_pose_absolute
        if "ee_pose_relative" in include_actions and current_step % n_action_steps == 0:
            p_base = batch["observation.state"][..., state_slices["ee_pos"]][:, -1, :]   # (B,3)
            quat_base = batch["observation.state"][..., state_slices["ee_quaternion"]][:, -1, :]  # (B,4)
            quat_base = quat_base[:, [3,0,1,2]]          # xyzw → wxyz
            R_base = quaternion_to_matrix(quat_base)     # (B,3,3)
            p_base = p_base.squeeze(0)
            R_base = R_base.squeeze(0)

        # Extract real action from dataset
        real_action = batch["action"][:, N_history, ...].to("cpu")  # (B, D)
        q = real_action[:, 3:7]  # (B, 4)
        q = q[:, [3, 0, 1, 2]] # Permute to PyTorch3D format: w x y z
        q = normalize_quat(q) # Normalize
        # q = standardize_quaternion(q) # Standardize (ensure w >= 0)
        q = q[:, [1, 2, 3, 0]] # Convert back to dataset format: x y z w
        real_action[:, 3:7] = q # Put back into action vector
        real_action_list.append(real_action.squeeze(0).numpy())

        # Apply custom transforms
        batch = tf_inference.transform(batch)

        # print("transform", {k:v.shape for k,v in batch.items() if isinstance(v, torch.Tensor)})

        # Remove "action" from observations
        batch.pop("action")

        # # Inject noise in state to see divergence
        # std_noise = 0.1
        # batch["observation.state"] += torch.randn_like(batch["observation.state"]) * std_noise

        # Inference
        with torch.inference_mode():
            action = policy.select_action(batch) # (B, N_hist, D) --> (B, D)
            # actions = self.policy.diffusion.generate_actions(obs) # (B, N_hist, D) --> (B, N_chunk, D)
            print("step: ", step)
        
        # Update policy steps
        current_step += 1

        # Move to CPU
        action = action.squeeze(0).to("cpu")

        action_parts = []

        for action_name in include_actions:

            if action_name == "ee_pose_absolute":

                pos = action[:3]

                if orientation_type == "quaternion":
                    quat = action[3:7]
                elif orientation_type == "axis_angle":
                    quat = axis_angle_to_quaternion(action[3:6]) 
                elif orientation_type == "6D":
                    quat = matrix_to_quaternion(rotation_6d_to_matrix(action[3:9]))
                
                quat = normalize_quat(quat)
                # quat = standardize_quaternion(quat) # same result with ON or OFF
                quat = quat[..., [1,2,3,0]] # PyTorch3D (w,x,y,z) → Dataset (x,y,z,w) 

                part = torch.cat([pos, quat], dim=-1)

            if action_name == "ee_pose_relative":
                
                # Get relative poses
                pos = action[:3]
                ori_6d = action[3:9]
                R = rotation_6d_to_matrix(ori_6d)
                
                # Transform relative ee_pose in absolute ee_pose wrt last observed state
                part = get_absolute_pose_wrt_last_state(pos, R, p_base.to("cpu"), R_base.to("cpu")) # quaternion notation for orientation

            elif action_name == "gripper":
                part = action[-1:]

            # Add part to state vector
            action_parts.append(part)

        # Get final action with format as defined in the dataset
        action_pre_tf = torch.cat(action_parts, dim=-1).numpy()

        # Save estimated action from policy
        net_action_list.append(action_pre_tf)
        
        # Save action in buffer
        action_buffer.append(action_pre_tf)

    # Plotting

    checkpoint_name = os.path.basename(checkpoint_path)
    
    # Plot tracking errors

    real_actions = np.stack(real_action_list)   # (T, D)
    pred_actions = np.stack(net_action_list)    # (T, D)

    real_pos = real_actions[:, :3]
    pred_pos = pred_actions[:, :3]

    real_quat = real_actions[:, 3:7]
    pred_quat = pred_actions[:, 3:7]  

    real_gripper = real_actions[:, -1]
    pred_gripper_cont = pred_actions[:, -1]
    pred_gripper_disc = CustomTransforms.gripper_action_continuous2discrete(pred_gripper_cont) # convert gripper in binary {0,1}

    # Position error
    pos_error = pred_pos - real_pos
    pos_error_norm = np.linalg.norm(pos_error, axis=1)

    # Orientation error 
    ori_error_rad = quaternion_geodesic_error(torch.tensor(pred_quat), torch.tensor(real_quat))
    ori_error_rad = ori_error_rad.numpy()
    ori_error_deg = np.degrees(ori_error_rad)

    # Compute mean errors
    mean_pos_error = np.mean(pos_error_norm)
    mean_ori_error = np.mean(ori_error_deg)

    print(f"Mean position error: {mean_pos_error:.4f} m")
    print(f"Mean orientation error: {mean_ori_error:.2f} deg")

    accuracy = np.mean(pred_gripper_disc == real_gripper)
    print(f"Gripper accuracy: {accuracy*100:.2f} %")

    t = np.arange(len(pos_error_norm))

    # Plot pred vs real actions
    num_dims = pred_actions.shape[1]
    fig, axes = plt.subplots(num_dims, 1, figsize=(12, 2 * num_dims), sharex=True)

    for i in range(num_dims):
        axes[i].plot(real_actions[:, i], label="Dataset (Ground Truth)", color="blue", linestyle="--", alpha=0.7)
        axes[i].plot(pred_actions[:, i], label="Model (Prediction)", color="red", alpha=0.8)
        axes[i].set_ylabel(f"Dim {i}")
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend()

    # axes[-1].set_xlabel("Timesteps")
    plt.suptitle(f"Comparison Actions : Model vs Dataset\n")
    plt.tight_layout()
    plt.figtext(0.99, 0.01, f"Checkpoint: {checkpoint_name}", ha="right", va="bottom", fontsize=10, color="gray")
    plt.savefig(os.path.join(img_dir, "actions.jpeg"), format="jpeg", dpi=200)
    plt.show()
    plt.close()

    # Plot position error norm
    plt.figure(figsize=(10, 4))
    fig = plt.gcf()
    plt.plot(t, pos_error_norm, label=f"Mean: {mean_pos_error:.4f} m")
    plt.ylabel("||pos error|| [m]")
    plt.xlabel("Timestep")
    plt.grid(True)
    plt.title("Cartesian position error")
    plt.legend()
    plt.tight_layout()
    plt.figtext(0.99, 0.01, f"Checkpoint: {checkpoint_name}", ha="right", va="bottom", fontsize=10, color="gray")
    plt.savefig(os.path.join(img_dir, "pos_error.jpeg"), format="jpeg", dpi=200)
    plt.show()
    plt.close()

    # Plot orientation error
    plt.figure(figsize=(10, 4))
    plt.plot(t, ori_error_deg, label=f"Mean: {mean_ori_error:.2f}°")
    plt.ylabel("Orientation error [deg]")
    plt.xlabel("Timestep")
    plt.grid(True)
    plt.title("Orientation tracking error")
    plt.legend()
    plt.tight_layout()
    plt.figtext(0.99, 0.01, f"Checkpoint: {checkpoint_name}", ha="right", va="bottom", fontsize=10, color="gray")
    plt.savefig(os.path.join(img_dir, "ori_error.jpeg"), format="jpeg", dpi=200)
    plt.show()
    plt.close()

    # Plot gripper
    plt.figure(figsize=(10, 3))
    plt.step(t, real_gripper, where="post", label="Real", linewidth=2)
    plt.step(t, pred_gripper_disc, where="post", linestyle="--", label="Pred")
    plt.ylabel("Gripper")
    plt.xlabel("Timestep")
    plt.yticks([0, 1])
    plt.grid(True)
    plt.legend(title=f"Accuracy: {accuracy*100:.2f}%")
    plt.title("Gripper command tracking")
    plt.tight_layout()
    plt.figtext(0.99, 0.01, f"Checkpoint: {checkpoint_name}", ha="right", va="bottom", fontsize=10, color="gray")
    plt.savefig(os.path.join(img_dir, "gripper_tracking.jpeg"), format="jpeg", dpi=200)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()