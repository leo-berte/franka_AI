import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from franka_ai.dataset.transforms import CustomTransforms
from franka_ai.dataset.load_dataset import make_dataloader
from franka_ai.dataset.utils import get_configs_dataset
from franka_ai.models.utils import get_configs_models
from franka_ai.models.factory import get_policy_class


"""
Run the code: 

python src/franka_ai/inference/evaluate.py --dataset /home/leonardo/Documents/Coding/franka_AI/data/single/single_outliers \
                                           --checkpoint outputs/checkpoints/single_outliers_diffusion_2025-12-14_10-38-20 \
                                           --policy diffusion 

python src/franka_ai/inference/evaluate.py --dataset /home/leonardo/Documents/Coding/franka_AI/data/single/single_outliers \
                                           --checkpoint outputs/checkpoints/single_outliers_diffusion_2025-12-14_12-14-34 \
                                           --policy diffusion 

python src/franka_ai/inference/evaluate.py --dataset /home/leonardo/Documents/Coding/franka_AI/data/single/single_outliers \
                                           --checkpoint outputs/checkpoints/single_outliers_diffusion_2025-12-14_18-43-48 \
                                           --policy diffusion

python src/franka_ai/inference/evaluate.py --dataset /home/leonardo/Documents/Coding/franka_AI/data/single/single_outliers \
                                           --checkpoint outputs/checkpoints/single_outliers_diffusion_2025-12-14_20-04-38 \
                                           --policy diffusion
                                
python src/franka_ai/inference/evaluate.py --dataset /home/leonardo/Documents/Coding/franka_AI/data/single/single_outliers \
                                           --checkpoint outputs/checkpoints/single_outliers_diffusion_2025-12-14_21-21-50 \
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
                    choices=["diffusion", "act"],
                    help="Policy name")
    
    args = parser.parse_args()

    # return args
    return args.dataset, args.checkpoint, args.policy

def quat_angle_error(q1, q2):
    """
    q1, q2: (..., 4) quaternion [x,y,z,w]
    ritorna errore angolare in radianti
    """
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)

    dot = np.sum(q1 * q2, axis=-1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)

    return 2 * np.arccos(dot)

def main():

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
    include_states = transforms_cfg["state"]["include"]
    include_gripper = "gripper" in include_states

    # Consistency checks
    # if transforms_cfg["state"]["use_past_actions"] == True:
    #     raise ValueError("Not implemented yet the possibility to use past actions in state.")

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
    train_loader, _, val_loader, _ = make_dataloader(
        dataset_path=dataset_path,
        dataloader_cfg=dataloader_cfg,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg
    )

    # Save data
    real_action_list = []
    net_action_list = []

    # iterate over dataloader
    for step, batch in enumerate(train_loader):
        
        print("step: ", step)

        # Move data to device
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}      

        # Save real action from dataset
        real_action = batch["action"][:, N_history, ...].to("cpu").numpy() # (B, D)
        real_action_list.append(real_action.squeeze(0))  

        # Apply custom transforms
        batch = tf_inference.transform(batch)

        # Convert (B,N_hist, D) in (B, D) by taking last timestep
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 3:
                v = v[:, -1, ...].contiguous()
                batch[k] = v

        # Remove "action" from observations
        batch.pop("action")

        # Inference
        with torch.inference_mode():
            action = policy.select_action(batch) # (B, D) --> (B, D)
            # actions = self.policy.diffusion.generate_actions(obs) # (B, N_hist, D) --> (N_chunk, D)

        # Move to CPU
        action = action.squeeze(0).to("cpu")

        # Convert axis-angle to quaternion
        quat = CustomTransforms.axis_angle2quaternion(action[3:6])

        # Convert gripper in binary {0,1}
        if (include_gripper):
            action[-1] = CustomTransforms.gripper_continuous2discrete(action[-1])

        # Convert tensors to numpy
        action_np = action.numpy()
        quat_np = quat.numpy()

        # Build final action as expected by controller
        if (include_gripper):
            action_pre_tf = np.concatenate([action_np[:3], quat_np, [action_np[-1]]])
        else:
            action_pre_tf = np.concatenate([action_np[:3], quat_np])

        # Save estimated action from policy
        net_action_list.append(action_pre_tf)
        
        # # Save action in buffer safely --> con dataloader se uso past actions = true come faccio? input policy è batch dataloader, devo mettere output policy
        # with self.buffer_lock:
        #     self.buffers["action"].append((self.get_clock().now().to_msg(), action_pre_tf))

    # Plot tracking errors

    real_actions = np.stack(real_action_list)   # (T, D)
    pred_actions = np.stack(net_action_list)    # (T, D)

    real_pos = real_actions[:, :3]
    pred_pos = pred_actions[:, :3]

    real_quat = real_actions[:, 3:7]
    pred_quat = pred_actions[:, 3:7]

    if include_gripper:
        real_gripper = real_actions[:, -1]
        pred_gripper = pred_actions[:, -1]

    # Position error
    pos_error = pred_pos - real_pos
    pos_error_norm = np.linalg.norm(pos_error, axis=1)

    # Orientation error 
    ori_error_rad = quat_angle_error(pred_quat, real_quat)
    ori_error_deg = np.degrees(ori_error_rad)

    # Compute mean errors
    mean_pos_error = np.mean(pos_error_norm)
    mean_ori_error = np.mean(ori_error_deg)

    print(f"Mean position error: {mean_pos_error:.4f} m")
    print(f"Mean orientation error: {mean_ori_error:.2f} deg")

    if include_gripper:
        accuracy = np.mean(pred_gripper == real_gripper)
        print(f"Gripper accuracy: {accuracy*100:.2f} %")

    t = np.arange(len(pos_error_norm))

    # Plot XYZ
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["x", "y", "z"]

    for i in range(3):
        axs[i].plot(t, real_pos[:, i], label="Real")
        axs[i].plot(t, pred_pos[:, i], "--", label="Pred")
        axs[i].set_ylabel(f"{labels[i]} [m]")
        axs[i].grid(True)
        axs[i].legend()

    axs[-1].set_xlabel("Timestep")
    fig.suptitle("End-effector position tracking")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "xyz_tracking.jpeg"), format="jpeg", dpi=200)
    plt.show()
    plt.close()

    # Plot position error norm
    plt.figure(figsize=(10, 4))
    plt.plot(t, pos_error_norm, label=f"Mean: {mean_pos_error:.4f} m")
    plt.ylabel("||pos error|| [m]")
    plt.xlabel("Timestep")
    plt.grid(True)
    plt.title("Cartesian position error")
    plt.legend()
    plt.tight_layout()
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
    plt.savefig(os.path.join(img_dir, "ori_error.jpeg"), format="jpeg", dpi=200)
    plt.show()
    plt.close()

    # Plot gripper
    if include_gripper:
        plt.figure(figsize=(10, 3))
        plt.step(t, real_gripper, where="post", label="Real", linewidth=2)
        plt.step(t, pred_gripper, where="post", linestyle="--", label="Pred")
        plt.ylabel("Gripper")
        plt.xlabel("Timestep")
        plt.yticks([0, 1])
        plt.grid(True)
        plt.legend(title=f"Accuracy: {accuracy*100:.2f}%")
        plt.title("Gripper command tracking")
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, "gripper_tracking.jpeg"), format="jpeg", dpi=200)
        plt.show()
        plt.close()



if __name__ == "__main__":
    main()








