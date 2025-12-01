import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time
import torch
import csv
import os

from lerobot.configs.types import FeatureType
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.datasets.compute_stats import aggregate_stats

from torch.nn.utils import clip_grad_norm_

from franka_ai.utils.seed_everything import seed_everything
from franka_ai.dataset.load_dataset import make_dataloader
from franka_ai.dataset.transforms import CustomTransforms
from franka_ai.dataset.utils import get_configs_dataset, load_episodes_stats_patch
from franka_ai.training.utils import *

"""
Run the code: 

python src/franka_ai/training/train.py --dataset /home/leonardo/Documents/Coding/franka_AI/data/today_data/today_outliers
                                                     --pretrained /home/leonardo/Documents/Coding/franka_AI/outputs/checkpoints
                                                     --policy ACT

python src/franka_ai/training/train.py --dataset /workspace/data/today_data/today_outliers
                                       --pretrained /workspace/outputs/checkpoints
                                       --policy ACT

Activate tensorboard (from where there is this code): 

python -m tensorboard.main --logdir ../outputs/train/example_pusht_diffusion/tensorboard
"""


# TODO:
# 1) Optimize training (see notes)
# 2) Add ACT, DP, Mine policyFactory
# Handle correctly pre-training (i.e. I add Fext in input features for example)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True,
                        help="Absolute path to dataset folder")

    parser.add_argument("--pretrained", type=str, default=None,
                        help="Absolute path to pretrained checkpoint")

    parser.add_argument("--policy", type=str, default="DP",
                        choices=["DP", "ACT"],
                        help="Policy type")

    return parser.parse_args()


def train():

    """

    INSERT CLEAN DESCRIPTION

    If pretrained_path is None → train from scratch.
    If pretrained_path is provided → fine-tune from checkpoint.
    """

    # -------------------
    # INIT FOLDERS & ARGS
    # -------------------

    # Extract command line args
    args = parse_args()
    dataset_path = args.dataset
    pretrained_path = args.pretrained
    policy_type = args.policy

    # Load configs
    dataloader_cfg, dataset_cfg, transforms_cfg = get_configs_dataset("configs/dataset.yaml")
    train_cfg, normalization_cfg = get_configs_training("configs/train.yaml")

    # Get folders to save weights and tensorboard logs
    checkpoints_dir, tensorboard_dir, tsbrd_writer = set_output_folders_train(policy_type, dataset_path)

    # create CSV file to store training losses
    csv_path = os.path.join(checkpoints_dir, "loss_log.csv")
    with open(csv_path, mode="w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["step", "train_loss", "val_loss"])

    # Set parameters from config
    device = torch.device(dataloader_cfg["device"])
    seed_val = dataloader_cfg["seed_val"]
    training_steps = train_cfg["training_steps"] # number of training steps [60000]
    log_freq = train_cfg["log_freq"] # logs train loss, gradNorm, lr [200]
    eval_freq = train_cfg["eval_freq"] # logs eval loss [2000]
    save_ckpt_freq = train_cfg["save_ckpt_freq"] # 20000 # save checkpoints [20000]
    learning_rate = train_cfg["learning_rate"]

    # Consistency checks
    if eval_freq % log_freq != 0:
        raise ValueError("eval_freq must be a multiple of log_freq")
    if save_ckpt_freq < eval_freq or save_ckpt_freq % eval_freq != 0:
        raise ValueError("save_ckpt_freq must be >= eval_freq and a multiple of it")
    if training_steps % save_ckpt_freq != 0:
        raise ValueError("training_steps must be a multiple of save_ckpt_freq to ensure final evaluation alignment")

    # Eventually freeze seed for reproducibility
    if seed_val is not None and seed_val >= 0:
        seed_everything(seed_val)

    # Prepare transforms for training, inference and for computing dataset stats
    transforms_train = CustomTransforms(dataloader_cfg, dataset_cfg, transforms_cfg, train=True) 
    transforms_val = CustomTransforms(dataloader_cfg, dataset_cfg, transforms_cfg, train=False)

    # Create loaders
    train_loader, train_ep, val_loader, val_ep = make_dataloader(
        dataset_path=dataset_path,
        dataloader_cfg=dataloader_cfg,
        feature_groups=dataset_cfg["features"],
        transforms_train=transforms_train,
        transforms_val=transforms_val
    )

    # ---------------------
    # POLICY INITIALIZATION
    # ---------------------

    if pretrained_path:

        # Load from checkpoint
        policy = DiffusionPolicy.from_pretrained(pretrained_path)
        print(f"Loaded pretrained policy from {pretrained_path}")

    else:

        # Get dataset input/output stats
        features = dataset_to_policy_features_patch(train_loader, dataset_cfg["features"])
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if ft.type is not FeatureType.ACTION}
        print("New output_features: ", output_features)
        print("New input_features: ", input_features)
        
        # Define normalization and unnormalization mode 
        normalization_mapping = parse_normalization_mapping(normalization_cfg)

        # Policies are initialized with a configuration class, in this case `DiffusionConfig`. 
        cfg = DiffusionConfig(input_features=input_features, 
                            output_features=output_features, 
                            normalization_mapping=normalization_mapping,
                            n_obs_steps=dataloader_cfg["N_history"],
                            horizon=dataloader_cfg["N_chunk"])

        # Get episodes stats
        episode_stats_path = Path(dataset_path) / "meta" / "episodes_stats_transformed.jsonl"
        episodes_stats = load_episodes_stats_patch(episode_stats_path)
        # Aggregate episodes stats in a unique global stats
        new_dataset_stats = aggregate_stats([episodes_stats[ep] for ep in train_ep])
        # Setup policy
        policy = DiffusionPolicy(cfg, dataset_stats=new_dataset_stats)

    # Set training mode
    policy.train() # during training layers like Dropout or BatchNorm are ON 
    policy.to(device)

    # Then we create our optimizer and dataloader for offline training.
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate) 

    # -------------------
    # TRAINING LOOP
    # -------------------

    step = 0
    done = False
    train_losses, val_losses = [], []
    train_steps, val_steps = [], []
    running_train_loss = 0.0
    running_time_sum = 0.0
    running_grad_norm_sum = 0.0
    best_val_loss = float("inf")
    avg_val_loss = float("inf")

    print("Starting training loop...")

    while not done:

        for batch in train_loader:
            
            # Sart timer
            step_start = time.perf_counter()

            # Move data to device
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            # Computes the loss (and optionally predictions)
            loss, _ = policy.forward(batch)

            loss.backward() # compute gradients
            running_grad_norm_sum +=clip_grad_norm_(policy.parameters(), max_norm=float("inf")).item() # measure gradients norm
            optimizer.step() # do gradient step
            optimizer.zero_grad()
            
            # Stop and store timer
            running_time_sum +=  time.perf_counter() - step_start
  
            # Accumulate training loss for averaging
            running_train_loss += loss.item()

            # Log metrics
            if step % log_freq == 0:

                # Compute metrics
                avg_step_time = running_time_sum / log_freq
                avg_grad_norm = running_grad_norm_sum / log_freq
                avg_train_loss = running_train_loss / log_freq
                train_losses.append(avg_train_loss)
                train_steps.append(step)
                running_time_sum = 0.0
                running_grad_norm_sum = 0.0
                running_train_loss = 0.0

                # Log to TensorBoard
                tsbrd_writer.add_scalar("Loss/train", avg_train_loss, step)
                tsbrd_writer.add_scalar("Metrics/grad_norm", avg_grad_norm, step) # gradients magnitude
                tsbrd_writer.add_scalar("Metrics/step_time_sec", avg_step_time, step) # latency for processing 1 batch (dataloader, forward + backward pass, optimizer update)
                tsbrd_writer.add_scalar("Metrics/lr", optimizer.param_groups[0]["lr"], step) # learning rate
            
            # Log eval loss
            if step % eval_freq == 0:

                # Run evaluation on validation set
                policy.eval()
                val_loss_total = 0.0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in val_batch.items()}
                        val_loss, _ = policy.forward(val_batch)
                        val_loss_total += val_loss.item()
                avg_val_loss = val_loss_total / len(val_loader)
                val_losses.append(avg_val_loss)
                val_steps.append(step)
                print(f"step: {step} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | grad_norm={avg_grad_norm:.4f} | step_time_sec={avg_step_time:.4f}")
                policy.train()

                # Log to TensorBoard
                tsbrd_writer.add_scalar("Loss/val", avg_val_loss, step)

                # append to CSV
                with open(csv_path, mode="a", newline="") as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([step, avg_train_loss, avg_val_loss])

            # Save checkpoints
            if step > 0 and step % save_ckpt_freq == 0: 

                # Save checkpoint
                ckpt_path = os.path.join(checkpoints_dir, f"step_{step:08d}.pt")
                # torch.save(policy.state_dict(), ckpt_path)
                policy.save_pretrained(ckpt_path)

                # Update symlink if it's the new best
                best_val_loss = update_best_model_symlink(checkpoints_dir, ckpt_path, best_val_loss, avg_val_loss, step)

            step += 1
            if step > training_steps:
                done = True
                break

    # Plot losses
    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_losses, label="Train Loss")
    plt.plot(val_steps, val_losses, label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(checkpoints_dir, "loss_curve.png")
    plt.savefig(plot_path, dpi=200)
    print(f"Saved training curve at: {plot_path}")
    plt.close()


if __name__ == "__main__":
    
    train()