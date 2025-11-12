import matplotlib.pyplot as plt
from pathlib import Path
import collections
import time
import torch
import csv
import os

from lerobot.configs.types import FeatureType
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from franka_ai.utils.seed_everything import seed_everything
from franka_ai.dataset.load_dataset import make_dataloader
from franka_ai.training.utils import setup_folders, update_best_model_symlink

# Set reproducibility
SEED = 40
seed_everything(SEED)


"""
Run the code: python src/franka_ai/training/train.py
Activate tensorboard: (from where there is this code): python -m tensorboard.main --logdir ../outputs/train/example_pusht_diffusion/tensorboard
"""


# TODO:
# 1) params da config file oppure da argparser? ne basta uno dei due?
# 2) optimize training (see notes)


def train(pretrained_path = None, learning_rate = 1e-4):

    """

    INSERT CLEAN DESCRIPTION

    If pretrained_path is None → train from scratch.
    If pretrained_path is provided → fine-tune from checkpoint.
    """

    # Get folders to save weights and tensorboard logs
    checkpoints_dir, tensorboard_dir, tsbrd_writer = setup_folders()

    # create CSV file to store training losses
    csv_path = os.path.join(checkpoints_dir, "loss_log.csv")
    with open(csv_path, mode="w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["step", "train_loss", "val_loss"])

    # Get parameters
    root_dataset_path = os.path.join(os.getcwd(), "data/test1")
    repo_id=None # "lerobot/pusht" # dataset folder name # magari preso in input da utente questo percorso? 
    # pretrained_path = None
    device = torch.device("cuda") # select your device
    training_steps = 12 # number of training steps [60000]
    log_freq = 2    # logs train loss, gradNorm, lr [200]
    eval_freq = 4   # logs eval loss [2000]
    save_ckpt_freq = 4 # 20000 # save checkpoints [20000]

    # consistency checks
    assert eval_freq % log_freq == 0, "eval_freq must be a multiple of log_freq"
    assert save_ckpt_freq % eval_freq == 0 and save_ckpt_freq >= eval_freq, "save_ckpt_freq must be >= eval_freq and a multiple of it"
    assert training_steps % save_ckpt_freq == 0, "training_steps must be a multiple of save_ckpt_freq to ensure final evaluation alignment"

    # Get dataset input/output stats
    dataset_metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=root_dataset_path)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features and key != 'observation.images.gripper_camera_depth'}

    print("NEW input_features: ", input_features)

    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. 
    cfg = DiffusionConfig(input_features=input_features, output_features=output_features)

    # -------------------
    # POLICY INITIALIZATION
    # -------------------

    if pretrained_path is None:
        # From scratch
        policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
    else:
        # Load from checkpoint
        policy = DiffusionPolicy.from_pretrained(pretrained_path)
        print(f"Loaded pretrained policy from {pretrained_path}")

    # Set training mode
    policy.train() # during training layers like Dropout or BatchNorm are ON 
    policy.to(device)

    # Dataloader
    train_loader, val_loader = make_dataloader(
        local_root=root_dataset_path,
        visual_obs_names=['observation.images.front_cam1', 'observation.images.front_cam2', 'observation.images.front_cam3', 'observation.images.gripper_camera'],
        device=device,
        seed_val=SEED,
        batch_size=16,
        N_history=2, 
        N_chunk=16,
        fps_sampling=10,
        print_ds_info=True
    )

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

    while not done:

        for batch in train_loader:
            
            # Sart timer
            step_start = time.perf_counter()

            # ove data to device
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            # computes the loss (and optionally predictions)
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
    
    # Example usage:
    # train()  # from scratch
    # train(pretrained_path="outputs/train/example_pusht_diffusion", learning_rate = 1e-5)  # fine-tune (lower learning rate)

    train()


