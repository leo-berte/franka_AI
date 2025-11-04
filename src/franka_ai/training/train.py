import matplotlib.pyplot as plt
from pathlib import Path
import collections
import time
import torch
import csv
import os

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from franka_ai.utils.seed_everything import seed_everything
from franka_ai.datasets.load_dataset import make_dataloader
from franka_ai.training.utils import setup_folders, update_best_model_symlink

# Set reproducibility
SEED = 40
seed_everything(SEED)


"""
Run the code: python src/franka_ai/training/train.py
Activate tensorboard: (from where there is this code): python -m tensorboard.main --logdir ../outputs/train/example_pusht_diffusion/tensorboard
"""


# TODO:
# 0) define metrics to measure how much training / inference code is efficient (measure time, memory, ..)
# 1) params da config file
# 2) optimize training (see notes)



# System / Efficiency Metrics

# For understanding how efficiently training runs:

# Metric	Why it matters
# GPU utilization (%)	To ensure data pipeline isn’t the bottleneck.
# GPU memory usage	To detect leaks or overly large batches.
# CPU utilization	For DataLoader tuning.
# Dataset loading time per batch	See if num_workers or transforms are slowing you down.


# train_metrics = {
#     "loss": AverageMeter("loss", ":.3f"),
#     "grad_norm": AverageMeter("grdn", ":.3f"),
#     "lr": AverageMeter("lr", ":0.1e"),
#     "update_s": AverageMeter("updt_s", ":.3f"),
#     "dataloading_s": AverageMeter("data_s", ":.3f"),





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
    # root_datasets_path = "temp" # magari preso in input da utente questo percorso? 
    dataset_id="lerobot/pusht" # dataset folder name # magari preso in input da utente questo percorso? 
    device = torch.device("cuda") # select your device
    training_steps = 10 # number of training steps 
    save_ckpt_freq = 3

    # Get dataset input/output stats
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. 
    cfg = DiffusionConfig(input_features=input_features, output_features=output_features)

    # -------------------
    # POLICY INITIALIZATION
    # -------------------

    if pretrained_path is None:
        # From scratch
        policy = DiffusionPolicy(cfg)
    else:
        # Load from checkpoint
        policy = DiffusionPolicy.from_pretrained(pretrained_path)
        print(f"Loaded pretrained policy from {pretrained_path}")

    # Set training mode
    policy.train() # during training layers like Dropout or BatchNorm are ON 
    policy.to(device)

    # Get normalization stats (lerobot simply compute mean, std_dev for each feature [q1, q2, ..] over all frame in the dataset)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    # Dataloader
    train_loader, val_loader = make_dataloader(
        repo_id=dataset_id,
        device=device,
        seed_val=SEED,
        batch_size=16,
        N_history=2, 
        N_chunk=16,
        fps=10,
        print_ds_info=True
    )

    # Then we create our optimizer and dataloader for offline training.
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate) 

    # -------------------
    # TRAINING LOOP
    # -------------------

    step = 0
    done = False
    train_losses = []
    val_losses = []
    running_train_loss = 0.0
    best_val_loss = float("inf")

    time_window = collections.deque(maxlen=50)
    grad_norm_window = collections.deque(maxlen=50)

    while not done:

        for batch in train_loader:
            
            # ???
            step_start = time.perf_counter()

            # batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            # batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # ???
            batch = preprocessor(batch)

            # computes the loss (and optionally predictions)
            loss, _ = policy.forward(batch)

            loss.backward() # compute gradients
            grad_norm_window.append(clip_grad_norm_(policy.parameters(), max_norm=float("inf")).item()) # measure gradients norm
            optimizer.step() # do gradient step
            optimizer.zero_grad()
            
            # ???
            time_window.append(time.perf_counter() - step_start)

            # Accumulate training loss for averaging
            running_train_loss += loss.item()

            # Compute averaged train loss + validation loss periodically
            if step % save_ckpt_freq == 0:
                avg_train_loss = running_train_loss / save_ckpt_freq
                train_losses.append(avg_train_loss)
                running_train_loss = 0.0

                # Run evaluation on validation set
                policy.eval()
                val_loss_total = 0.0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = preprocessor(val_batch)
                        val_loss, _ = policy.forward(val_batch)
                        val_loss_total += val_loss.item()
                avg_val_loss = val_loss_total / len(val_loader)
                val_losses.append(avg_val_loss)
                print(f"step: {step} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}")
                policy.train()

                # Save checkpoint
                ckpt_path = os.path.join(checkpoints_dir, f"step_{step:08d}.pt")
                torch.save(policy.state_dict(), ckpt_path)

                # Update symlink if it's the new best
                best_val_loss = update_best_model_symlink(checkpoints_dir, ckpt_path, best_val_loss, avg_val_loss, step)

                # append to CSV
                with open(csv_path, mode="a", newline="") as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([step, avg_train_loss, avg_val_loss])
                
                # Compute metrics
                avg_step_time = sum(time_window) / len(time_window)
                avg_grad_norm = sum(grad_norm_window) / len(grad_norm_window)

                # Log to TensorBoard
                tsbrd_writer.add_scalar("Loss/train", avg_train_loss, step)
                tsbrd_writer.add_scalar("Loss/val", avg_val_loss, step)
                tsbrd_writer.add_scalar("Metrics/grad_norm", avg_grad_norm, step) # gradients magnitude
                tsbrd_writer.add_scalar("Metrics/step_time_sec", avg_step_time, step) # latency for processing 1 batch (dataloader, forward + backward pass, optimizer update)
                tsbrd_writer.add_scalar("Metrics/lr", optimizer.param_groups[0]["lr"], step) # learning rate
            
            step += 1
            if step >= training_steps:
                done = True
                break

    # Save final weights
    final_ckpt = os.path.join(checkpoints_dir, f"step_{step:08d}_final.pt")
    torch.save(policy.state_dict(), final_ckpt)
    # dump also config policy file to have policy params --> la nostra policy avrà config_file e config_struct? anche ACT, DP hannpo queste cose?
    best_val_loss = update_best_model_symlink(checkpoints_dir, ckpt_path, best_val_loss, avg_val_loss, step)
    print(f"Final checkpoint saved at: {final_ckpt}")
    
    # HF style
    # policy.save_pretrained(checkpoints_dir)
    # preprocessor.save_pretrained(checkpoints_dir)
    # postprocessor.save_pretrained(checkpoints_dir)

    # Plot losses
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(train_losses)), train_losses, label="Train Loss")
    plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
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


