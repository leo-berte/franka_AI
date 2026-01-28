import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import argparse
import time
import torch
import csv
import os

from lerobot.configs.types import FeatureType
from lerobot.common.datasets.compute_stats import aggregate_stats

from torch.nn.utils import clip_grad_norm_

from franka_ai.utils.seed_everything import seed_everything
from franka_ai.dataset.load_dataset import make_dataloader
from franka_ai.dataset.transforms import CustomTransforms
from franka_ai.dataset.utils import get_configs_dataset, load_episodes_stats_patch
from franka_ai.training.utils import *
from franka_ai.models.utils import get_configs_models
from franka_ai.models.factory import get_policy_config_class, make_policy, get_policy_class

"""
Run the code: 

python src/franka_ai/training/train.py --dataset /mnt/Data/datasets/lerobot/one_bag \
                                       --config config \
                                       --policy act \
                                       --pretrained outputs/checkpoints/kinematics_tests/single_outliers_act_2025-12-25_21-08-42/best_model.pt

python src/franka_ai/training/train.py --dataset /workspace/data/single_outliers \
                                       --config config_full_state \
                                       --policy diffusion \
                                       --pretrained /workspace/outputs/checkpoints/....

Activate tensorboard (from where there is this code): 

tensorboard --logdir outputs/tensorboard
http://localhost:6006/#timeseries
"""


# TODO:

# 1) Check pre-training feature (i.e. When I add Fext in input features for example or new image)


 
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True,
                        help="Absolute path to dataset folder")

    parser.add_argument("--pretrained", type=str, default=None,
                        help="Absolute path to pretrained checkpoint")

    parser.add_argument("--policy", type=str, required=True,
                        choices=["diffusion", "act", "flowLeonardo", "template"],
                        help="Policy name")
    
    parser.add_argument("--config", type=str, default="config",
                    help="Config folder name")

    return parser.parse_args()


def train():

    """
        Train or fine-tune a policy on a LeRobot dataset using supervised learning.

        This function implements the full training pipeline, including:
        - Configuration loading for dataset, model and optimization
        - Dataset and DataLoader construction with temporal histories
        - Policy initialization from scratch or from a pretrained checkpoint
        - Loading dataset statistics for feature normalization
        - Automatic Mixed-Precision training with gradient clipping and LR scheduling
        - Periodic evaluation, checkpointing and logging

        Training behavior:
        - If `pretrained_path` is not provided, a new policy is initialized
            using dataset-derived input/output feature specifications and
            aggregated normalization statistics.
        - If `pretrained_path` is provided, the policy is loaded and fine-tuned
            from the given checkpoint.

        Optimization details:
        - Uses AdamW with warmup followed by cosine decay
        - Supports automatic mixed precision (AMP) on CUDA devices
        - Logs training and validation metrics to TensorBoard and CSV
        - Saves periodic checkpoints and tracks the best validation model
        """

    # -------------------
    # INIT FOLDERS & ARGS
    # -------------------

    # Extract command line args
    args = parse_args()
    dataset_path = args.dataset
    pretrained_path = args.pretrained
    policy_name = args.policy
    config_folder = args.config

    # Load configs
    dataloader_cfg, dataset_cfg, transforms_cfg = get_configs_dataset(f"configs/{config_folder}/dataset.yaml")
    train_cfg, policies_cfg = get_configs_training(f"configs/{config_folder}/train.yaml")
    models_cfg = get_configs_models(f"configs/{config_folder}/models.yaml")
    policy_cfg = policies_cfg[policy_name]
    model_cfg = models_cfg[policy_name]

    # Get folders to save weights and tensorboard logs
    checkpoints_dir, tensorboard_dir, tsbrd_writer = set_output_folders_train(policy_name, dataset_path, config_folder)

    # create CSV file to store training losses
    csv_path = os.path.join(checkpoints_dir, "loss_log.csv")
    with open(csv_path, mode="w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["step", "train_loss", "val_loss"])

    # Set parameters from config
    device          = torch.device(dataloader_cfg["device"])
    seed_val        = dataloader_cfg["seed_val"]
    training_steps  = train_cfg["training_steps"]    # number of training steps
    log_freq        = train_cfg["log_freq"]          # logs train loss, gradNorm, lr
    eval_freq       = train_cfg["eval_freq"]         # logs eval loss
    save_ckpt_freq  = train_cfg["save_ckpt_freq"]    # save checkpoints
    learning_rate   = policy_cfg["optimizer"]["learning_rate"] 
    weight_decay    = policy_cfg["optimizer"]["weight_decay"]
    lr_warmup_steps = policy_cfg["scheduler"]["lr_warmup_steps"]

    # Consistency checks
    if eval_freq % log_freq != 0:
        raise ValueError("eval_freq must be a multiple of log_freq")
    if save_ckpt_freq < eval_freq or save_ckpt_freq % eval_freq != 0:
        raise ValueError("save_ckpt_freq must be >= eval_freq and a multiple of it")
    if training_steps % save_ckpt_freq != 0:
        raise ValueError("training_steps must be a multiple of save_ckpt_freq to ensure final evaluation alignment")
    if model_cfg["sampling"]["fps_sampling_hist"] != model_cfg["sampling"]["fps_sampling_chunk"]:
        raise ValueError("fps_sampling_hist must be the same as fps_sampling_chunk.")

    # Eventually freeze seed for reproducibility
    if seed_val is not None and seed_val >= 0:
        seed_everything(seed_val)

    # Prepare transforms for training, inference and for computing dataset stats
    transforms_train = CustomTransforms(dataset_cfg, transforms_cfg, model_cfg, train=True)
    transforms_val = CustomTransforms(dataset_cfg, transforms_cfg, model_cfg, train=False)

    # Create loaders
    train_loader, train_ep, val_loader, val_ep = make_dataloader(
        dataset_path=dataset_path,
        dataloader_cfg=dataloader_cfg,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        selected_episodes=[0]
    )

    # ---------------------
    # POLICY INITIALIZATION
    # ---------------------
    
    if pretrained_path:

        # Load from checkpoint
        PolicyClass = get_policy_class(policy_name)
        policy = PolicyClass.from_pretrained(pretrained_path)
        print(f"Loaded pretrained policy from {pretrained_path}")

    else:
    
        # Get dataset input/output stats
        features = dataset_to_policy_features_patch(train_loader, dataset_cfg["features"], transforms_train)
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if ft.type is not FeatureType.ACTION}
        print("New output_features: ", output_features)
        print("New input_features: ", input_features)
        
        # Define normalization and unnormalization mode 
        normalization_mapping = parse_normalization_mapping(policy_cfg["normalization"])
        
        # Get policy configuration class
        ConfigClass = get_policy_config_class(policy_name)

        cfg = ConfigClass(
            input_features=input_features,
            output_features=output_features,
            normalization_mapping=normalization_mapping,
            **model_cfg["params"]
        )

        # Get episodes stats
        episode_stats_path = Path(f"configs/{config_folder}") / "episodes_stats_transformed.jsonl"
        episodes_stats = load_episodes_stats_patch(episode_stats_path)
        # Aggregate episodes stats in a unique global stats
        new_dataset_stats = aggregate_stats([episodes_stats[ep] for ep in train_ep])
        # Save transformed stats in checkpoint dir as backup
        dst_path = Path(checkpoints_dir) / "episodes_stats_transformed.jsonl"
        shutil.copy(episode_stats_path, dst_path)
        
        # Setup policy
        policy = make_policy(policy_name, cfg, dataset_stats=new_dataset_stats)

    # Set training mode
    policy.train() # during training layers like Dropout or BatchNorm are ON 
    policy.to(device)

    # Activate Automatic Mixed Precision (AMP): forward/backward in FP16, loss/optimizer/weights in FP32
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    # Optimizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Linear LR scheduler for warmup (from start_factor*lr to lr)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=lr_warmup_steps)
    # Cosine LR scheduler after warmup (from lr to eta_min)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_steps - lr_warmup_steps, eta_min=learning_rate * 0.5)
    # Sequential LR scheduler
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[lr_warmup_steps])

    # -------------
    # TRAINING LOOP
    # -------------

    step = 0
    done = False
    train_losses, val_losses = [], []
    train_steps, val_steps = [], []
    running_train_loss = 0.0
    running_time_sum = 0.0
    running_grad_norm_sum = 0.0
    best_val_loss = float("inf")
    avg_val_loss = float("inf")

    print("\nStarting training loop...\n")

    while not done:

        for batch in train_loader:

            # print("transform", {k:v.shape for k,v in batch.items() if isinstance(v, torch.Tensor)})
        
            # Sart timer
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_start = time.perf_counter()

            # Move data to device
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}      
            
            # Apply custom transforms
            batch = transforms_train.transform(batch) # out: (B, N_h, ...)

            # print("transform", {k:v.shape for k,v in batch.items() if isinstance(v, torch.Tensor)})

            # Use Automatic Mixed Precision (AMP)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                loss, _ = policy.forward(batch)

            # Scale loss (multiply by bigger number) to avoid gradients computation (g'=dL'/dW) is zero
            scaler.scale(loss).backward() 

            # Unscale gradients before clipping (g=g'/s)
            scaler.unscale_(optimizer)

            # Prevent exploding gradients
            total_norm = clip_grad_norm_(policy.parameters(), max_norm=1.0)
            running_grad_norm_sum += total_norm

            # Update model weights + adapt scaling number
            scaler.step(optimizer)
            scaler.update()

            # Update learning rate + reset gradients
            scheduler.step()
            optimizer.zero_grad()

            # Stop and store timer
            if device.type == "cuda":
                torch.cuda.synchronize()
            running_time_sum +=  time.perf_counter() - step_start
  
            # Accumulate training loss for averaging
            running_train_loss += loss.item()

            # Log metrics
            if step > 0 and step % log_freq == 0:

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
            if step > 0 and step % eval_freq == 0:

                # Run evaluation on validation set
                policy.eval()
                val_loss_total = 0.0

                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in val_batch.items()}
                        val_batch = transforms_val.transform(val_batch) 
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
                policy.save_pretrained(ckpt_path)

                # Update symlink if it's the new best
                best_val_loss = update_best_model_symlink(checkpoints_dir, ckpt_path, best_val_loss, avg_val_loss, step)

            # Increment step
            step += 1

            if step > training_steps:
                done = True
                break
    
    # Skip steps < lr_warmup_steps for plotting losses
    train_steps_f, train_losses_f = zip(*[(s, l) for s, l in zip(train_steps, train_losses) if s > lr_warmup_steps])
    val_steps_f, val_losses_f = zip(*[(s, l) for s, l in zip(val_steps, val_losses) if s > lr_warmup_steps])

    # Plot losses
    plt.figure(figsize=(8, 5))
    plt.plot(train_steps_f, train_losses_f, label="Train Loss")
    plt.plot(val_steps_f, val_losses_f, label="Validation Loss")
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