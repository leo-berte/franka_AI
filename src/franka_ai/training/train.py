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

python src/franka_ai/training/train.py --dataset /home/leonardo/Documents/Coding/franka_AI/data/single_outliers
                                                     --pretrained /home/leonardo/Documents/Coding/franka_AI/outputs/checkpoints/.....
                                                     --config config1
                                                     --policy diffusion

python src/franka_ai/training/train.py --dataset /workspace/data/single_outliers
                                       --pretrained /workspace/outputs/checkpoints/....
                                       --config config1
                                       --policy diffusion

Activate tensorboard (from where there is this code): 

tensorboard --logdir outputs/tensorboard
http://localhost:6006/#timeseries
"""


# TODO:


# train on SINGLE dataset --> STATE only --> 1 episode reply
# train on SINGLE dataset --> full dataset for single bag pick & place


# try ACT

# 1) Handle correctly pre-training (i.e. I add Fext in input features for example)
# 2) Optimize training (see notes




def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True,
                        help="Absolute path to dataset folder")

    parser.add_argument("--pretrained", type=str, default=None,
                        help="Absolute path to pretrained checkpoint")

    parser.add_argument("--policy", type=str, required=True,
                        choices=["diffusion", "act"],
                        help="Policy name")
    
    parser.add_argument("--config", type=str, default="config",
                    help="Config folder name")

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
    policy_name = args.policy
    config_folder = args.config

    # Load configs
    dataloader_cfg, dataset_cfg, transforms_cfg = get_configs_dataset(f"configs/{config_folder}/dataset.yaml")
    train_cfg, normalization_cfg = get_configs_training(f"configs/{config_folder}/train.yaml")
    models_cfg = get_configs_models(f"configs/{config_folder}/models.yaml")

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
    learning_rate   = train_cfg["learning_rate"] 
    lr_warmup_steps = train_cfg["lr_warmup_steps"]
    weight_decay    = train_cfg["weight_decay"]

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
    transforms_train = CustomTransforms(dataset_cfg, transforms_cfg, models_cfg[policy_name], train=True) 
    transforms_val = CustomTransforms(dataset_cfg, transforms_cfg, models_cfg[policy_name], train=False)

    # Create loaders
    train_loader, train_ep, val_loader, val_ep = make_dataloader(
        dataset_path=dataset_path,
        dataloader_cfg=dataloader_cfg,
        dataset_cfg=dataset_cfg,
        model_cfg=models_cfg[policy_name],
        transforms_train=transforms_train,
        transforms_val=transforms_val
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
        features = dataset_to_policy_features_patch(train_loader, dataset_cfg["features"])
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if ft.type is not FeatureType.ACTION}
        print("New output_features: ", output_features)
        print("New input_features: ", input_features)
        
        # Define normalization and unnormalization mode 
        normalization_mapping = parse_normalization_mapping(normalization_cfg)

        # Get policy configuration class
        ConfigClass = get_policy_config_class(policy_name)

        cfg = ConfigClass(
            input_features=input_features,
            output_features=output_features,
            normalization_mapping=normalization_mapping,
            **models_cfg[policy_name]["params"]
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

    # Optimizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)

    # Linear LR scheduler for warmup (from start_factor*lr to lr)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=lr_warmup_steps)
    # Cosine LR scheduler after warmup (from lr to eta_min)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_steps - lr_warmup_steps, eta_min=learning_rate * 0.1)
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
            
            # Sart timer
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_start = time.perf_counter()

            # Move data to device
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}      

            # Computes the loss (and optionally predictions)
            loss, _ = policy.forward(batch)

            loss.backward() # compute gradients
            total_norm = clip_grad_norm_(policy.parameters(), max_norm=1.0) # measure gradients norm and clip it
            running_grad_norm_sum += total_norm
            optimizer.step() # do gradient step
            scheduler.step()
            optimizer.zero_grad()
            
            # Stop and store timer
            if device.type == "cuda":
                torch.cuda.synchronize()
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
                policy.save_pretrained(ckpt_path)

                # Update symlink if it's the new best
                best_val_loss = update_best_model_symlink(checkpoints_dir, ckpt_path, best_val_loss, avg_val_loss, step)

            # Increment step
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



# ## DP

# def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:

#     """Run the batch through the model and compute the loss for training or validation."""

#     batch = self.normalize_inputs(batch)
#     if self.config.image_features:
#         batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
#         batch["observation.images"] = torch.stack(
#             [batch[key] for key in self.config.image_features], dim=-4
#         )
#     batch = self.normalize_targets(batch)
#     loss = self.diffusion.compute_loss(batch)

#     return loss, None

# def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:

#     """
#     This function expects `batch` to have (at least):
#     {
#         "observation.state": (B, n_obs_steps, state_dim)

#         "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
#             AND/OR
#         "observation.environment_state": (B, environment_dim)

#         "action": (B, horizon, action_dim)
#     """



# ## ACT

#     def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:

#         """Run the batch through the model and compute the loss for training or validation."""

#         batch = self.normalize_inputs(batch)
#         if self.config.image_features:
#             batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
#             batch["observation.images"] = [batch[key] for key in self.config.image_features]

#         batch = self.normalize_targets(batch)
#         actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

#         l1_loss = (
#             F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
#         ).mean()

#         loss_dict = {"l1_loss": l1_loss.item()}

#         if self.config.use_vae:
#             loss_dict["kld_loss"] = mean_kld.item()
#             loss = l1_loss + mean_kld * self.config.kl_weight
#         else:
#             loss = l1_loss

#         return loss, loss_dict
    

#     def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:

#         """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

#         `batch` should have the following structure:
#         {
#             [robot_state_feature] (optional): (B, state_dim) batch of robot states.

#             [image_features]: (B, n_cameras, C, H, W) batch of images.
#                 AND/OR
#             [env_state_feature]: (B, env_dim) batch of environment states.

#             [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
#         }

#         Returns:
#             (B, chunk_size, action_dim) batch of action sequences
#             Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
#             latent dimension.
#         """










            # batch["observation.state"].squeeze(1)
            # batch["observation.images.front_cam1"].squeeze(1)
            # batch["observation.images.front_cam2"].squeeze(1)
            # batch["observation.images.front_cam3"].squeeze(1)
            # batch["observation.images.gripper_camera"].squeeze(1)

            # batch["observation.state"] = batch["observation.state"].squeeze(1)

            # cams = [
            #     batch.pop("observation.images.front_cam1").squeeze(1),
            #     batch.pop("observation.images.front_cam2").squeeze(1),
            #     batch.pop("observation.images.front_cam3").squeeze(1),
            #     batch.pop("observation.images.gripper_camera").squeeze(1),
            # ]

            # batch["observation.images"] = torch.stack(cams, dim=1)



            # for k in list(batch.keys()):
            #     if k.startswith("observation."):
            #         # Replace time sequence with last timestep
            #         batch[k] = batch[k][:, -1]    # (B, C, H, W)


            # if policy.config.image_features:
            #     batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            #     batch["observation.images"] = [batch[key] for key in policy.config.image_features]
            #     print(len(batch["observation.images"]))
            #     print(batch["observation.images"][0].shape)



            # # print all keys in dataset
            # for k, v in batch.items():
            #     if isinstance(v, torch.Tensor):
            #         print(k, v.shape)
            #     else:
            #         print(k, type(v))