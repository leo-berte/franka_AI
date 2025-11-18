from datetime import datetime
from shutil import copyfile
import os
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter

from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode


def get_train_config(config_rel_path):

    # set path
    path = os.path.join(os.getcwd(), config_rel_path)

    # open config
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    normalization_cfg = cfg["normalization_mapping"]

    return train_cfg, normalization_cfg
    
def setup_folders(policy_type, dataset_path):

    """Create timestamped output folder structure for training run."""
    
    # Base folder for all experiments
    base_output_dir = os.path.join(os.getcwd(), "outputs")

    # Take timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Set run name
    dataset_name = dataset_path.split('/')[-1]
    run_name = f"{dataset_name}_{policy_type}_{timestamp}"

    # Subfolders
    checkpoints_dir = os.path.join(base_output_dir, "checkpoints", run_name)
    tensorboard_dir = os.path.join(base_output_dir, "tensorboard", run_name)

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Save configs inside checkpoint folder for reproducibility
    copyfile("configs/dataset.yaml", os.path.join(checkpoints_dir, "dataset.yaml"))
    copyfile("configs/train.yaml", os.path.join(checkpoints_dir, "train.yaml"))

    return checkpoints_dir, tensorboard_dir, writer

def update_best_model_symlink(checkpoints_dir, ckpt_path, best_val_loss, avg_val_loss, step):

    """Create or update the symlink to the best model checkpoint."""

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_checkpoint_path = os.path.join(checkpoints_dir, "best_model.pt")

        # Remove old symlink if it exists
        if os.path.islink(best_checkpoint_path):
            os.remove(best_checkpoint_path)

        # Create new symlink to the best checkpoint
        os.symlink(os.path.basename(ckpt_path), best_checkpoint_path)
        print(f"New best model at step {step}, val_loss={best_val_loss:.4f}")

    return best_val_loss

def parse_normalization_mapping(normalization_cfg):

    str_to_enum = {
        "MEAN_STD": NormalizationMode.MEAN_STD,
        "MIN_MAX": NormalizationMode.MIN_MAX,
        "IDENTITY": NormalizationMode.IDENTITY,
    }

    return {k: str_to_enum[v] for k, v in normalization_cfg.items()}

def dataset_to_policy_features_patch(dataloader, features):

    batch = next(iter(dataloader))

    policy_features = {}

    for ft_type, ft_list in features.items():
        
        # skip REMOVE features
        if ft_type == "REMOVE":
            continue

        for ft in ft_list:
            
            # get data
            v = batch[ft]

            # get full shape
            shape = tuple(v.shape) 
            
            if ft_type == "VISUAL":
                shape = shape[-3:]
            elif ft_type in ["STATE", "ACTION"]:
                shape = shape[-1:]
            
            # compose policy feature
            policy_features[ft] = PolicyFeature(type=FeatureType[ft_type], shape=shape)

    return policy_features

def compute_dataset_stats_patch(dataloader, feature_groups):

    print("\nComputing dataset statistics...")

    mins = {}
    maxs = {}
    sums = {}
    sq_sums = {}
    counts = {}
    stats = {}

    # extract first batch to infer feature shapes
    first_batch = next(iter(dataloader))

    # select features to be used to compute statistics
    keep_keys = []
    for ft_type, ft_list in feature_groups.items():
        for ft in ft_list:
            if (ft_type != "REMOVE"):
                keep_keys.append(ft)

    # initialize accumulators
    for k in keep_keys:
        
        # get features for a given key and collapse batch + history
        x = first_batch[k].detach().cpu().float()
        x_flat = x.reshape(-1, *x.shape[2:])  # (B*N_h, ...)

        if k in feature_groups["VISUAL"]:
            # x_flat: (B*N_h, C, H, W)
            BNH, C, H, W = x_flat.shape
            prototype = torch.zeros(C)  # stats per channel
        else:
            # x_flat: (B*N_h, D)
            prototype = torch.zeros(*x_flat.shape[1:])

        mins[k] = torch.full_like(prototype, float('inf'))
        maxs[k] = torch.full_like(prototype, float('-inf'))
        sums[k] = torch.zeros_like(prototype)
        sq_sums[k] = torch.zeros_like(prototype)
        counts[k] = 0

    # Scan entire dataset
    for batch in dataloader:

        for k in keep_keys:

            # get features for a given key and collapse batch + history
            x = batch[k].detach().cpu().float()   # (B, N_h, ...)
            x_flat = x.reshape(-1, *x.shape[2:])  # (B*N_h, ...)

            if k in feature_groups["VISUAL"]:
                # (B*N_h, C, H, W) → (C, total_pixels)
                BNH, C, H, W = x_flat.shape

                # reshape to (BNH, C, H*W)
                x_temp = x_flat.reshape(BNH, C, -1)

                # permute to (C, BNH, H*W)
                x_temp = x_temp.permute(1, 0, 2)

                # collapse into (C, BNH*H*W)
                x_vis = x_temp.reshape(C, -1)

                # update stats over dim=1 (pixels)
                batch_min = x_vis.min(dim=1).values
                batch_max = x_vis.max(dim=1).values
                batch_sum = x_vis.sum(dim=1)
                batch_sq_sum = (x_vis ** 2).sum(dim=1)
                batch_count = x_vis.shape[1]

            else:
                # Non-visual → (BNH, D)
                batch_min = x_flat.min(dim=0).values
                batch_max = x_flat.max(dim=0).values
                batch_sum = x_flat.sum(dim=0)
                batch_sq_sum = (x_flat ** 2).sum(dim=0)
                batch_count = x_flat.shape[0]

            # Accumulate global stats
            mins[k] = torch.minimum(mins[k], batch_min)
            maxs[k] = torch.maximum(maxs[k], batch_max)
            sums[k] += batch_sum
            sq_sums[k] += batch_sq_sum
            counts[k] += batch_count

    # Finalize statistics
    for k in keep_keys:

        total = counts[k]
        mean = sums[k] / total
        var = (sq_sums[k] / total) - mean ** 2
        std = torch.sqrt(var + 1e-8)

        if k in feature_groups["VISUAL"]:
            # Shape must be (C,1,1) (LeRobot format)
            mean = mean.reshape(-1, 1, 1)
            std = std.reshape(-1, 1, 1)
            mins[k] = mins[k].reshape(-1, 1, 1)
            maxs[k] = maxs[k].reshape(-1, 1, 1)

        stats[k] = {
            "min": mins[k],
            "max": maxs[k],
            "mean": mean,
            "std": std,
        }

    print("Finished computing dataset statistics!\n")

    print(stats)
    
    return stats