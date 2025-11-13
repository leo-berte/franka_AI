import os
from datetime import datetime
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter

from lerobot.configs.types import FeatureType, PolicyFeature

# TODO:
# 2) magari in input setup_folders accetta policy name o percorso di salvataggio dall utente?


def setup_folders():

    """Create timestamped output folder structure for training run."""
    
    # Base folder for all experiments
    base_output_dir = os.path.join(os.getcwd(), "outputs")

    # Unique run name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # run_name = f"{timestamp}_{cfg.__class__.__name__.lower()}_{dataset_id.split('/')[-1]}"
    # output_dir = os.path.join(base_output_dir, run_name)

    # Subfolders
    checkpoints_dir = os.path.join(base_output_dir, "checkpoints", timestamp)
    tensorboard_dir = os.path.join(base_output_dir, "tensorboard")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=tensorboard_dir)

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

def dataset_to_policy_features_patch(sample, features):

    policy_features = {}

    for ft_type, ft_list in features.items():

        for ft in ft_list:
            
            # get data
            v = sample[ft]

            # get full shape
            shape = tuple(v.shape) 
            
            if FeatureType[ft_type] == FeatureType.VISUAL:
                shape = shape[-3:]
            elif FeatureType[ft_type] in [FeatureType.STATE, FeatureType.ACTION]:
                shape = shape[-1:]
            
            # compose policy feature
            policy_features[ft] = PolicyFeature(type=FeatureType[ft_type], shape=shape)

    return policy_features

def compute_dataset_stats_patch(dataloader, feature_groups):

    stats = {}

    for ft_type, ft_list in feature_groups.items():

        for ft in ft_list:

            values = []

            for batch in dataloader:

                print("dio")

                x = batch[ft].detach().cpu()
                x = x.float().reshape(-1, x.shape[-1]) if x.ndim > 2 else x
                values.append(x)

            values = torch.cat(values, dim=0)

            print(ft)

            stats[ft] = {
                "mean": values.mean(dim=0),
                "std": values.std(dim=0),
                "min": values.min(dim=0).values,
                "max": values.max(dim=0).values,
            }

    return stats