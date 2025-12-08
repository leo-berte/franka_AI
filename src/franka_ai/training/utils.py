from datetime import datetime
from shutil import copyfile
import os
import yaml
from torch.utils.tensorboard import SummaryWriter

from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode


def get_configs_training(config_rel_path):

    # set path
    path = os.path.join(os.getcwd(), config_rel_path)

    # open config
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    normalization_cfg = cfg["normalization_mapping"]

    return train_cfg, normalization_cfg
    
def set_output_folders_train(policy_type, dataset_path, config_folder):

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
    copyfile(f"configs/{config_folder}/dataset.yaml", os.path.join(checkpoints_dir, "dataset.yaml"))
    copyfile(f"configs/{config_folder}/train.yaml", os.path.join(checkpoints_dir, "train.yaml"))
    copyfile(f"configs/{config_folder}/models.yaml", os.path.join(checkpoints_dir, "models.yaml"))

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
        
        # skip unwanted features
        if ft_type == "REMOVE" or ft_type == "META":
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