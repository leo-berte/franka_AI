from datetime import datetime
from shutil import copyfile
import os
import yaml
import json
from torch.utils.tensorboard import SummaryWriter

from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode


# ---------
# FUNCTIONS
# ---------

def get_configs_training(config_rel_path):

    # set path
    path = os.path.join(os.getcwd(), config_rel_path)

    # open config
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    policies_cfg = cfg["policies"]

    return train_cfg, policies_cfg
    
def set_output_folders_train(policy_type, dataset_path, config_folder, use_tensorboard=True):

    """Create timestamped output folder structure for training run."""
    
    # Base folder for all experiments
    base_output_dir = os.path.join(os.getcwd(), "outputs")

    # Take timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Set run name
    dataset_name = dataset_path.split('/')[-1]
    run_name = f"{dataset_name}_{policy_type}_{config_folder}_{timestamp}"

    # Subfolders and writers
    checkpoints_dir = os.path.join(base_output_dir, "checkpoints", run_name)
    os.makedirs(checkpoints_dir, exist_ok=True)

    writer = None
    if use_tensorboard:
        tensorboard_dir = os.path.join(base_output_dir, "tensorboard", run_name)
        os.makedirs(tensorboard_dir, exist_ok=True)

        # TensorBoard writer
        writer = SummaryWriter(log_dir=tensorboard_dir)

    # Inject policy name and dataset fps into inference.yaml and save it in 'checkpoints_dir'

    with open(f"configs/{config_folder}/inference.yaml", "r") as f:
        inference_cfg = yaml.safe_load(f)

    inference_cfg.setdefault("inference", {})
    inference_cfg["inference"]["policy_name"] = policy_type
    inference_cfg["inference"]["fps_dataset"] = load_dataset_fps(dataset_path)

    with open(os.path.join(checkpoints_dir, "inference.yaml"), "w") as f:
        yaml.safe_dump(inference_cfg, f, sort_keys=False)

    # Save configs inside checkpoint folder for reproducibility
    copyfile(f"configs/{config_folder}/dataset.yaml", os.path.join(checkpoints_dir, "dataset.yaml"))
    copyfile(f"configs/{config_folder}/train.yaml", os.path.join(checkpoints_dir, "train.yaml"))
    copyfile(f"configs/{config_folder}/models.yaml", os.path.join(checkpoints_dir, "models.yaml"))

    return checkpoints_dir, writer

def load_dataset_fps(dataset_path):

    info_path = os.path.join(dataset_path, "meta", "info.json")

    with open(info_path, "r") as f:
        info = json.load(f)

    fps = info["fps"]
    return float(fps)

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

def dataset_to_policy_features_patch(dataloader, features, transforms):

    batch = next(iter(dataloader))

    # apply custom transforms to have the batch in the final shape
    batch = transforms.transform(batch) 

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