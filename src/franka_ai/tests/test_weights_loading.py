#!/usr/bin/env python3

from lerobot.configs.types import FeatureType

from franka_ai.dataset.utils import get_configs_dataset, LeRobotDatasetPatch
from franka_ai.training.utils import *
from franka_ai.models.factory import get_policy_config_class, make_policy, get_policy_class
from franka_ai.inference.utils import get_configs_inference


"""

Run the code: 

python src/franka_ai/tests/test_weights_loading.py 

"""


# Load configs
dataloader_cfg, dataset_cfg, transforms_cfg = get_configs_dataset("configs/dataset.yaml")
inference_cfg = get_configs_inference("configs/inference.yaml")

# Get parameter values from inference.yaml
dataset_path = "data/today_data/today_outliers"
policy_name = inference_cfg["policy_name"]

# Get folders to save weights and tensorboard logs
checkpoints_dir, tensorboard_dir, tsbrd_writer = set_output_folders_train(policy_name, dataset_path)

# Load the raw dataset (hub or local)
train_ds = LeRobotDatasetPatch(
    repo_id=None,
    root=dataset_path
)

# Get dataset input/output stats
features = dataset_to_policy_features_patch(train_ds, dataset_cfg["features"])
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if ft.type is not FeatureType.ACTION}
# print("New output_features: ", output_features)
# print("New input_features: ", input_features)

# Get policy configuration class
ConfigClass = get_policy_config_class(policy_name)

cfg = ConfigClass(
    input_features=input_features,
    output_features=output_features,
)

# Setup policy
policy = make_policy(policy_name, cfg)

# Save checkpoint
ckpt_path = os.path.join(checkpoints_dir, f"step.pt")
print("ckpt_path: ", ckpt_path)
policy.save_pretrained(ckpt_path)

# Load policy
PolicyClass = get_policy_class(policy_name)
policy = PolicyClass.from_pretrained(ckpt_path)