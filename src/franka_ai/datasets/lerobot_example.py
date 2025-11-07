# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script demonstrates the use of `LeRobotDataset` class for handling and processing robotic datasets from Hugging Face.
It illustrates how to load datasets, manipulate them, and apply transformations suitable for machine learning tasks in PyTorch.

Features included in this script:
- Viewing a dataset's metadata and exploring its properties.
- Loading an existing dataset from the hub or a subset of it.
- Accessing frames by episode number.
- Using advanced dataset features like timestamp-based frame selection.
- Demonstrating compatibility with PyTorch DataLoader for batch processing.

The script ends with examples of how to batch process data using PyTorch's DataLoader.
"""

from pprint import pprint

import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


"""
Run the code:

python src/franka_ai/datasets/lerobot_example.py

Visualize lerobot dataset uploaded in HF

lerobot-dataset-viz     --repo-id lerobot/pusht     --episode-index 0

Visualize custom dataset stored like:

~/Documents/LeRobot_datasets/
└── my_user/
    └── softbag_pick_place/
        ├── metadata/
        ├── videos/
        ├── data/
        └── version.txt

Visualize it:

lerobot-dataset-viz \
  --repo-id my_user/softbag_pick_place \
  --root ~/Documents/LeRobot_datasets \
  --mode local \
  --episode-index 0
"""


# # We ported a number of existing datasets ourselves, use this to see the list:
# print("List of available datasets:")
# pprint(lerobot.available_datasets)

# # You can also browse through the datasets created/ported by the community on the hub using the hub api:
# hub_api = HfApi()
# repo_ids = [info.id for info in hub_api.list_datasets(task_categories="robotics", tags=["LeRobot"])]
# pprint(repo_ids)

# Or simply explore them in your web browser directly at:
# https://huggingface.co/datasets?other=LeRobot

# Let's take this one for this example
# repo_id = "lerobot/aloha_mobile_cabinet" --> may need: LeRobotDataset(.., video_backend="pyav")
# repo_id = "lerobot/aloha_mobile_cabinet" 
repo_id = "lerobot/droid_100"
# We can have a look and fetch its metadata to know more about it:
ds_meta = LeRobotDatasetMetadata(repo_id)

# By instantiating just this class, you can quickly access useful information about the content and the
# structure of the dataset without downloading the actual data yet (only metadata files — which are
# lightweight).
print(f"Total number of episodes: {ds_meta.total_episodes}")
print(f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}")
print(f"Frames per second used during data collection: {ds_meta.fps}")
print(f"Robot type: {ds_meta.robot_type}")
print(f"keys to access images from cameras: {ds_meta.camera_keys=}\n")

print("Tasks:")
print(ds_meta.tasks)
print("Features:")
pprint(ds_meta.features)

# # You can also get a short summary by simply printing the object:
# print(ds_meta)

# You can then load the actual dataset from the hub.
# Either load any subset of episodes:
dataset = LeRobotDataset(repo_id, episodes=[0]) # episodes=[0, 10, 11, 23]

# And see how many frames you have:
print(f"Selected episodes: {dataset.episodes}")
print(f"Number of episodes selected: {dataset.num_episodes}")
print(f"Number of frames selected: {dataset.num_frames}")

# select data from current frame (it is a dictionary!)
print(f"\n{dataset[0][dataset.meta.camera_keys[0]].shape=}")  # (c, h, w)
print(f"{dataset[0]['observation.state'].shape=}")  # (1, c)
print(f"{dataset[0]['action'].shape=}\n")  # (1, c)

# Or simply load the entire dataset:
dataset = LeRobotDataset(repo_id)
print(f"Number of episodes selected: {dataset.num_episodes}")
print(f"Number of frames selected: {dataset.num_frames}")

# The previous metadata class is contained in the 'meta' attribute of the dataset:
print(dataset.meta)

# LeRobotDataset actually wraps an underlying Hugging Face dataset
# (see https://huggingface.co/docs/datasets for more information).
# print(dataset.hf_dataset)

# LeRobot datasets also subclasses PyTorch datasets so you can do everything you know and love from working
# with the latter, like iterating through the dataset.
# The __getitem__ iterates over the frames of the dataset. Since our datasets are also structured by
# episodes, you can access the frame indices of any episode using dataset.meta.episodes. Here, we access
# frame indices associated to the first episode:
episode_index = 0
from_idx = dataset.episode_data_index["from"][episode_index].item()
to_idx = dataset.episode_data_index["to"][episode_index].item()
print(f"from_idx: {from_idx}") # 0
print(f"to_idx: {to_idx}") # 1500

# Then we grab all the image frames from the first camera:
camera_key = dataset.meta.camera_keys[0]
frames = [dataset[idx][camera_key] for idx in range(from_idx, to_idx)]

# The objects returned by the dataset are all torch.Tensors
print(type(frames[0]))
print(frames[0].shape)

# Finally, our datasets are fully compatible with PyTorch dataloaders and samplers because they are just
# PyTorch datasets.
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=0,
    batch_size=32,
    shuffle=True,
)

for batch in dataloader:
    print(f"{batch[camera_key].shape=}")  # (32, c, h, w)
    print(f"{batch['observation.state'].shape=}")  # (32, 14)
    print(f"{batch['action'].shape=}")  # (32, 14)
    break


# For many machine learning applications we need to load the history of past observations or trajectories of
# future actions. Our datasets can load previous and future frames for each key/modality, using timestamps
# differences with the current loaded frame. For instance:

fps = dataset.fps  # e.g. 30 Hz
dt = 1/fps

# history: N_h frames back (including current)
N_h = 5
history = [-(i * dt) for i in range(N_h-1, 0, -1)] + [0]

# future: 64 frames ahead
N_a = 5
future = [(i * dt) for i in range(N_a)]

delta_timestamps = {
    camera_key: history,
    "observation.state": history,
    "action": future,
}

dataset_with_history = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
print(f"\n{dataset_with_history[0][camera_key].shape=}")  # (N_h, c, h, w)
print(f"{dataset_with_history[0]['observation.state'].shape=}")  # (N_h, c)
print(f"{dataset_with_history[0]['action'].shape=}\n")  # (N_a, c)

# Finally, our datasets are fully compatible with PyTorch dataloaders and samplers because they are just
# PyTorch datasets.
dataloader_with_history = torch.utils.data.DataLoader(
    dataset_with_history,
    num_workers=0,
    batch_size=32,
    shuffle=True,
)

for batch in dataloader_with_history:
    print(f"{batch[camera_key].shape=}")  # (32, N_h, c, h, w)
    print(f"{batch['observation.state'].shape=}")  # (32, N_h, 14)
    print(f"{batch['action'].shape=}")  # (32, N_a, 14)
    break