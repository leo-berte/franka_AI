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

"""This script demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""



from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from torch.utils.tensorboard import SummaryWriter





def main(pretrained_path = None, learning_rate = 1e-4):

    """
    If pretrained_path is None → train from scratch.
    If pretrained_path is provided → fine-tune from checkpoint.
    """

    # Get folder where this script lives
    script_dir = Path(__file__).parent

    # Create a directory to store the training checkpoint.
    output_directory = script_dir / "outputs/train/example_pusht_diffusion"
    output_directory.mkdir(parents=True, exist_ok=True)

    # Select your device
    device = torch.device("cuda")

    # Number of training steps 
    training_steps = 10 # 5000
    log_freq = 1

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs 
    # NB: for normalization, lerobot simply compute mean, std_dev for each feature [q1, q2, ..] over all frame in the dataset
    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
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
        policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
    else:
        # Load from checkpoint
        policy = DiffusionPolicy.from_pretrained(pretrained_path)
        policy.dataset_stats = dataset_metadata.stats # it would be from a new dataset --> dataset_metadata_new.stats
        print(f"Loaded pretrained policy from {pretrained_path}")

    policy.train() # during training layers like Dropout or BatchNorm are ON 
    policy.to(device) # moves the model’s parameters to the correct device, since PyTorch operations require inputs and model to be on the same device

    # # Another policy-dataset interaction is with the delta_timestamps. Each policy expects a given number frames
    # # which can differ for inputs, outputs and rewards (if there are some).
    # delta_timestamps = {
    #     "observation.image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
    #     "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
    #     "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    # }

    # In this case with the standard configuration for Diffusion Policy, it is equivalent to this:
    delta_timestamps = {
        # Load the previous image and state at -0.1 seconds before current frame,
        # then load current image and state corresponding to 0.0 second.
        "observation.image": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        # Load the previous action (-0.1), the next action to be executed (0.0),
        # and 14 future actions with a 0.1 seconds spacing. All these actions will be
        # used to supervise the policy.
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }

    # We can then instantiate the dataset with these delta_timestamps configuration.
    dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)

    # Then we create our optimizer and dataloader for offline training.
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate) 
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True, # If the total number of samples is not divisible by batch_size, the last incomplete batch is dropped.
                        # Ensures every batch has exactly batch_size samples.
    )

    # -------------------
    # TRAINING LOOP
    # -------------------

    step = 0
    done = False

    tensorboard_dir = output_directory / "tensorboard"
    writer = SummaryWriter(log_dir=tensorboard_dir)
    # activate tensorboard (from where there is this code): python -m tensorboard.main --logdir ../outputs/train/example_pusht_diffusion/tensorboard

    while not done:

        for batch in dataloader:

            # This moves all tensor data in the batch to the GPU (device), if you’re using CUDA
            # batch_example = {
            #     "observation.state": torch.randn(32, 14),  # tensor
            #     "action": torch.randn(32, 14),             # tensor
            #     "episode_id": [0, 0, 0, ..., 0]            # list of ints
            # }
            # For each key k and value v in the batch:
            # If v is a tensor → move it to device
            # Otherwise → leave it as is
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            
            # computes the loss (and optionally predictions)
            loss, _ = policy.forward(batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break

            # Log to TensorBoard
            writer.add_scalar("Loss/train", loss.item(), step)

    # Save a policy checkpoint.
    policy.save_pretrained(output_directory)


if __name__ == "__main__":
    
    # Example usage:
    # main()  # from scratch
    # main(pretrained_path="outputs/train/example_pusht_diffusion", learning_rate = 1e-5)  # fine-tune (lower learning rate)

    main()
