from pathlib import Path
import gymnasium as gym
import gym_pusht
import imageio
import numpy
import torch
import os

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

from franka_ai.inference.utils import setup_eval_folder

# Reproducibility
SEED=42

"""
Run the code: python src/franka_ai/inference/evaluate.py
"""


# TODO:
# 0) define metrics to measure how much inference code is efficient (measure time, memory, ..)





def evaluate():

    """

    INSERT CLEAN DESCRIPTION

    """

    # Get folder to save eval video
    eval_dir = setup_eval_folder()

    # Parameters
    device = "cuda"  

    # Base folder for all experiments
    base_output_dir = os.path.join(os.getcwd())
    pretrained_policy_path = os.path.join(base_output_dir, "outputs", "checkpoints", "today_outliers_DP_2025-12-02_16-40-09", "best_model.pt") # SI, allenato da conda
    # pretrained_policy_path = os.path.join(base_output_dir, "outputs", "checkpoints", "today_outliers_DP_2025-12-02_17-10-22", "best_model.pt") # NO, allenato da docker
    print(pretrained_policy_path)

    # --> not working even with: leonardo@leonardo-Precision-5490:~/Documents/Coding/franka_AI/outputs$ sudo chown -R $USER:$USER checkpoints/


    # Load policy
    policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)

    # Initialize evaluation environment to render two observation types:
    # an image of the scene and state/position of the agent. The environment
    # also automatically stops running after 300 interactions/steps.
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=10, # 300
    )

    # We can verify that the shapes of the features expected by the policy match the ones from the observations
    # produced by the environment
    print("Check input shapes")
    print(policy.config.input_features)
    print(env.observation_space)

    # Same check on actions
    print("Check action shapes")
    print(policy.config.output_features)
    print(env.action_space)

    # Reset the policy and environments to prepare for rollout
    policy.reset()
    numpy_observation, info = env.reset(seed=SEED)

    # Prepare to collect every rewards and all the frames of the episode
    rewards = []
    frames = []

    # Render frame of the initial state
    frames.append(env.render())

    step = 0
    done = False

    while not done:

        # Prepare observation for the policy running in Pytorch
        state = torch.from_numpy(numpy_observation["agent_pos"])
        image = torch.from_numpy(numpy_observation["pixels"])

        # Convert to float32 with image from channel first in [0,255]
        # to channel last in [0,1]
        state = state.to(torch.float32)
        image = image.to(torch.float32) / 255
        image = image.permute(2, 0, 1)

        # Send data tensors from CPU to GPU
        state = state.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)

        # Add extra (empty) batch dimension, required to forward the policy
        state = state.unsqueeze(0)
        image = image.unsqueeze(0)

        # Create the policy input dictionary
        observation = {
            "observation.state": state,
            "observation.image": image,
        }

        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            action = policy.select_action(observation)

        # Prepare the action for the environment
        numpy_action = action.squeeze(0).to("cpu").numpy()

        # Step through the environment and receive a new observation
        numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
        print(f"{step=} {reward=} {terminated=}")

        # Keep track of all the rewards and frames
        rewards.append(reward)
        frames.append(env.render())

        # The rollout is considered done when the success state is reached (i.e. terminated is True),
        # or the maximum number of iterations is reached (i.e. truncated is True)
        done = terminated | truncated | done
        step += 1

    if terminated:
        print("Success!")
    else:
        print("Failure!")

    # Get the speed of environment (i.e. its number of frames per second).
    fps = env.metadata["render_fps"]

    # Encode all frames into a mp4 video.
    video_path = os.path.join(eval_dir, "rollout.mp4")
    imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

    print(f"Video of the evaluation is available in '{video_path}'.")




if __name__ == "__main__":
    
    evaluate()