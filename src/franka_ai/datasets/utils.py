import torchvision.transforms as T
import torch


# TODO: 
# 1) Do I want images close to each others or far in time? And for proprioception?

  
def build_delta_timestamps(fps_dataset, fps_sampling, N_h, N_c, visual_obs_names):

    """
    Build the delta_timestamps dict for LeRobotDataset history and future.

    Args:
        fps_dataset: dataset original frame rate (frames per second)
        fps_sampling: frame rate used to import the dataset (frames per second)
        N_h: number of history frames (including current)
        N_c: number of future action frames (including current)
        visual_obs_names: names for camera observations in dataset
    """

    # adjust fps_dataset to fps_sampling
    dt_dataset = 1 / fps_dataset
    step = round(fps_dataset / fps_sampling) if fps_sampling else 1

    # history frames: [-2dt, -dt, ..., 0]
    history = [-(i * step * dt_dataset) for i in range(N_h - 1, 0, -1)] + [0]

    # future frames: [dt, 2dt, ...]
    chunk   = [(i * step * dt_dataset) for i in range(N_c)]

    # create dict
    delta_timestamps = {}

    # camera obs
    for cam in visual_obs_names:
        delta_timestamps[cam] = history

    # proprio obs
    delta_timestamps["observation.state"] = history
    
    # actions
    delta_timestamps["action"] = chunk

    return delta_timestamps
