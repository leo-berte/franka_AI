import torchvision.transforms as T
import torch


# TODO: 
# 1) Do I want images close to each others or farther? And for proprioception?

  
def build_delta_timestamps(fps, N_h, N_c):

    """
    Build the delta_timestamps dict for LeRobotDataset history and future.

    Args:
        fps: dataset frame rate (frames per second)
        N_h: number of history frames (including current)
        N_c: number of future action frames (including current)
        camera_key: key name for camera observation in dataset
    """

    dt = 1 / fps

    # history frames: [-2dt, -dt, ..., 0]
    history = [-(i * dt) for i in range(N_h - 1, 0, -1)] + [0]

    # future frames: [dt, 2dt, ...]
    chunk = [(i * dt) for i in range(N_c)]

    delta_timestamps = {
        "observation.image": history,
        "observation.state": history,
        "action": chunk,
    }
    return delta_timestamps
