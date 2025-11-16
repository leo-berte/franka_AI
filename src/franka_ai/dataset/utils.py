import torchvision.transforms as T
import argparse
import torch
import time
import yaml
import os


# TODO: 
# 1) Do I want images close to each others or far in time? And for proprioception?
# 2) maybe f_sampling_history and f_sampling_chunk
  
def build_delta_timestamps(fps_dataset, fps_sampling, N_h, N_c, feature_groups):

    """
    Build the delta_timestamps dict for LeRobotDataset history and future.

    Args:
        fps_dataset: dataset original frame rate (frames per second).
        fps_sampling: frame rate used to import the dataset (frames per second).
        N_h: number of history frames (including current).
        N_c: number of future action frames (including current).
        feature_groups: feature names used in the dataset.
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

    # cameras
    for cam in feature_groups["VISUAL"]:
        delta_timestamps[cam] = history

    # state
    for s in feature_groups["STATE"]:
        delta_timestamps[s] = history

    # actions
    for a in feature_groups["ACTION"]:
        delta_timestamps[a] = chunk

    return delta_timestamps

def get_dataset_path(default_rel_path="data/test1"):

    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--abs_path",
        type=str,
        required=False,
        help="Absolute path to the dataset folder"
    )
    args = parser.parse_args()

    # return absolute path to the dataset
    return os.path.join(os.getcwd(), default_rel_path) if not args.abs_path else args.abs_path

def get_configs_dataset(config_rel_path):

    # set path
    config_path = os.path.join(os.getcwd(), config_rel_path)  

    # load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # extract sections
    dataset_cfg = cfg["dataset"]
    dataloader_cfg = cfg["dataloader"]
    transformations_cfg = cfg["transformations"]

    return dataloader_cfg, dataset_cfg, transformations_cfg

def benchmark_loader(loader, num_batches=100):

    it = iter(loader)
    _ = next(it)  # warm-up workers

    num_batches = min(num_batches, len(loader))

    times = []
    for _ in range(num_batches):
        t0 = time.perf_counter()
        batch = next(it)
        times.append(time.perf_counter() - t0)

    print(f"Time stats per-batch processing (n_batches={num_batches}):")
    print("Avg:", sum(times)/len(times))
    print("Min:", min(times))
    print("Max:", max(times))

def benchmark_loader2(loader):

    # Warm-up workers
    _ = next(iter(loader))

    # Start timing
    start = time.perf_counter()
    for batch in loader:
        pass
    elapsed = time.perf_counter() - start

    num_batches = len(loader)
    print(f"Time stats per-batch processing (n_batches={num_batches}):")
    print(f"Avg time per batch: {elapsed / num_batches:.4f}s")