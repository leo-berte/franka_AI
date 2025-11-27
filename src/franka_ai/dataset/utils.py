import torchvision.transforms as T
import torch
import time
import yaml
import os

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import dataset_to_policy_features, load_jsonlines, cast_stats_to_numpy
from lerobot.configs.types import FeatureType


# TODO: 
# 1) Do I want images close to each others or far in time? 
# 2) capire se devo usare N_chunk = 2 --> [0.0, 0.1] o [0.1, 0.2] ???


class LeRobotDatasetPatch(LeRobotDataset):
    
    def init(self, args, **kwargs):
        super().init(args, **kwargs)

    def _get_query_indices(self, idx, ep_idx):
        current_ep_idx = self.episodes.index(ep_idx) if self.episodes is not None else ep_idx
        return super()._get_query_indices(idx, current_ep_idx)


def build_delta_timestamps(feature_groups, N_h, N_c, fps_dataset, fps_sampling_hist=10, fps_sampling_chunk=10):

    """
    Build the delta_timestamps dict for LeRobotDataset history and future.

    Args:
        feature_groups: feature names used in the dataset.
        N_h: number of history frames (including current).
        N_c: number of future action frames (including current).
        fps_dataset: dataset original frame rate (frames per second).
        fps_sampling_hist: frame rate used to import history data (frames per second).
        fps_sampling_chunk: frame rate used to import chunk data (frames per second).
    """

    # consistency checks
    if N_h < 1:
        raise ValueError("N_history must be ≥ 1")
    if N_c < 1:
        raise ValueError("N_chunk must be ≥ 1")

    # adjust fps_dataset to fps_sampling
    dt_dataset = 1 / fps_dataset
    step_hist = round(fps_dataset / fps_sampling_hist)
    step_chunk = round(fps_dataset / fps_sampling_chunk)

    # history frames: [..., -2dt, -dt, 0]
    history = [-(i * step_hist * dt_dataset) for i in range(N_h-1, -1, -1)]

    # future frames: [dt, 2dt, ...]
    chunk   = [(i * step_chunk * dt_dataset) for i in range(1, N_c+1)]

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
        delta_timestamps[a] = history + chunk

    # print(delta_timestamps["action"]) --> [-0.0, 0.033337900999492256]
    # print(delta_timestamps["observation.state"]) -> [-0.0]

    return delta_timestamps

def get_configs_dataset(config_rel_path):

    # set path
    config_path = os.path.join(os.getcwd(), config_rel_path)  

    # load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # extract sections
    dataset_cfg = cfg["dataset"]
    dataloader_cfg = cfg["dataloader"]
    transforms_cfg = cfg["transforms"]

    return dataloader_cfg, dataset_cfg, transforms_cfg

def load_episodes_stats_patch(episode_stats_path):

    if not episode_stats_path.exists():
                raise FileNotFoundError(
                    f"\nERROR: Missing statistics file: {episode_stats_path}\n"
                    "You need to compute new episodes stats by executing: generate_updated_stats.py\n"
                )

    episodes_stats_raw = load_jsonlines(episode_stats_path)
    episodes_stats = {
    item["episode_index"]: cast_stats_to_numpy(item["stats"])
    for item in sorted(episodes_stats_raw, key=lambda x: x["episode_index"])
    }

    return episodes_stats

def print_dataset_info(ds, ds_type):
    
    print("\n")
    print(f"Info about {ds_type}:")
    print(f"Number of total frames: {ds.meta.total_frames}")
    print(f"Number of selected frames: {ds.num_frames}")
    print(f"Number of total episodes: {ds.meta.total_episodes}")
    print(f"Number of selected episodes: {ds.num_episodes}")
    print(f"Number of fps: {ds.fps}")
    print(f"Camera keys: {ds.meta.camera_keys}")
    features = features = dataset_to_policy_features(ds.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    print("Output features: \n", output_features)
    print("Input features: \n", input_features)
    # print("Full features:")
    # print(ds.features)
    print("\n")

def benchmark_loader(loader, num_batches=100):

    print(f"\nStarting Per-Batch Latency Benchmark analysis...")

    # warm-up workers
    it = iter(loader)
    _ = next(it)  

    num_batches = min(num_batches, len(loader)-1)

    times = []
    for _ in range(num_batches):
        t0 = time.perf_counter()
        _ = next(it)
        times.append(time.perf_counter() - t0)

    print(f"\n[Per-Batch Latency Benchmark] ({num_batches} batches)")
    print(f"Avg batch time : {sum(times)/len(times):.5f} s")
    print(f"Min batch time : {min(times):.5f} s")
    print(f"Max batch time : {max(times):.5f} s")

def benchmark_loader2(loader, num_batches=100):

    print(f"\nStarting Throughput Benchmark analysis...")

    # warm-up workers
    it = iter(loader)
    _ = next(it)  

    num_batches = min(num_batches, len(loader)-1)

    # Start timing
    start = time.perf_counter()
    for _ in range(num_batches):
        _ = next(it)
    elapsed = time.perf_counter() - start

    print(f"\n[Throughput Benchmark] ({num_batches} batches)")
    print(f"Avg batch time: {elapsed / num_batches:.5f} s")
    print(f"Total time: {elapsed:.5f} s")