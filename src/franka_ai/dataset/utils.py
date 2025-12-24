from pathlib import Path
from typing import Callable
import logging
import time
import yaml
import os

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from lerobot.common.datasets.utils import dataset_to_policy_features, load_jsonlines, cast_stats_to_numpy
from lerobot.configs.types import FeatureType


# TODO: 
# 1) Do I want images close to each others or far in time? 

# -------
# PATCHES
# -------

class LeRobotDatasetPatch(LeRobotDataset):
    
    def init(self, args, **kwargs):
        super().init(args, **kwargs)

    def _get_query_indices(self, idx, ep_idx):
        current_ep_idx = self.episodes.index(ep_idx) if self.episodes is not None else ep_idx
        return super()._get_query_indices(idx, current_ep_idx)

class MultiLeRobotDatasetPatch(MultiLeRobotDataset):
    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        super().__init__(repo_ids, root, episodes, image_transforms, delta_timestamps, tolerances_s, download_videos, video_backend)
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        self._datasets = [
            LeRobotDatasetPatch(
                repo_id,
                root=self.root / repo_id,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=self.tolerances_s[repo_id],
                download_videos=download_videos,
                video_backend=video_backend,
            )
            for repo_id in repo_ids
        ]

        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            logging.warning(
                f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                "other datasets."
            )
            self.disabled_features.update(extra_keys)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        # TODO(rcadene, aliberts): We should not perform this aggregation for datasets
        # with multiple robots of different ranges. Instead we should have one normalization
        # per robot.
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])




# ---------
# FUNCTIONS
# ---------

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

    Example:
        Obs history    (N_h=5)	[s(t=-4), s(t=-3), s(t=-2), s(t=-1), s(t=0)]
        Action history (N_h=5)	[a(t=-5), a(t=-4), a(t=-3), a(t=-2), a(t=-1)]
        Action chunk   (N_c=3)	[a(t=0), a(t=1), a(t=2)]
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

    # state history frames: [-(N_h-1)dt, ..., -2dt, -dt, 0]
    state_history = [-(i * step_hist * dt_dataset) for i in range(N_h-1, -1, -1)]

    # action history frames: [-(N_h)dt, ..., -2dt, -dt, 0]
    action_history = [-(i * step_chunk * dt_dataset) for i in range(N_h, 0, -1)]

    # future frames: [0, dt, 2dt, ..., (N_c-1)dt]
    chunk   = [(i * step_chunk * dt_dataset) for i in range(0, N_c)]

    # Combine history + chunk to potentially extrapolate also past actions
    full_actions = action_history + chunk

    # create dict
    delta_timestamps = {}

    # cameras
    for cam in feature_groups["VISUAL"]:
        delta_timestamps[cam] = state_history

    # state
    for s in feature_groups["STATE"]:
        delta_timestamps[s] = state_history

    # actions
    for a in feature_groups["ACTION"]:
        delta_timestamps[a] = full_actions

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
    
    print(f"Info about {ds_type}:")
    print(f"Number of total frames: {ds.meta.total_frames}")
    print(f"Number of selected frames: {ds.num_frames}")
    print(f"Number of total episodes: {ds.meta.total_episodes}")
    print(f"Number of selected episodes: {ds.num_episodes}")
    print(f"Number of fps: {ds.fps}")
    features = features = dataset_to_policy_features(ds.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    print("Output features: \n", output_features)
    print("Input features: \n", input_features)
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