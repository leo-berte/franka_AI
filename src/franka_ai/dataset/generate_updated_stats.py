from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import torch
import time


from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import append_jsonlines, serialize_dict

from franka_ai.dataset.load_dataset import TransformedDataset
from franka_ai.dataset.transforms import CustomTransforms
from franka_ai.dataset.utils import get_configs_dataset, build_delta_timestamps, LeRobotDatasetPatch

"""
Run the code: python src/franka_ai/dataset/generate_updated_stats.py --dataset /home/leonardo/Documents/Coding/franka_AI/data/today_data/today
"""

# TODO:
# 1) loro fanno anche sampling delle immagini per calcolare stats piu velocmente, io come posso farlo?
# 3) check global stats e episdoe stats in comparison avec les original


def get_dataset_path():

    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Absolute path to the dataset folder"
    )
    args = parser.parse_args()

    # return absolute path to the dataset
    return args.dataset

def compute_episode_stats_streaming(dataloader, keep_keys, feature_groups, batch_size, N_history):

    mins = {}
    maxs = {}
    sums = {}
    sq_sums = {}
    counts_elems = {}
    counts_frames = {}
    stats = {}

    # extract first batch to infer feature shapes
    first_batch = next(iter(dataloader))

    # initialize accumulators
    for k in keep_keys:
        
        # get features for a given key and collapse batch + history
        x = first_batch[k].detach().cpu().float() # (B,N_h,...)

        if k in feature_groups["VISUAL"]:
            x_flat = x.reshape(-1, *x.shape[2:])  # (B*N_h, C, H, W)
            BNH, C, H, W = x_flat.shape
            prototype = torch.zeros(C)  # (C,) stats per channel
        elif k in feature_groups["META"]:
            x_flat = x.reshape(-1)  # (1,)
            prototype = torch.zeros(x_flat.shape[0]) # (1,)
        else: # "STATE", "ACTION"
            x_flat = x.reshape(-1, *x.shape[2:])  # (B*N_h, D)
            prototype = torch.zeros(*x_flat.shape[1:]) # (D,)

        mins[k] = torch.full_like(prototype, float('inf'))
        maxs[k] = torch.full_like(prototype, float('-inf'))
        sums[k] = torch.zeros_like(prototype)
        sq_sums[k] = torch.zeros_like(prototype)
        counts_elems[k] = 0
        counts_frames[k] = 0

    # Scan entire dataset
    for batch in dataloader:

        for k in keep_keys:

            # get features for a given key and collapse batch + history
            x = batch[k].detach().cpu().float()   # (B, N_h, ...)

            # reshape values
            if k in feature_groups["VISUAL"]:
                x_flat = x.reshape(-1, *x.shape[2:])  # reshape to (B*N_h, C, H, W)
                BNH, C, H, W = x_flat.shape
                x_flat = x_flat.reshape(BNH, C, -1) # reshape to (BNH, C, H*W)
                x_flat = x_flat.permute(1, 0, 2) # permute to (C, BNH, H*W)
                x_flat = x_flat.reshape(C, -1) # collapse into (C, BNH*H*W)
                batch_count_elems = x_flat.shape[1]
                batch_count_frames = batch_size * N_history
                dim = 1
            elif k in feature_groups["META"]:
                x_flat = x.reshape(-1)  # (1,)
                batch_count_elems = x_flat.shape[0]
                batch_count_frames = batch_size
                dim = 0
            else: # "STATE", "ACTION"
                x_flat = x.reshape(-1, *x.shape[2:])  # reshape to (B*N_h, D)
                batch_count_elems = x_flat.shape[0]
                batch_count_frames = batch_size * N_history
                dim = 0

            # batch stats
            batch_min = x_flat.min(dim=dim).values
            batch_max = x_flat.max(dim=dim).values
            batch_sum = x_flat.sum(dim=dim)
            batch_sq_sum = (x_flat ** 2).sum(dim=dim)

            # Accumulate global stats
            mins[k] = torch.minimum(mins[k], batch_min)
            maxs[k] = torch.maximum(maxs[k], batch_max)
            sums[k] += batch_sum
            sq_sums[k] += batch_sq_sum
            counts_elems[k] += batch_count_elems
            counts_frames[k] += batch_count_frames

    # Finalize statistics
    for k in keep_keys:

        total = counts_elems[k]
        mean = sums[k] / total
        var = (sq_sums[k] / total) - mean ** 2
        std = torch.sqrt(var + 1e-8)

        if k in feature_groups["VISUAL"]:
            # Shape must be (C,1,1) (LeRobot format)
            mean = mean.reshape(-1, 1, 1)
            std = std.reshape(-1, 1, 1)
            mins[k] = mins[k].reshape(-1, 1, 1)
            maxs[k] = maxs[k].reshape(-1, 1, 1)

        stats[k] = {
            "min": mins[k],
            "max": maxs[k],
            "mean": mean,
            "std": std,
            "counts": counts_frames[k]
        }
   
    return stats

def main(): 
    
    # Get path to dataset (via argparser)
    dataset_path = get_dataset_path()

    # Get episode indeces
    meta = LeRobotDatasetMetadata(repo_id=None, root=dataset_path)
    all_eps = list(meta.episodes.keys()) # list of episode indeces

    # Set output folder to save new stats
    output_path = Path(dataset_path) / "meta" / "episodes_stats_transformed.jsonl"

    # Get configs about dataloader and dataset 
    dataloader_cfg, dataset_cfg, transforms_cfg = get_configs_dataset("configs/dataset.yaml")
    
    # Override values for episode stats generation
    dataloader_cfg["batch_size"] = 1
    dataloader_cfg["N_history"] = 1
    dataloader_cfg["N_chunk"] = 1
    dataloader_cfg["fps_sampling_hist"] = meta.fps
    dataloader_cfg["fps_sampling_chunk"] = meta.fps
    dataloader_cfg["num_workers"] = 4
    dataloader_cfg["prefetch_factor"] = 2

    # Prepare transforms for computing dataset statistics only
    transforms_stats = CustomTransforms(
        dataloader_cfg=dataloader_cfg,
        dataset_cfg=dataset_cfg,
        transforms_cfg=transforms_cfg,
        train=False
    )

    # Build the delta_timestamps dict for LeRobotDataset history and future        
    delta_timestamps = build_delta_timestamps(dataset_cfg["features"], 
                                              dataloader_cfg["N_history"], 
                                              dataloader_cfg["N_chunk"], 
                                              meta.fps, 
                                              dataloader_cfg["fps_sampling_hist"], 
                                              dataloader_cfg["fps_sampling_chunk"])
    
    # select features to be used to compute statistics
    keep_keys = []
    for ft_type, ft_list in dataset_cfg["features"].items():
        for ft in ft_list:
            if (ft_type != "REMOVE"):
                keep_keys.append(ft)

    print("Compute dataset statistics for the following keys: ", keep_keys)

    # start timer
    t_start = time.perf_counter()

    for ep in all_eps:

        print(f"\nComputing dataset statistics for episode {ep} ...")

        # Load 1 episode at a time
        ep_data = LeRobotDatasetPatch(repo_id=None, 
                                      root=dataset_path, 
                                      delta_timestamps=delta_timestamps, 
                                      episodes=[ep])

        # Create transforms
        ep_data_transformed = TransformedDataset(ep_data, 
                                                 transforms_stats)

        # Create dataloader
        ep_loader = DataLoader(
            ep_data_transformed,
            batch_size=dataloader_cfg["batch_size"],
            shuffle=False,
            num_workers=dataloader_cfg["num_workers"],
            prefetch_factor=dataloader_cfg["prefetch_factor"],
            pin_memory=True,
        )

        # Compute stats per each transformed episode
        ep_stats = compute_episode_stats_streaming(ep_loader,
                                                   keep_keys,
                                                   dataset_cfg["features"],
                                                   dataloader_cfg["batch_size"],
                                                   dataloader_cfg["N_history"])
        
        print(ep_stats)

        # Save on file episode stats
        ep_stats = {"episode_index": ep, "stats": serialize_dict(ep_stats)}
        append_jsonlines(ep_stats, output_path)

    # stop timer
    t_elapsed = time.perf_counter() - t_start

    print("\nFinished computing dataset statistics!")
    print(f"Number of total episodes: {meta.total_episodes}")
    print(f"Number of total frames: {meta.total_frames}")
    print(f"Time elapsed: {t_elapsed:.4f}s")


if __name__ == "__main__":

    main()