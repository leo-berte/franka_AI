from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import torch
import time


from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import append_jsonlines, serialize_dict

from franka_ai.dataset.transforms import CustomTransforms
from franka_ai.dataset.utils import get_configs_dataset, build_delta_timestamps, LeRobotDatasetPatch


"""
Run the code: 

python src/franka_ai/dataset/generate_updated_stats.py --dataset /mnt/Data/datasets/lerobot/single_outliers --config config1
python src/franka_ai/dataset/generate_updated_stats.py --dataset /workspace/data/today_data/today_outliers --config config1
"""

# TODO:
# 1) How to sample images to compute stats faster as in lerobot?


def parse_args():

    # set parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
        type=str,
        required=True,
        help="Absolute path to the dataset folder")
        
    parser.add_argument("--config", type=str, default="config",
                help="Config folder name")
    
    args = parser.parse_args()

    # return absolute path to the dataset and config folder name
    return args.dataset, args.config

def compute_episode_stats_streaming(dataloader, transforms, keep_keys, feature_groups, batch_size, N_history):

    mins = {}
    maxs = {}
    sums = {}
    sq_sums = {}
    counts_elems = {}
    counts_frames = {}
    stats = {}

    # extract first batch to infer feature shapes
    first_batch = next(iter(dataloader))

    # Apply custom transforms
    first_batch = transforms.transform(first_batch) 

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

        # Apply custom transforms
        batch = transforms.transform(batch) 

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
            "count": torch.tensor([counts_frames[k]], dtype=torch.int64)
        }
   
    return stats

def main(): 
    
    # Get path to dataset
    dataset_path, config_folder = parse_args()

    # Get episode indeces
    meta = LeRobotDatasetMetadata(repo_id=None, root=dataset_path)
    all_eps = list(meta.episodes.keys()) # list of episode indeces
    
    # Set output folder to save new stats
    output_path = Path(f"configs/{config_folder}") / "episodes_stats_transformed.jsonl"

    # Safety check: avoid silently appending to an existing file
    if output_path.exists():
        raise RuntimeError(
            f"ERROR: {output_path} already exists.\n"
            "Delete it before running stats generation to avoid corrupting the file."
        )

    # Get configs
    dataloader_cfg, dataset_cfg, transforms_cfg = get_configs_dataset(f"configs/{config_folder}/dataset.yaml")
    
    # Override values for episode stats generation
    
    dataloader_cfg["batch_size"] = 1
    dataloader_cfg["num_workers"] = 4
    dataloader_cfg["prefetch_factor"] = 2

    model_cfg_override = {}
    model_cfg_override.setdefault("params", {})
    model_cfg_override.setdefault("sampling", {})
    model_cfg_override["params"]["N_history"] = 1
    model_cfg_override["params"]["N_chunk"] = 1
    model_cfg_override["sampling"]["fps_sampling_hist"] = meta.fps
    model_cfg_override["sampling"]["fps_sampling_chunk"] = meta.fps

    # Prepare transforms for computing dataset statistics only
    transforms_stats = CustomTransforms(
        dataset_cfg=dataset_cfg,
        transforms_cfg=transforms_cfg,
        model_cfg=model_cfg_override,
        train=False
    )

    # Build the delta_timestamps dict for LeRobotDataset history and future        
    delta_timestamps = build_delta_timestamps(dataset_cfg["features"], 
                                              model_cfg_override["params"]["N_history"],
                                              model_cfg_override["params"]["N_chunk"],
                                              meta.fps, 
                                              model_cfg_override["sampling"]["fps_sampling_hist"],
                                              model_cfg_override["sampling"]["fps_sampling_chunk"])
    
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

        # Create dataloader
        ep_loader = DataLoader(
            ep_data,
            batch_size=dataloader_cfg["batch_size"],
            shuffle=False,
            num_workers=dataloader_cfg["num_workers"],
            prefetch_factor=dataloader_cfg["prefetch_factor"],
            pin_memory=True,
        )

        # Compute stats per each transformed episode
        ep_stats = compute_episode_stats_streaming(ep_loader,
                                                   transforms_stats,
                                                   keep_keys,
                                                   dataset_cfg["features"],
                                                   dataloader_cfg["batch_size"],
                                                   model_cfg_override["params"]["N_history"])
        
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