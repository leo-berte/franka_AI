from pathlib import Path
import argparse
import torch

from torch.utils.data import DataLoader

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import append_jsonlines, serialize_dict

from franka_ai.dataset.load_dataset import TransformedDataset
from franka_ai.dataset.transforms import CustomTransforms
from franka_ai.dataset.utils import get_configs_dataset, build_delta_timestamps, LeRobotDatasetPatch

"""
Run the code: python src/franka_ai/dataset/generate_updated_stats.py --dataset /home/leonardo/Documents/Coding/franka_AI/data/today_data/today
"""

# TODO:
# 1) loro fanno anche sampling delle immagini per calcolare stats piu velocmente, io come posso farlo reciclando loro funzione? funziona anche se uso dataloader?
# 2) capire come usare le due funzioni loro per rendere piu snella la mia (occhio a B, N_h dimensions)


# def get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
#     return {
#         "min": np.min(array, axis=axis, keepdims=keepdims),
#         "max": np.max(array, axis=axis, keepdims=keepdims),
#         "mean": np.mean(array, axis=axis, keepdims=keepdims),
#         "std": np.std(array, axis=axis, keepdims=keepdims),
#         "count": np.array([len(array)]),
#     }


# def compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray], features: dict) -> dict:
#     ep_stats = {}
#     for key, data in episode_data.items():
#         if features[key]["dtype"] == "string":
#             continue  # HACK: we should receive np.arrays of strings
#         elif features[key]["dtype"] in ["image", "video"]:
#             ep_ft_array = sample_images(data)  # data is a list of image paths
#             axes_to_reduce = (0, 2, 3)  # keep channel dim
#             keepdims = True
#         else:
#             ep_ft_array = data  # data is already a np.ndarray
#             axes_to_reduce = 0  # compute stats over the first axis
#             keepdims = data.ndim == 1  # keep as np.array

#         ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

#         # finally, we normalize and remove batch dim for images
#         if features[key]["dtype"] in ["image", "video"]:
#             ep_stats[key] = {
#                 k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
#             }

#     return ep_stats





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

def compute_episode_stats_patch(dataloader, feature_groups):

    print("\nComputing dataset statistics...")

    mins = {}
    maxs = {}
    sums = {}
    sq_sums = {}
    counts = {}
    stats = {}

    # extract first batch to infer feature shapes
    first_batch = next(iter(dataloader))

    # select features to be used to compute statistics
    keep_keys = []
    for ft_type, ft_list in feature_groups.items():
        for ft in ft_list:
            if (ft_type != "REMOVE"):
                keep_keys.append(ft)

    print(keep_keys)

    # initialize accumulators
    for k in keep_keys:
        
        # get features for a given key and collapse batch + history
        x = first_batch[k].detach().cpu().float()
        x_flat = x.reshape(-1, *x.shape[2:])  # (B*N_h, ...)

        if k in feature_groups["VISUAL"]:
            # x_flat: (B*N_h, C, H, W)
            BNH, C, H, W = x_flat.shape
            prototype = torch.zeros(C)  # stats per channel
        else:
            # x_flat: (B*N_h, D)
            prototype = torch.zeros(*x_flat.shape[1:])

        mins[k] = torch.full_like(prototype, float('inf'))
        maxs[k] = torch.full_like(prototype, float('-inf'))
        sums[k] = torch.zeros_like(prototype)
        sq_sums[k] = torch.zeros_like(prototype)
        counts[k] = 0

    cont = 0

    # Scan entire dataset
    for batch in dataloader:

        cont += 1

        if cont > 40:
            break

        for k in keep_keys:

            # get features for a given key and collapse batch + history
            x = batch[k].detach().cpu().float()   # (B, N_h, ...)
            x_flat = x.reshape(-1, *x.shape[2:])  # (B*N_h, ...)

            if k in feature_groups["VISUAL"]:
                # (B*N_h, C, H, W) → (C, total_pixels)
                BNH, C, H, W = x_flat.shape

                # reshape to (BNH, C, H*W)
                x_temp = x_flat.reshape(BNH, C, -1)

                # permute to (C, BNH, H*W)
                x_temp = x_temp.permute(1, 0, 2)

                # collapse into (C, BNH*H*W)
                x_vis = x_temp.reshape(C, -1)

                # update stats over dim=1 (pixels)
                batch_min = x_vis.min(dim=1).values
                batch_max = x_vis.max(dim=1).values
                batch_sum = x_vis.sum(dim=1)
                batch_sq_sum = (x_vis ** 2).sum(dim=1)
                batch_count = x_vis.shape[1]

            else:
                # Non-visual → (BNH, D)
                batch_min = x_flat.min(dim=0).values
                batch_max = x_flat.max(dim=0).values
                batch_sum = x_flat.sum(dim=0)
                batch_sq_sum = (x_flat ** 2).sum(dim=0)
                batch_count = x_flat.shape[0]

            # Accumulate global stats
            mins[k] = torch.minimum(mins[k], batch_min)
            maxs[k] = torch.maximum(maxs[k], batch_max)
            sums[k] += batch_sum
            sq_sums[k] += batch_sq_sum
            counts[k] += batch_count

    # Finalize statistics
    for k in keep_keys:

        total = counts[k]
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
            "counts": counts[k]
        }

    print("Finished computing dataset statistics!\n")

    print(stats)
    
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
    dataloader_cfg["N_history"] = 1
    dataloader_cfg["N_chunk"] = 1
    dataloader_cfg["fps_sampling_hist"] = meta.fps
    dataloader_cfg["fps_sampling_chunk"] = meta.fps

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

    for ep in all_eps:

        print(ep)

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
            batch_size=1,
            shuffle=False,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True,
        )

        # Compute stats per each transformed episode
        ep_stats = compute_episode_stats_patch(ep_loader, dataset_cfg["features"])

        # Save on file episode stats
        ep_stats = {"episode_index": ep, "stats": serialize_dict(ep_stats)}
        append_jsonlines(ep_stats, output_path)


if __name__ == "__main__":

    main()







