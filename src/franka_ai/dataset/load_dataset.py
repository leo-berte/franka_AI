from torch.utils.data import Dataset, DataLoader
from pprint import pprint
import torch
import random

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from franka_ai.dataset.utils import build_delta_timestamps, print_dataset_info, LeRobotDatasetPatch
from franka_ai.utils.seed_everything import make_worker_init_fn




def make_dataloader(
    repo_id=None, 
    dataset_path=None, 
    dataloader_cfg=None,
    dataset_cfg=None,
    model_cfg=None,
    selected_episodes=None
):
    
    """
    
    TO UPDATE
    
    Build PyTorch DataLoaders for training and validation from a LeRobot dataset.

    Handles:
    - Loading dataset from Hugging Face hub or local path
    - Building history and future frames (delta_timestamps)
    - Splitting into training and validation sets
    - Applying image and proprioceptive transformations
    - Creating optimized DataLoaders for training

    Args:
        repo_id: Hugging Face repo ID
        dataseth_path: local dataset root directory

    Returns:
        (DataLoader, DataLoader): train and validation dataloaders
    """

    # Extract parameters
    batch_size         = dataloader_cfg["batch_size"]
    shuffle            = dataloader_cfg["shuffle"]
    train_split_ratio  = dataloader_cfg["train_split_ratio"] 
    val_split_ratio    = dataloader_cfg["val_split_ratio"]
    num_workers        = dataloader_cfg["num_workers"]
    prefetch_factor    = dataloader_cfg["prefetch_factor"]
    seed_val           = dataloader_cfg["seed_val"]
    print_ds_info      = dataloader_cfg["print_ds_info"]
    device             = torch.device(dataloader_cfg["device"])

    # Extract parameters
    N_history          = model_cfg["params"].get("n_obs_steps") or model_cfg["params"].get("N_history")
    N_chunk            = model_cfg["params"].get("horizon")     or model_cfg["params"].get("chunk_size") or model_cfg["params"].get("N_chunk")
    fps_sampling_hist  = model_cfg["sampling"]["fps_sampling_hist"]
    fps_sampling_chunk = model_cfg["sampling"]["fps_sampling_chunk"]

    # Get dataset metadata
    dataset_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_path)

    # Build the delta_timestamps dict for LeRobotDataset history and future        
    delta_timestamps = build_delta_timestamps(dataset_cfg["features"], N_history, N_chunk, dataset_meta.fps, fps_sampling_hist, fps_sampling_chunk)

    # Generate random list of indeces covering all the episodes
    num_episodes = dataset_meta.total_episodes
    episode_ids = list(range(num_episodes))
    random.shuffle(episode_ids)
    
    # Consistency check
    if train_split_ratio +  val_split_ratio > 1.0:
        raise ValueError("train_split_ratio + val_split_ratio must not be greater than 1.0.")

    # Select episodes (manually or with train/val split ratio)
    if selected_episodes:
        train_episodes=selected_episodes
        val_episodes=selected_episodes
    else:
        num_train_episodes = int(num_episodes * train_split_ratio)
        num_val_episodes = int(num_episodes * val_split_ratio)
        train_episodes = episode_ids[:num_train_episodes] 
        val_episodes = episode_ids[num_train_episodes:num_train_episodes + num_val_episodes]

    # Load the raw dataset (hub or local)
    train_ds = LeRobotDatasetPatch(
        repo_id=repo_id,
        root=dataset_path,   
        delta_timestamps=delta_timestamps,
        episodes=train_episodes       
    )

    # Load the raw dataset (hub or local)
    val_ds = LeRobotDatasetPatch(
        repo_id=repo_id,
        root=dataset_path,   
        delta_timestamps=delta_timestamps,
        episodes=val_episodes
    )

    # print stats
    if print_ds_info:
        print_dataset_info(train_ds, "training dataset")
        print_dataset_info(val_ds, "validation dataset")

    # fix seed for reproducibility
    if seed_val is not None and seed_val >= 0:
        worker_fn = make_worker_init_fn(seed_val)
    else:
        worker_fn = None   

    # Wrap in DataLoaders

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers, # number of CPU processes that load data in parallel
        pin_memory=device.type!="cpu", # allocate CPU memory as pinned for faster CPU to GPU transfers
        prefetch_factor=prefetch_factor, # each worker preloads 2 batches ahead
        persistent_workers=True, # keep workers alive between epochs 
        worker_init_fn=worker_fn, # ensures each worker has a reproducible deterministic seed
        drop_last=True # if the total number of samples is not divisible by batch_size, the last incomplete batch is dropped
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type!="cpu",
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        worker_init_fn=worker_fn,
        drop_last=True            
    )

    return train_loader, train_episodes, val_loader, val_episodes