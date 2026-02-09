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
    Build PyTorch DataLoaders for training and validation from a LeRobot dataset.

    This function constructs dataset instances and DataLoaders by:
      - Loading a LeRobot dataset from a local path or Hugging Face repository
      - Building temporal histories and future action chunks via delta timestamps
      - Subsampling observations and actions according to model sampling rates
      - Splitting episodes into training and validation sets
      - Creating optimized DataLoaders with reproducible worker behavior

    Args:
        repo_id (str, optional): Path or Hugging Face repository ID of the dataset.
        dataset_path (str, optional): Local root directory of the dataset.
        dataloader_cfg (dict): Configuration for DataLoader behavior.
        dataset_cfg (dict): Dataset configuration describing features and state/action layouts.
        model_cfg (dict): Model configuration specifying temporal parameters and sampling rates.
        selected_episodes (list[int], optional): Explicit list of episode indices to use for both training and validation.

    Returns:
        train_loader (DataLoader): DataLoader for training episodes.
        train_episodes (list[int]): List of episode indices used for training.
        val_loader (DataLoader): DataLoader for validation episodes.
        val_episodes (list[int]): List of episode indices used for validation.
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
        if num_train_episodes +  num_val_episodes > num_episodes:
            raise ValueError(f"num_train_episodes ({num_train_episodes}) + num_val_episodes ({num_val_episodes}) > num_episodes ({num_episodes}).")
        if num_train_episodes < 1 or num_val_episodes < 1:
            raise ValueError(f"Both num_train_episodes ({num_train_episodes}) and num_val_episodes ({num_val_episodes}) must be greater than zero.")
        train_episodes = episode_ids[:num_train_episodes] 
        val_episodes = episode_ids[num_train_episodes:num_train_episodes + num_val_episodes]
        print("Training episodes indeces: ", train_episodes)
        print("Validation episodes indeces: ", val_episodes)

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
        g = torch.Generator()
        g.manual_seed(seed_val)
    else:
        worker_fn = None   
        g = torch.Generator()

    # Wrap in DataLoaders

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers, # number of CPU processes that load data in parallel
        pin_memory=device.type!="cpu", # allocate CPU memory as pinned for faster CPU to GPU transfers
        prefetch_factor=prefetch_factor, # each worker preloads 2 batches ahead
        persistent_workers=False, # keep workers alive between epochs 
        worker_init_fn=worker_fn, # ensures each worker has a reproducible deterministic seed
        generator=g, # ensures shuffle=True has a reproducible deterministic seed
        drop_last=True # if the total number of samples is not divisible by batch_size, the last incomplete batch is dropped
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type!="cpu",
        prefetch_factor=prefetch_factor,
        persistent_workers=False,
        worker_init_fn=worker_fn,
        drop_last=True
    )

    return train_loader, train_episodes, val_loader, val_episodes